"""Experimental CP transport of FP8 activations and exact local router results.

Only GLM's standard TP MoE with audited block-128 FP8 shared/routed
experts is eligible. Residual layout, expert weights and output reduction stay
with the ordinary layer boundary. The default is off.
"""

import logging

import torch

from sglang.srt.environ import envs

_ENABLED = envs.SGLANG_CP_MOE_FP8_TRANSPORT.get()
_FUSED_PACKET = envs.SGLANG_CP_FP8_FUSED_PACKET.get()
_LAYERS = {}
_LOGGED = False
logger = logging.getLogger(__name__)


def cp_moe_fp8_transport_enabled():
    if not _ENABLED:
        return False
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    return (
        p.enable_prefill_cp
        and p.enable_cp_tp_group_sharing
        and p.cp_strategy == "interleave"
        and p.attn_cp_size == p.tp_size
        and p.attn_cp_size > 1
        and p.attn_dp_size == 1
        and getattr(p, "moe_dp_size", 1) == 1
        and getattr(p, "moe_ep_size", 1) == 1
        and getattr(p, "pp_size", 1) == 1
        and get_moe_a2a_backend().value == "none"
        and (
            get_moe_runner_backend().is_triton()
            or get_moe_runner_backend().is_deep_gemm()
        )
    )


def register_cp_moe_fp8_layer(layer):
    from sglang.srt.runtime_context import get_exec

    layer._cp_fp8_transport_eligible = False
    if (
        not cp_moe_fp8_transport_enabled()
        or layer.is_nextn
        or layer.reduce_results
        or layer.moe_ep_size != 1
        or getattr(layer, "is_hash", False)
        or layer.gate.e_score_correction_bias_vl is not None
        or get_exec().moe.enable_eplb
    ):
        return
    # The quant runner is finalized after weight loading, not at construction.
    layer._cp_fp8_transport_eligible = None
    _LAYERS[layer.layer_id] = layer


def cp_moe_fp8_layer_eligible(layer):
    if layer._cp_fp8_transport_eligible is None:
        eligible, reason = layer._compute_moe_quant_once_enabled(
            ignore_env=True, allow_deep_gemm=True, allow_fused_shared=True
        )
        layer._cp_fp8_transport_eligible = eligible
        if not eligible:
            logger.info(
                "CP FP8 MoE layer %s uses BF16 fallback: %s", layer.layer_id, reason
            )
    return layer._cp_fp8_transport_eligible


def pack_fp8_moe_payload(q, scale, weights, ids):
    """One byte all-gather instead of separate activation/scale/routing gathers."""
    assert q.dtype == torch.float8_e4m3fn
    assert scale.dtype == weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    rows, hidden = q.shape
    assert hidden % 128 == 0
    assert scale.shape == (rows, hidden // 128)
    assert weights.shape == ids.shape and weights.shape[0] == rows
    if _FUSED_PACKET:
        from sglang.kernels.ops.moe.cp_fp8_packet import pack

        return pack(q, scale, weights, ids)

    def row_bytes(tensor):
        # A size-one final dimension may be "contiguous" with stride != 1.
        # Flatten first to make dtype reinterpretation valid in that case.
        return (
            tensor.reshape(-1)
            .contiguous()
            .view(torch.uint8)
            .reshape(rows, tensor.shape[1] * tensor.element_size())
        )

    pieces = [
        q.view(torch.uint8),
        row_bytes(scale),
        row_bytes(weights),
        row_bytes(ids),
    ]
    row_width = hidden + scale.shape[1] * 4 + weights.shape[1] * 8
    padding = (-row_width) % 16
    if padding:
        pieces.append(q.new_zeros((rows, padding), dtype=torch.uint8))
    return torch.cat(pieces, dim=1)


def unpack_fp8_moe_payload(payload, hidden, topk):
    if _FUSED_PACKET:
        from sglang.kernels.ops.moe.cp_fp8_packet import unpack

        return unpack(payload, hidden, topk)
    rows = payload.shape[0]
    scale_end = hidden + hidden // 128 * 4
    weights_end = scale_end + topk * 4
    ids_end = weights_end + topk * 4
    assert payload.shape[1] == (ids_end + 15) // 16 * 16
    q = payload[:, :hidden].contiguous().view(torch.float8_e4m3fn)
    # Shared FP8 linear expects column-major FP32 scales; routed GEMM accepts
    # the same strides through the existing pre-quantized-input contract.
    scale = payload[:, hidden:scale_end].contiguous().view(torch.float32)
    scale = scale.t().contiguous().t()
    weights = payload[:, scale_end:weights_end].contiguous().view(torch.float32)
    ids = payload[:, weights_end:ids_end].contiguous().view(torch.int32)
    assert q.shape == (rows, hidden)
    return q, scale, weights, ids


@torch.library.custom_op(
    "sglang::cp_moe_fp8_transport", mutates_args=(), device_types="cuda"
)
def cp_moe_fp8_transport(
    local_hidden: torch.Tensor, layer_id: int, cp_size: int
) -> torch.Tensor:
    global _LOGGED
    from sglang.kernels.ops.quantization.per_token_group_quant import (
        per_token_group_quant,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        maybe_fuse_routed_scale_and_shared_add,
    )
    from sglang.srt.runtime_context import get_parallel

    layer = _LAYERS[layer_id]
    # Router consumes the ORIGINAL BF16 input, never dequantized FP8 values.
    local_logits = layer.gate(local_hidden)
    topk = layer.topk(local_hidden, local_logits, num_token_non_padded=None)
    assert TopKOutputChecker.format_is_standard(topk)
    if layer.experts.quant_method.runner.runner_backend.is_deep_gemm():
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8_row_padded,
        )

        q, scale = sglang_per_token_group_quant_fp8_row_padded(local_hidden, 128)
        # Quantization may pad rows for GEMM alignment; routing and CP transport
        # describe logical tokens only. Keep the padded scale storage strides.
        q, scale = q[: local_hidden.shape[0]], scale[: local_hidden.shape[0]]
    else:
        q, scale = per_token_group_quant(local_hidden, group_size=128)
    payload = pack_fp8_moe_payload(q, scale, topk.topk_weights, topk.topk_ids)
    gathered = payload.new_empty((payload.shape[0] * cp_size, payload.shape[1]))
    get_parallel().attn_cp_group.all_gather_into_tensor(gathered, payload)
    q, scale, weights, ids = unpack_fp8_moe_payload(
        gathered, local_hidden.shape[1], topk.topk_ids.shape[1]
    )
    # All consumers below use pre_quant; this buffer supplies only shape/dtype
    # and, if inplace=True, the routed output storage. It is never an operand.
    shape_only = local_hidden.new_empty((q.shape[0], q.shape[1]))
    shared = layer._forward_shared_experts(shape_only, pre_quant_input=(q, scale))
    routed = layer.experts.forward_impl(
        shape_only,
        StandardTopKOutput(weights, ids, None),
        pre_quant_input=(q, scale),
    )
    output = maybe_fuse_routed_scale_and_shared_add(
        layer.experts,
        routed,
        None if layer._shared_expert_tp1 else shared,
        layer.routed_scaling_factor,
    )
    if (
        shared is not None
        and layer._shared_expert_tp1
        and layer._adds_replicated_output()
    ):
        output += shared
    if not _LOGGED:
        logger.info(
            "CP FP8 MoE transport: local_rows=%s, payload_row_bytes=%s, BF16_row_bytes=%s",
            local_hidden.shape[0],
            payload.shape[1],
            local_hidden.shape[1] * 2,
        )
        _LOGGED = True
    return output


@cp_moe_fp8_transport.register_fake
def _cp_moe_fp8_transport_fake(local_hidden, layer_id, cp_size):
    return local_hidden.new_empty(
        (local_hidden.shape[0] * cp_size, local_hidden.shape[1])
    )
