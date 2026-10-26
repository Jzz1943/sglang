"""Optional all-gather/GEMM overlap for interleave CP linear attention."""

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel


def kda_cp_async_tp_enabled() -> bool:
    if not envs.SGLANG_CP_KDA_ASYNC_TP.get():
        return False
    parallel = get_parallel()
    return (
        parallel.enable_prefill_cp
        and parallel.enable_cp_tp_group_sharing
        and parallel.cp_strategy == "interleave"
        and parallel.attn_cp_size == parallel.tp_size
        and parallel.attn_cp_size > 1
        and parallel.attn_dp_size == 1
    )


def try_kda_cp_async_projection(shard: torch.Tensor, linear):
    """Return projected rank-major rows, or None before issuing any collective.

    Captured/compiled paths keep the ordinary gather and projection. Eligibility
    uses equal physical shard shapes, so every CP rank makes the same decision.
    """
    if (
        torch.compiler.is_compiling()
        or not shard.is_cuda
        or shard.ndim != 2
        or shard.dtype != torch.bfloat16
        or shard.shape[0] < 4096
        or not shard.is_contiguous()
    ):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    weight = linear.weight
    if (
        not isinstance(linear.quant_method, UnquantizedLinearMethod)
        or getattr(linear, "bias", None) is not None
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or weight.shape[1] != shard.shape[1]
        or not weight.is_contiguous()
    ):
        return None
    # Register the same Torch symmetric-memory operator used by layernorm SP.
    from torch.distributed import _symmetric_memory  # noqa: F401

    if not hasattr(torch.ops.symm_mem, "fused_all_gather_matmul"):
        return None
    group = get_parallel().attn_cp_group.device_group
    _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
        shard, [weight.t()], 0, group.group_name, return_A=False
    )
    return outputs[0]
