"""Focused wire-format and eligibility coverage for opt-in CP FP8 transport."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.cp import moe_fp8_transport as transport
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_transport_requires_opt_in_and_caches_loaded_weight_eligibility():
    with patch.object(transport, "_ENABLED", False):
        assert not transport.cp_moe_fp8_transport_enabled()
    check = Mock(return_value=(True, "eligible"))
    layer = SimpleNamespace(
        _cp_fp8_transport_eligible=None, _compute_moe_quant_once_enabled=check
    )
    assert transport.cp_moe_fp8_layer_eligible(layer)
    assert transport.cp_moe_fp8_layer_eligible(layer)
    check.assert_called_once_with(
        ignore_env=True, allow_deep_gemm=True, allow_fused_shared=True
    )


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"moe_ep_size": 4},
        {"attn_dp_size": 2},
        {"cp_strategy": "zigzag"},
        {"pp_size": 2},
    ],
)
def test_transport_topology_guard(change):
    fields = dict(
        enable_prefill_cp=True,
        enable_cp_tp_group_sharing=True,
        cp_strategy="interleave",
        attn_cp_size=4,
        tp_size=4,
        attn_dp_size=1,
    )
    fields.update(change)
    with (
        patch.object(transport, "_ENABLED", True),
        patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(**fields),
        ),
        patch(
            "sglang.srt.layers.moe.utils.get_moe_runner_backend",
            return_value=SimpleNamespace(
                is_triton=lambda: False, is_deep_gemm=lambda: True
            ),
        ),
        patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=SimpleNamespace(value="none"),
        ),
    ):
        assert transport.cp_moe_fp8_transport_enabled() == (not change)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("rows,hidden,topk", [(0, 4096, 9), (17, 4096, 9), (4, 128, 2)])
def test_wire_roundtrip_preserves_bits(fused, rows, hidden, topk):
    # Arbitrary FP8 bytes also exercise NaN and signed zero encodings.
    q = torch.randint(256, (rows, hidden), device="cuda", dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    scale = torch.rand(hidden // 128, (rows + 3) // 4 * 4, device="cuda").t()[:rows]
    weights = torch.rand(rows, topk, device="cuda")
    ids = torch.randint(289, (rows, topk), device="cuda", dtype=torch.int32)
    if rows:
        ids[-1].fill_(-1)
    with patch.object(transport, "_FUSED_PACKET", fused):
        payload = transport.pack_fp8_moe_payload(q, scale, weights, ids)
        actual = transport.unpack_fp8_moe_payload(payload, hidden, topk)
    assert payload.shape == (
        rows,
        ((hidden + hidden // 128 * 4 + topk * 8 + 15) // 16) * 16,
    )
    for before, after in zip((q, scale, weights, ids), actual, strict=True):
        assert torch.equal(
            before.reshape(-1).contiguous().view(torch.uint8),
            after.reshape(-1).contiguous().view(torch.uint8),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x"]))
