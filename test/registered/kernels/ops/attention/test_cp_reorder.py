"""Exact CP permutation and padding checks, including CUDA Graph replay."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "cp_size,rows,width,dtype",
    [
        (1, 13, 9, torch.float32),
        (2, 7, 17, torch.float32),
        (4, 9, 31, torch.bfloat16),
        (4, 8192, 6144, torch.bfloat16),
        (4, 0, 17, torch.bfloat16),
        (4, 7, 0, torch.bfloat16),
    ],
)
def test_cp_reorder_padding_and_graph(cp_size, rows, width, dtype):
    from sglang.srt.layers.cp.interleave import (
        InterleaveContextParallelMetadata,
        cp_interleave_to_sequence_order,
        cp_sequence_to_interleave_order,
    )

    capacity = (rows + cp_size - 1) // cp_size * cp_size
    sequence = torch.randn(rows, width, device="cuda", dtype=dtype)
    padded = torch.zeros(capacity, width, device="cuda", dtype=dtype)
    padded[:rows] = sequence
    rank_major = torch.cat([padded[rank::cp_size] for rank in range(cp_size)])
    metadata = InterleaveContextParallelMetadata(total_seq_lens=rows)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
        attn_cp_metadata=metadata,
    )
    with patch(
        "sglang.srt.layers.cp.interleave.get_parallel",
        return_value=SimpleNamespace(attn_cp_size=cp_size),
    ):
        restored = cp_interleave_to_sequence_order(rank_major, batch)
        actual = cp_sequence_to_interleave_order(sequence, batch, capacity)
        torch.testing.assert_close(restored, sequence, rtol=0, atol=0)
        torch.testing.assert_close(actual, rank_major, rtol=0, atol=0)
        if not rank_major.numel():
            return

        # Graphs keep physical capacity, including inactive logical rows.
        metadata.graph_num_tokens = capacity
        restored = cp_interleave_to_sequence_order(rank_major, batch)
        cp_sequence_to_interleave_order(restored, batch, capacity)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            restored = cp_interleave_to_sequence_order(rank_major, batch)
            actual = cp_sequence_to_interleave_order(restored, batch, capacity)
        rank_major.add_(1)
        graph.replay()
        expected = rank_major.reshape(cp_size, -1, width).transpose(0, 1).flatten(0, 1)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, rank_major, rtol=0, atol=0)
        graph.reset()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x"]))
