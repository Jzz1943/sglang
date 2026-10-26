"""Single-pass conversion between rank-major and interleaved token order."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["src_rows", "local_rows"])
def _cp_reorder_kernel(
    src,
    dst,
    src_rows,
    local_rows,
    WIDTH: tl.constexpr,
    CP_SIZE: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if INVERSE:
        src_row = (row % local_rows) * CP_SIZE + row // local_rows
    else:
        src_row = (row % CP_SIZE) * local_rows + row // CP_SIZE
    valid = cols < WIDTH
    value = tl.load(src + src_row * WIDTH + cols, valid & (src_row < src_rows), 0)
    tl.store(dst + row * WIDTH + cols, value, valid)


def cp_reorder(
    hidden_states: torch.Tensor,
    cp_size: int,
    output_rows: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Copy a contiguous 2D tensor to the other CP order, zeroing absent rows.

    Forward consumes equal-sized rank-major shards and removes tail padding.
    Inverse consumes sequence order and restores padded rank-major shards.
    Callers with per-request gather indices must use their indexed path instead.
    """
    assert hidden_states.is_cuda and hidden_states.ndim == 2
    assert hidden_states.is_contiguous() and cp_size > 0 and output_rows >= 0
    source_rows, width = hidden_states.shape
    gathered_rows = output_rows if inverse else source_rows
    assert gathered_rows % cp_size == 0
    if inverse:
        assert source_rows <= output_rows
    else:
        assert output_rows <= source_rows
    output = hidden_states.new_empty((output_rows, width))
    if output.numel() == 0:
        return output
    _cp_reorder_kernel[(output_rows, triton.cdiv(width, 1024))](
        hidden_states,
        output,
        source_rows,
        gathered_rows // cp_size,
        width,
        cp_size,
        inverse,
        1024,
        num_warps=4,
    )
    return output
