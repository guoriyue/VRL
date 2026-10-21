"""Fused CUDA quantization kernels for NVFP4 rollout linears."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vrl.nn.quantization.formats import (
    E2M1_MAX,
    E2M1_MIDPOINTS,
    E2M1_TIES_ROUND_UP,
    FP8_E4M3_MAX,
    NVFP4_BLOCK_SIZE,
    NVFP4_SCALE_COL_TILE,
    NVFP4_SCALE_INNER_ROWS,
    NVFP4_SCALE_ROW_TILE,
)


@triton.jit
def _e2m1_candidate(
    values,
    block_amax,
    tensor_scale,
    target,
    FP4_MAX: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BOUND_0: tl.constexpr,
    BOUND_1: tl.constexpr,
    BOUND_2: tl.constexpr,
    BOUND_3: tl.constexpr,
    BOUND_4: tl.constexpr,
    BOUND_5: tl.constexpr,
    BOUND_6: tl.constexpr,
    TIE_UP_0: tl.constexpr,
    TIE_UP_1: tl.constexpr,
    TIE_UP_2: tl.constexpr,
):
    """E4M3 block scale (as fp32), signed E2M1 codes and squared error for one target."""

    block_scale = tl.clamp(block_amax / target / tensor_scale, 1.0e-12, FP8_MAX)
    block_scale = block_scale.to(tl.float8e4nv).to(tl.float32)
    effective_scale = block_scale * tensor_scale
    scaled = tl.where(
        effective_scale[:, None] > 0.0,
        values / effective_scale[:, None],
        0.0,
    )
    scaled = tl.clamp(scaled, -FP4_MAX, FP4_MAX)
    magnitude = tl.abs(scaled)
    # Strict comparisons select the lower code at a midpoint; advance only odd
    # lower codes to implement round-to-nearest-even.
    codes = (
        (magnitude > BOUND_0).to(tl.int32)
        + (magnitude > BOUND_1).to(tl.int32)
        + (magnitude > BOUND_2).to(tl.int32)
        + (magnitude > BOUND_3).to(tl.int32)
        + (magnitude > BOUND_4).to(tl.int32)
        + (magnitude > BOUND_5).to(tl.int32)
        + (magnitude > BOUND_6).to(tl.int32)
    )
    codes += ((magnitude == TIE_UP_0) | (magnitude == TIE_UP_1) | (magnitude == TIE_UP_2)).to(
        tl.int32
    )
    # E2M1 grid 0, 0.5, 1, 1.5, 2, 3, 4, 6: half the code up to 2.0, then 3, 4, 6.
    grid = tl.where(codes <= 4, 0.5 * codes, tl.where(codes == 7, 6.0, codes - 2.0))
    dequantized = tl.where(scaled < 0.0, -grid, grid) * effective_scale[:, None]
    error = tl.sum((dequantized - values) * (dequantized - values), axis=1)
    return block_scale, codes + tl.where(scaled < 0.0, 8, 0), error


@triton.jit
def _quantize_nvfp4_kernel(
    input_ptr,
    packed_ptr,
    scale_ptr,
    tensor_scale_ptr,
    K: tl.constexpr,
    N_COL_BLOCKS: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BOUND_0: tl.constexpr,
    BOUND_1: tl.constexpr,
    BOUND_2: tl.constexpr,
    BOUND_3: tl.constexpr,
    BOUND_4: tl.constexpr,
    BOUND_5: tl.constexpr,
    BOUND_6: tl.constexpr,
    FP4_MAX: tl.constexpr,
    FP8_MAX: tl.constexpr,
    SCALE_COL_TILE: tl.constexpr,
    SCALE_INNER_ROWS: tl.constexpr,
    SCALE_ROW_TILE: tl.constexpr,
    TIE_UP_0: tl.constexpr,
    TIE_UP_1: tl.constexpr,
    TIE_UP_2: tl.constexpr,
):
    """Quantize one row batch and write cuBLAS-swizzled E4M3 scales."""

    row = tl.program_id(0)
    batch = tl.program_id(1)
    block_ids = batch * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    valid_blocks = block_ids < K // BLOCK_SIZE
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = block_ids[:, None] * BLOCK_SIZE + lanes[None, :]
    values = tl.load(
        input_ptr + row * K + offsets,
        mask=valid_blocks[:, None],
        other=0.0,
    ).to(tl.float32)

    tensor_scale = tl.load(tensor_scale_ptr)
    block_amax = tl.max(tl.abs(values), axis=1)
    # Four over six: quantize each block with its amax mapped onto 6 and onto 4
    # and keep the smaller squared error (a tie keeps 6, as the CPU reference).
    scale_six, codes_six, error_six = _e2m1_candidate(
        values, block_amax, tensor_scale, FP4_MAX, FP4_MAX, FP8_MAX,
        BOUND_0, BOUND_1, BOUND_2, BOUND_3, BOUND_4, BOUND_5, BOUND_6,
        TIE_UP_0, TIE_UP_1, TIE_UP_2,
    )  # fmt: skip
    scale_four, codes_four, error_four = _e2m1_candidate(
        values, block_amax, tensor_scale, 4.0, FP4_MAX, FP8_MAX,
        BOUND_0, BOUND_1, BOUND_2, BOUND_3, BOUND_4, BOUND_5, BOUND_6,
        TIE_UP_0, TIE_UP_1, TIE_UP_2,
    )  # fmt: skip
    four = error_four < error_six
    block_scale_e4m3 = tl.where(four, scale_four, scale_six).to(tl.float8e4nv)
    codes = tl.where(four[:, None], codes_four, codes_six)

    # Logical scale [row, block_id] maps into cuBLAS's 128x4 blocked layout.
    row_in_tile = row % SCALE_ROW_TILE
    swizzled_row_width = SCALE_ROW_TILE // SCALE_INNER_ROWS * SCALE_COL_TILE
    scale_offsets = (
        ((row // SCALE_ROW_TILE) * N_COL_BLOCKS + block_ids // SCALE_COL_TILE)
        * (SCALE_ROW_TILE * SCALE_COL_TILE)
        + (row % SCALE_INNER_ROWS) * swizzled_row_width
        + (row_in_tile // SCALE_INNER_ROWS) * SCALE_COL_TILE
        + block_ids % SCALE_COL_TILE
    )
    tl.store(scale_ptr + scale_offsets, block_scale_e4m3, mask=valid_blocks)

    paired = tl.reshape(codes, (BLOCKS_PER_PROGRAM, BLOCK_SIZE // 2, 2))
    low, high = tl.split(paired)
    packed = low | (high << 4)
    packed_offsets = (
        block_ids[:, None] * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)[None, :]
    )
    tl.store(
        packed_ptr + row * (K // 2) + packed_offsets,
        packed,
        mask=valid_blocks[:, None],
    )


def quantize_nvfp4_cuda(
    tensor: torch.Tensor,
    tensor_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous CUDA matrix into packed FP4 and swizzled scales."""

    if tensor.ndim != 2 or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("fused nvfp4 quantization requires a contiguous CUDA matrix")
    rows, k = tensor.shape
    if k % NVFP4_BLOCK_SIZE:
        raise ValueError(f"nvfp4 needs K % {NVFP4_BLOCK_SIZE} == 0, got K={k}")
    blocks_per_program = 16
    scale_cols = k // NVFP4_BLOCK_SIZE
    padded_rows = triton.cdiv(rows, NVFP4_SCALE_ROW_TILE) * NVFP4_SCALE_ROW_TILE
    padded_cols = triton.cdiv(scale_cols, NVFP4_SCALE_COL_TILE) * NVFP4_SCALE_COL_TILE
    packed = torch.empty((rows, k // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.zeros(
        padded_rows * padded_cols,
        device=tensor.device,
        dtype=torch.float8_e4m3fn,
    )
    grid = (rows, triton.cdiv(scale_cols, blocks_per_program))
    _quantize_nvfp4_kernel[grid](
        tensor,
        packed,
        scales,
        tensor_scale,
        K=k,
        N_COL_BLOCKS=triton.cdiv(scale_cols, NVFP4_SCALE_COL_TILE),
        BLOCKS_PER_PROGRAM=blocks_per_program,
        BLOCK_SIZE=NVFP4_BLOCK_SIZE,
        BOUND_0=E2M1_MIDPOINTS[0],
        BOUND_1=E2M1_MIDPOINTS[1],
        BOUND_2=E2M1_MIDPOINTS[2],
        BOUND_3=E2M1_MIDPOINTS[3],
        BOUND_4=E2M1_MIDPOINTS[4],
        BOUND_5=E2M1_MIDPOINTS[5],
        BOUND_6=E2M1_MIDPOINTS[6],
        FP4_MAX=E2M1_MAX,
        FP8_MAX=FP8_E4M3_MAX,
        SCALE_COL_TILE=NVFP4_SCALE_COL_TILE,
        SCALE_INNER_ROWS=NVFP4_SCALE_INNER_ROWS,
        SCALE_ROW_TILE=NVFP4_SCALE_ROW_TILE,
        TIE_UP_0=E2M1_TIES_ROUND_UP[0],
        TIE_UP_1=E2M1_TIES_ROUND_UP[1],
        TIE_UP_2=E2M1_TIES_ROUND_UP[2],
        num_warps=4,
    )
    return packed.view(torch.float4_e2m1fn_x2), scales


__all__ = ["quantize_nvfp4_cuda"]
