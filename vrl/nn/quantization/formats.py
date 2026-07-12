"""Numeric limits shared by low-precision quantization formats."""

from __future__ import annotations

from itertools import pairwise

# Maximum finite magnitude of the E4M3 scale format used by both FP8 GEMMs and
# NVFP4's per-block decode scales.
FP8_E4M3_MAX = 448.0

# NVFP4 protocol values and the additional shape constraints of the packed
# scaled-mm backend. Keeping them here gives the CPU reference, Triton kernel,
# module gate, and tests one source of truth.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = E2M1_VALUES[-1]
E2M1_MIDPOINTS = tuple((lower + upper) / 2 for lower, upper in pairwise(E2M1_VALUES))
E2M1_TIES_ROUND_UP = tuple(
    midpoint for lower_code, midpoint in enumerate(E2M1_MIDPOINTS) if lower_code & 1
)
NVFP4_BLOCK_SIZE = 16
NVFP4_K_ALIGNMENT = 2 * NVFP4_BLOCK_SIZE
NVFP4_N_ALIGNMENT = 16
NVFP4_SCALE_ROW_TILE = 128
NVFP4_SCALE_COL_TILE = 4
NVFP4_SCALE_INNER_ROWS = 32


__all__ = [
    "E2M1_MAX",
    "E2M1_MIDPOINTS",
    "E2M1_TIES_ROUND_UP",
    "E2M1_VALUES",
    "FP8_E4M3_MAX",
    "NVFP4_BLOCK_SIZE",
    "NVFP4_K_ALIGNMENT",
    "NVFP4_N_ALIGNMENT",
    "NVFP4_SCALE_COL_TILE",
    "NVFP4_SCALE_INNER_ROWS",
    "NVFP4_SCALE_ROW_TILE",
]
