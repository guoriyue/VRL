"""Quantization schemes for model GEMMs (rollout-side).

The home for low-precision linear kernels and the module-tree swaps that install
them. Today it holds the fp8-e4m3 dynamic-quantization linear used by the
rollout DiT — ``rowwise``/``tensorwise`` on torch ``_scaled_mm``, and a
``blockwise`` recipe that **reuses vLLM's triton block kernel** rather than
hand-rolling — plus the NVFP4 sibling with two-level 1x16 scaling. int8 and
future schemes land here as siblings. The package ``__init__`` is the public
facade; consumers import from ``vrl.nn.quantization``, not per-scheme modules.
"""

from __future__ import annotations

from vrl.nn.quantization.base import QuantizedLinear, drop_quantized_masters
from vrl.nn.quantization.fp4 import Fp4Linear, nvfp4_available, swap_linears_to_nvfp4
from vrl.nn.quantization.fp8 import (
    Fp8Linear,
    swap_linears_to_fp8,
)
from vrl.nn.quantization.targeting import DEFAULT_EXCLUDE, LM_EXCLUDE, LinearTargetProfile

__all__ = [
    "DEFAULT_EXCLUDE",
    "LM_EXCLUDE",
    "Fp4Linear",
    "Fp8Linear",
    "LinearTargetProfile",
    "QuantizedLinear",
    "drop_quantized_masters",
    "nvfp4_available",
    "swap_linears_to_fp8",
    "swap_linears_to_nvfp4",
]
