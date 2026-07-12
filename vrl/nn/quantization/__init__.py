"""Quantization schemes for model GEMMs (rollout-side).

The home for low-precision linear kernels and the module-tree swap that installs
them. Today it
holds the fp8-e4m3 dynamic-quantization linear used by the
rollout DiT — ``rowwise``/``tensorwise`` on torch ``_scaled_mm``, and a
``blockwise`` recipe that **reuses vLLM's triton block kernel** rather than
hand-rolling. fp4 / int8 / other schemes land here as siblings. The package
``__init__`` is the public facade — consumers import from ``vrl.nn.quantization``,
not the per-scheme module.
"""

from __future__ import annotations

from vrl.nn.quantization.base import QuantizedLinear, drop_quantized_masters
from vrl.nn.quantization.fp8 import (
    DEFAULT_EXCLUDE,
    LM_EXCLUDE,
    Fp8Linear,
    swap_linears_to_fp8,
    vllm_block_fp8_available,
)

__all__ = [
    "DEFAULT_EXCLUDE",
    "LM_EXCLUDE",
    "Fp8Linear",
    "QuantizedLinear",
    "drop_quantized_masters",
    "swap_linears_to_fp8",
    "vllm_block_fp8_available",
]
