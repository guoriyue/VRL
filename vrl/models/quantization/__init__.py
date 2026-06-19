"""Quantization schemes for model GEMMs (rollout-side).

The home for low-precision linear kernels and the module-tree swap that installs
them, mirroring vLLM's ``model_executor/layers/quantization`` and slime's fp8
tooling. Today it holds the fp8-e4m3 dynamic-quantization linear used by the
rollout DiT; fp4 / int8 / other schemes land here as siblings. The package
``__init__`` is the public facade — consumers import from ``vrl.models.quantization``,
not the per-scheme module.
"""

from __future__ import annotations

from vrl.models.quantization.fp8 import (
    DEFAULT_EXCLUDE,
    Fp8Linear,
    swap_linears_to_fp8,
)

__all__ = ["DEFAULT_EXCLUDE", "Fp8Linear", "swap_linears_to_fp8"]
