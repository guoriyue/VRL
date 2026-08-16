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
from vrl.nn.quantization.fp4 import Fp4Linear, nvfp4_available
from vrl.nn.quantization.fp8 import Fp8Linear
from vrl.nn.quantization.targeting import DEFAULT_EXCLUDE, LM_EXCLUDE, LinearTargetProfile

# Config ``precision.rollout.quantization.format`` -> the scheme that implements
# it. Derived from each class's own ``quantization_scheme`` so the mapping cannot
# disagree with the identity the runtime guard reads off a swapped module; adding
# a scheme means adding it to this tuple and nothing else.
QUANTIZATION_SCHEMES: dict[str, type[QuantizedLinear]] = {
    scheme.quantization_scheme: scheme for scheme in (Fp8Linear, Fp4Linear)
}

__all__ = [
    "DEFAULT_EXCLUDE",
    "LM_EXCLUDE",
    "QUANTIZATION_SCHEMES",
    "Fp4Linear",
    "Fp8Linear",
    "LinearTargetProfile",
    "QuantizedLinear",
    "drop_quantized_masters",
    "nvfp4_available",
]
