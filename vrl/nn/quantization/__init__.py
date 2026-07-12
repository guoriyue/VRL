"""Quantization schemes for model GEMMs (rollout-side).

The home for low-precision linear kernels and the module-tree swap that installs
them. Today it holds the fp8-e4m3 dynamic-quantization linear used by the
rollout DiT — ``rowwise``/``tensorwise`` on torch ``_scaled_mm``, and a
``blockwise`` recipe that **reuses vLLM's triton block kernel** rather than
hand-rolling — plus the nvfp4 sibling (``Fp4Linear``, two-level 1x16 block
scaling on torch ``_scaled_mm``). int8 / other schemes land here as further
siblings. The package ``__init__`` is the public facade — consumers import from
``vrl.nn.quantization``, not the per-scheme module.
"""

from __future__ import annotations

from vrl.nn.quantization.base import QuantizedLinear, drop_quantized_masters
from vrl.nn.quantization.fp4 import Fp4Linear, nvfp4_available, swap_linears_to_fp4
from vrl.nn.quantization.fp8 import (
    Fp8Linear,
    swap_linears_to_fp8,
    vllm_block_fp8_available,
)
from vrl.nn.quantization.targeting import DEFAULT_EXCLUDE, LM_EXCLUDE, LinearTargetProfile


def drop_fp8_masters(root):
    """Compatibility facade for the scheme-neutral master-drop operation.

    New callers should use :func:`drop_quantized_masters`. Keeping this public
    name avoids breaking integrations written before FP4 generalized the
    ownership contract.
    """

    return drop_quantized_masters(root)


__all__ = [
    "DEFAULT_EXCLUDE",
    "LM_EXCLUDE",
    "Fp4Linear",
    "Fp8Linear",
    "LinearTargetProfile",
    "QuantizedLinear",
    "drop_fp8_masters",
    "drop_quantized_masters",
    "nvfp4_available",
    "swap_linears_to_fp4",
    "swap_linears_to_fp8",
    "vllm_block_fp8_available",
]
