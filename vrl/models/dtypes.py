"""Canonical torch dtype parsing for the model-loading boundary.

One place that maps the tolerant set of dtype spellings the codebase accepts at
config/checkpoint boundaries — ``bf16``/``bfloat16``/``half``/``torch.float16``
etc. — onto ``torch.dtype`` values, and back to canonical config strings. The
config layer (:mod:`vrl.config.precision`) speaks only the three canonical axis
names; this module is the lenient parser used where raw user/checkpoint strings
enter the model layer, so the alias table lives in exactly one place instead of
being re-hand-rolled in every model/runtime file.

``torch`` is imported lazily so this module stays importable in torch-free
contexts (config resolution, Ray launcher serialization).
"""

from __future__ import annotations

from typing import Any

# Canonical dtype name -> tolerant input spellings accepted at config/checkpoint
# boundaries. Source of truth; the lenient lookup below is derived from it so the
# identity entry (a canonical name accepts itself) is generated, not hand-listed.
# The sub-byte float8/float4 names are the rollout-side quantized GEMM dtypes
# (Hopper/Blackwell); they resolve only on a torch build that compiled the dtype
# in (guarded in resolve_torch_dtype), never silently degrading to a wider type.
_DTYPE_SPELLINGS: dict[str, tuple[str, ...]] = {
    "bfloat16": ("bf16",),
    "float16": ("fp16", "half"),
    "float32": ("fp32", "float"),
    "float8_e4m3fn": ("fp8", "e4m3", "float8", "float8_e4m3"),
    "float8_e5m2": ("e5m2",),
    "float4_e2m1fn_x2": ("fp4", "e2m1", "float4"),
}
# Derived tolerant-spelling lookup shared by both dtype resolution directions:
# string -> torch.dtype and dtype/string -> canonical config string.
_DTYPE_NAME_BY_INPUT: dict[str, str] = {
    spelling: canonical
    for canonical, spellings in _DTYPE_SPELLINGS.items()
    for spelling in (canonical, *spellings)
}


def resolve_torch_dtype(value: Any) -> Any:
    """Resolve a dtype spelling (or an existing ``torch.dtype``) into a ``torch.dtype``."""

    import torch

    if isinstance(value, torch.dtype):
        return value
    key = str(value).removeprefix("torch.").lower()
    try:
        name = _DTYPE_NAME_BY_INPUT[key]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {value!r}") from exc
    # Derive the torch.dtype from its canonical name instead of a hand-listed map
    # so float8/float4 need no per-dtype branch. A name the running torch build
    # did not compile in (older builds lack float8/float4) fails loudly rather
    # than silently widening to a different precision.
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"precision {value!r} resolves to torch dtype {name!r}, which this "
            f"torch build ({torch.__version__}) does not provide; FP8/FP4 needs a "
            "torch built with that dtype (and Hopper/Blackwell to execute it)",
        )
    return dtype


def dtype_to_config_string(value: Any) -> str:
    """Normalize a dtype spelling (or ``torch.dtype``) to its canonical config string.

    Unknown values pass through unchanged so callers can forward already-canonical
    or non-dtype tokens without losing them.
    """

    text = str(value).removeprefix("torch.")
    return _DTYPE_NAME_BY_INPUT.get(text.lower(), text)


__all__ = ["dtype_to_config_string", "resolve_torch_dtype"]
