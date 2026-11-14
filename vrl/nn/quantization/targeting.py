"""Shared module-path policy for linear quantization."""

from __future__ import annotations

# Small, numerically sensitive layers stay in the model's base dtype. These
# names describe model structure rather than one quantization scheme.
DEFAULT_EXCLUDE: tuple[str, ...] = (
    "norm",
    "embed",
    "proj_out",
    "time_",
    "_mod",
    "txt_in",
)

__all__ = ["DEFAULT_EXCLUDE"]
