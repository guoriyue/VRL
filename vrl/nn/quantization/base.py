"""Shared base for quantized linear schemes.

Every rollout quantization scheme — ``Fp8Linear`` today, future ``Fp4Linear`` /
``Int8Linear`` as siblings — subclasses :class:`QuantizedLinear`. This is the
single contract the rest of the stack keys off: the rollout backstop guard counts
"any quantized linear" via ``isinstance(m, QuantizedLinear)`` instead of a
hand-maintained per-scheme type list, so a newly added scheme is covered the
moment it subclasses this (no guard edit, no silent gap).
"""

from __future__ import annotations

from torch import nn


class QuantizedLinear(nn.Module):
    """Marker base for a low-precision drop-in ``nn.Linear`` replacement.

    Carries no behavior — subclasses implement the actual quantized forward. Its
    only job is to be the type the swap installs and the backstop guard counts, so
    "was a quantized rollout actually applied?" is one ``isinstance`` check across
    all present and future schemes. Every scheme also implements
    ``drop_master() -> int`` (free the high-precision master, return bytes freed)
    so :func:`drop_quantized_masters` covers it without a per-scheme list.
    """


def drop_quantized_masters(root: nn.Module) -> int:
    """Free every quantized linear's high-precision master under ``root``.

    Returns the bytes freed. Valid whenever weight-sync never loads base weights
    into these modules (adapter-only or sync-free rollouts) — see the per-scheme
    ``drop_master`` docstrings.
    """
    return sum(
        module.drop_master()
        for module in root.modules()
        if isinstance(module, QuantizedLinear)
    )


__all__ = ["QuantizedLinear", "drop_quantized_masters"]
