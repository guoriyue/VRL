"""VAE decode memory knobs (tiling/slicing) co-located with latent decoding.

VAE tiling and slicing trade latency for peak memory during decode. They live
here next to ``latent_decode.py`` because applying them requires executing on a
concrete diffusers VAE object (``enable_tiling()`` / ``enable_slicing()``), which
is decode-path execution — not a pure config view.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from vrl.models.interfaces.runtime import MEMORY_POLICY_METADATA_KEY
from vrl.utils.config import plain_mapping


@dataclass(frozen=True, slots=True)
class VaeDecodeMemory:
    """Explicit VAE decode memory behavior applied while building a runtime."""

    tiling: bool = False
    slicing: bool = False


# Accepted ``vae_decode`` keys are exactly the dataclass fields — derived so
# adding a knob never silently turns into an "unknown key" rejection.
_VAE_DECODE_KEYS = frozenset(f.name for f in fields(VaeDecodeMemory))


def vae_decode_memory_from_config(
    section: Mapping[str, Any] | None,
) -> VaeDecodeMemory:
    """Parse a ``vae_decode`` sub-block into explicit decode memory behavior."""

    if section is None:
        return VaeDecodeMemory()
    raw = plain_mapping(section, field_name="model.memory.vae_decode")
    unknown = sorted(set(raw) - _VAE_DECODE_KEYS)
    if unknown:
        expected = ", ".join(sorted(_VAE_DECODE_KEYS))
        raise ValueError(
            f"unknown model.memory.vae_decode key(s): {', '.join(unknown)}; "
            f"expected {expected}",
        )

    updates = {key: bool(value) for key, value in raw.items()}
    return replace(VaeDecodeMemory(), **updates)


def configure_vae_decode(
    vae: Any,
    mem: VaeDecodeMemory,
    *,
    owner: str,
) -> tuple[str, ...]:
    """Apply VAE decode memory knobs and fail on unsupported requests."""

    applied: list[str] = []
    if mem.tiling:
        _call_required(vae, "enable_tiling", owner=owner)
        applied.append("tiling")
    if mem.slicing:
        _call_required(vae, "enable_slicing", owner=owner)
        applied.append("slicing")
    return tuple(applied)


def apply_vae_decode_memory(
    vae: Any,
    *,
    memory_config: Mapping[str, Any] | None,
    owner: str,
) -> dict[str, dict[str, bool]]:
    """Apply ``model.memory.vae_decode`` and return bundle metadata."""

    section = memory_config.get("vae_decode") if memory_config else None
    mem = vae_decode_memory_from_config(section)
    applied = configure_vae_decode(vae, mem, owner=owner)
    return vae_decode_memory_metadata(mem, applied=applied)


def vae_decode_memory_metadata(
    mem: VaeDecodeMemory,
    *,
    applied: Iterable[str] = (),
) -> dict[str, dict[str, bool]]:
    """Return report-only metadata for VAE decode memory behavior.

    The ``vae_tiling`` / ``vae_slicing`` metadata key names are a downstream
    contract (bundle metadata + tests) and must stay stable.
    """

    applied_set = set(applied)
    return {
        MEMORY_POLICY_METADATA_KEY: {
            "model_build": {
                "vae_tiling": bool(mem.tiling and "tiling" in applied_set),
                "vae_slicing": bool(mem.slicing and "slicing" in applied_set),
            },
        },
    }


def _call_required(target: Any, method_name: str, *, owner: str) -> None:
    method = getattr(target, method_name, None)
    if not callable(method):
        raise TypeError(f"{owner} does not support requested {method_name}()")
    method()


__all__ = [
    "VaeDecodeMemory",
    "apply_vae_decode_memory",
    "configure_vae_decode",
    "vae_decode_memory_from_config",
    "vae_decode_memory_metadata",
]
