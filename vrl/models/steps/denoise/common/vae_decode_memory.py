"""VAE decode memory knobs (tiling/slicing) co-located with latent decoding.

VAE tiling and slicing trade latency for peak memory during decode. They live
here next to ``latent_decode.py`` because applying them requires executing on a
concrete diffusers VAE object (``enable_tiling()`` / ``enable_slicing()``), which
is decode-path execution — not a pure config view.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

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
            f"unknown model.memory.vae_decode key(s): {', '.join(unknown)}; expected {expected}",
        )

    updates = {key: bool(value) for key, value in raw.items()}
    return replace(VaeDecodeMemory(), **updates)


def configure_memory_mechanisms(
    target: Any,
    mem: VaeDecodeMemory,
    *,
    owner: str,
) -> None:
    """Apply memory mechanisms to one target; fail on unsupported requests."""

    if mem.tiling:
        _call_required(target, "enable_tiling", owner=owner)
    if mem.slicing:
        _call_required(target, "enable_slicing", owner=owner)


def apply_generation_memory_policy(
    model: Any,
    *,
    memory_config: Mapping[str, Any] | None,
    owner: str,
) -> None:
    """Apply ``model.memory`` to the model's declared generation targets.

    ``model.memory`` is target-keyed: every section name must match a key in
    the model's ``generation_memory_targets()`` (today ``vae_decode``; future
    targets — encoders, transformer offload — appear here without policy
    changes). Family models declare WHAT can be configured; this policy owns
    HOW and WHEN. Runtime builders call it once after model construction. A
    section naming an unknown target is a config error, never a silent no-op.
    """

    targets = model.generation_memory_targets()
    configured = dict(memory_config or {})
    unsupported = sorted(set(configured) - set(targets))
    if unsupported:
        exposed = ", ".join(sorted(targets)) or "<none>"
        raise ValueError(
            f"{owner} configures unsupported model.memory section(s) "
            f"{', '.join(unsupported)}; model exposes generation memory "
            f"target(s): {exposed}",
        )

    for target_name in sorted(targets):
        if target_name not in configured:
            continue
        configure_memory_mechanisms(
            targets[target_name],
            vae_decode_memory_from_config(configured[target_name]),
            owner=f"{owner}:{target_name}",
        )


def _call_required(target: Any, method_name: str, *, owner: str) -> None:
    method = getattr(target, method_name, None)
    if not callable(method):
        raise TypeError(f"{owner} does not support requested {method_name}()")
    method()


__all__ = [
    "VaeDecodeMemory",
    "apply_generation_memory_policy",
    "configure_memory_mechanisms",
    "vae_decode_memory_from_config",
]
