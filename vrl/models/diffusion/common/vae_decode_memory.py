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


def configure_memory_mechanisms(
    target: Any,
    mem: VaeDecodeMemory,
    *,
    owner: str,
) -> tuple[str, ...]:
    """Apply memory mechanisms to one target; fail on unsupported requests."""

    applied: list[str] = []
    if mem.tiling:
        _call_required(target, "enable_tiling", owner=owner)
        applied.append("tiling")
    if mem.slicing:
        _call_required(target, "enable_slicing", owner=owner)
        applied.append("slicing")
    return tuple(applied)


# Metadata prefixes are a downstream contract (bundle metadata + tests):
# the historical vae_decode target reports as ``vae_tiling`` / ``vae_slicing``.
# New targets default to their own name as prefix.
_METADATA_PREFIX = {"vae_decode": "vae"}


def apply_generation_memory_policy(
    model: Any,
    *,
    memory_config: Mapping[str, Any] | None,
    owner: str,
) -> dict[str, dict[str, bool]]:
    """Apply ``model.memory`` to the model's declared targets; return metadata.

    ``model.memory`` is target-keyed: every section name must match a key in
    the model's ``generation_memory_targets()`` (today ``vae_decode``; future
    targets — encoders, transformer offload — appear here without policy
    changes). Family models declare WHAT can be configured; this policy owns
    HOW and WHEN. Runtime builders call it once after model construction and
    attach the returned metadata to the bundle. A section naming a target the
    model does not expose (including typos) is a config error, never a silent
    no-op.
    """

    targets = model.generation_memory_targets()
    sections = dict(memory_config or {})
    unknown = sorted(set(sections) - set(targets))
    if unknown:
        exposed = ", ".join(sorted(targets)) or "<none>"
        raise ValueError(
            f"{owner} configures model.memory section(s) "
            f"{', '.join(unknown)} but the model only exposes generation "
            f"memory target(s): {exposed}",
        )

    metadata: dict[str, bool] = {}
    for target_name in sorted(set(targets) | set(sections)):
        mem = vae_decode_memory_from_config(sections.get(target_name))
        applied: tuple[str, ...] = ()
        if target_name in sections:
            applied = configure_memory_mechanisms(
                targets[target_name],
                mem,
                owner=f"{owner}:{target_name}",
            )
        prefix = _METADATA_PREFIX.get(target_name, target_name)
        applied_set = set(applied)
        metadata[f"{prefix}_tiling"] = bool(mem.tiling and "tiling" in applied_set)
        metadata[f"{prefix}_slicing"] = bool(mem.slicing and "slicing" in applied_set)
    return {MEMORY_POLICY_METADATA_KEY: {"model_build": metadata}}


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
