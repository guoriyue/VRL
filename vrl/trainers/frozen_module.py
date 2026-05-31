"""Driver-side parking of frozen generation-only modules.

Before Ray workers load model replicas, the driver process can move frozen
generation-only modules (text encoders, VAE) off CUDA to free room. This is
trainer-side execution (``module.to("cpu")`` + cache release), so it lives with
the trainers rather than in a pure config view.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from vrl.models.interfaces.runtime import MEMORY_POLICY_METADATA_KEY
from vrl.utils.config import plain_mapping


@dataclass(frozen=True, slots=True)
class FrozenModuleOffload:
    """Explicit driver-side frozen-module CPU offload behavior."""

    enable: bool = False
    module_names: tuple[str, ...] = ()
    empty_cache: bool = True


# Accepted ``frozen_offload`` sub-block keys map to the dataclass fields. The
# YAML key names (``modules``) differ from the field names, so this small mapping
# is the structural boundary between config vocabulary and the dataclass.
_FROZEN_OFFLOAD_KEYS = frozenset({"enable", "modules", "empty_cache"})


def frozen_module_offload_from_config(
    section: Mapping[str, Any] | None,
    *,
    defaults: FrozenModuleOffload | None = None,
) -> FrozenModuleOffload:
    """Merge a ``frozen_offload`` sub-block onto defaults."""

    offload = defaults or FrozenModuleOffload()
    if section is None:
        return offload
    raw = plain_mapping(section, field_name="model.memory.frozen_offload")
    unknown = sorted(set(raw) - _FROZEN_OFFLOAD_KEYS)
    if unknown:
        expected = ", ".join(sorted(_FROZEN_OFFLOAD_KEYS))
        raise ValueError(
            f"unknown model.memory.frozen_offload key(s): {', '.join(unknown)}; "
            f"expected {expected}",
        )

    updates: dict[str, Any] = {}
    if "enable" in raw:
        updates["enable"] = bool(raw["enable"])
    if "empty_cache" in raw:
        updates["empty_cache"] = bool(raw["empty_cache"])
    if "modules" in raw:
        updates["module_names"] = _normalize_names(raw["modules"], "modules")
    return replace(offload, **updates)


def park_frozen_modules(
    owner: Any,
    offload: FrozenModuleOffload,
) -> tuple[str, ...]:
    """Move declared frozen driver-only modules to CPU when enabled."""

    if not offload.enable:
        return ()

    moved: list[str] = []
    for name in offload.module_names:
        module = getattr(owner, name, None)
        if module is None:
            continue
        to = getattr(module, "to", None)
        if not callable(to):
            raise TypeError(f"cannot offload {name!r}: object has no callable to()")
        to("cpu")
        moved.append(name)

    if moved and offload.empty_cache:
        from vrl.utils.cuda_memory import empty_cuda_cache

        empty_cuda_cache()
    return tuple(moved)


def frozen_module_offload_metadata(
    offload: FrozenModuleOffload,
    *,
    moved: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Return report-only metadata for driver offload behavior.

    The ``runtime_lifecycle`` section key names are a downstream bundle-metadata
    contract (and tests) and must stay stable.
    """

    return {
        MEMORY_POLICY_METADATA_KEY: {
            "runtime_lifecycle": {
                "driver_frozen_module_cpu_offload": bool(offload.enable),
                "driver_frozen_module_names": list(offload.module_names),
                "driver_frozen_modules_moved": list(moved),
                "empty_cuda_cache_after_driver_offload": bool(offload.empty_cache),
            },
        },
    }


def _normalize_names(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"model.memory.frozen_offload.{field_name} must be a list")
    try:
        names = tuple(str(item) for item in value)
    except TypeError as exc:
        raise TypeError(
            f"model.memory.frozen_offload.{field_name} must be a list",
        ) from exc
    return tuple(name for name in names if name)


__all__ = [
    "FrozenModuleOffload",
    "frozen_module_offload_from_config",
    "frozen_module_offload_metadata",
    "park_frozen_modules",
]
