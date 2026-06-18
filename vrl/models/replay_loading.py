"""Runtime metadata helper for replay-only model loading.

The trainer and Ray rollout worker load different runtime surfaces: rollout
workers own full generation state for sampling/decoding; trainers will eventually
own only the modules needed to replay recorded trajectory actions. The one fact
the runtime actually reads is whether a bundle owns full generation modules,
exposed as ``loads_full_generation_modules`` and consumed by the colocated-RAM
guard (``validate_colocated_replay_memory``) to size host memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LOADS_FULL_GENERATION_MODULES_KEY = "loads_full_generation_modules"


def full_generation_bundle_metadata() -> dict[str, Any]:
    """Return metadata for a runtime bundle that owns full generation modules."""

    return {LOADS_FULL_GENERATION_MODULES_KEY: True}


def minimal_replay_bundle_metadata() -> dict[str, Any]:
    """Return metadata for a trainer bundle that owns only replay modules."""

    return {LOADS_FULL_GENERATION_MODULES_KEY: False}


def bundle_loads_full_generation_modules(bundle: Any) -> bool:
    """Return whether a runtime bundle declares full generation module ownership."""

    metadata = getattr(bundle, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get(LOADS_FULL_GENERATION_MODULES_KEY, False))


__all__ = [
    "LOADS_FULL_GENERATION_MODULES_KEY",
    "bundle_loads_full_generation_modules",
    "full_generation_bundle_metadata",
    "minimal_replay_bundle_metadata",
]
