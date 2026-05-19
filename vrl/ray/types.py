"""Shared Ray substrate types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RayActorHandle:
    """Driver-visible metadata for one Ray actor."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...] = ()
    actor: Any | None = None


__all__ = ["RayActorHandle"]
