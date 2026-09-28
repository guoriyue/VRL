"""Algorithm-facing training input derived from trajectory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.rollouts.evaluators.types import TrajectorySignalBatch


@dataclass(slots=True)
class AlgorithmInput:
    """Unified algorithm-facing input derived from trajectory contracts."""

    signals: TrajectorySignalBatch | None = None
    advantages: Any | None = None
    model: Any | None = None
    rollout_batch: Any | None = None
    timestep_index: int | None = None


__all__ = ["AlgorithmInput"]
