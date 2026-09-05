"""Trainer-side batches produced from rollout generation outputs.

``RolloutBatch`` is the single trainer-facing contract every engine converges
to: diffusion and token trajectories are both packed into it by the
collector's batch builder, and schedules, trainers, and evaluators consume it
without knowing which engine produced it. Kept separate from the operations
(``vrl.rollouts.batch.ops``) so the type stays a dependency-light leaf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from vrl.trajectory import TrajectoryBatch


@dataclass
class RolloutBatch:
    """Trainer-ready batch collected from model rollouts.

    Reward scoring finishes before this boundary. The collector retains only
    tensors and trajectory facts consumed by replay or training. Tensor and
    trajectory annotations are type-checking-only so this leaf stays torch-free
    at import.
    """

    rewards: torch.Tensor  # [B] scalar rewards per sample
    group_ids: torch.Tensor  # [B] prompt group assignment (for per-prompt normalization)
    extras: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # shared metadata (not stacked)
    # The collector always attaches the trajectory; None only exists for
    # synthetic batches (tests, batch-op fixtures) that never reach replay.
    trajectory: TrajectoryBatch | None = None


__all__ = ["RolloutBatch"]
