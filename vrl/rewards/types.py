"""Reward scoring data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RewardTrajectory:
    """Generation trajectory attached to reward scoring input."""

    prompt: str
    output: Any  # Final generated output (frames / latents)


@dataclass(slots=True)
class RewardRollout:
    """A single generation paired with reward-scoring metadata."""

    request: Any  # VideoGenerationRequest or similar
    trajectory: RewardTrajectory
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["RewardRollout", "RewardTrajectory"]
