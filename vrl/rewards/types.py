"""Reward scoring data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RewardTrajectoryStep:
    """Single denoising step within a reward-scored trajectory."""

    timestep: int
    log_prob: Any  # Tensor — log-probability under the policy
    noise_pred: Any  # Tensor — predicted noise at this step
    new_log_prob: Any = None  # Refreshed log-prob under current policy
    ref_log_prob: Any = None  # Log-prob under reference policy


@dataclass(slots=True)
class RewardTrajectory:
    """Generation trajectory attached to reward scoring input."""

    prompt: str
    seed: int
    steps: list[RewardTrajectoryStep]
    output: Any  # Final generated output (frames / latents)


@dataclass(slots=True)
class RewardRollout:
    """A single generation paired with reward-scoring metadata."""

    request: Any  # VideoGenerationRequest or similar
    trajectory: RewardTrajectory
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["RewardRollout", "RewardTrajectory", "RewardTrajectoryStep"]
