"""Trainer-side batches produced from rollout generation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutBatch:
    """Trainer-ready batch collected from model rollouts.

    Reward scoring finishes before this boundary. The collector retains only
    tensors and trajectory facts consumed by replay or training.
    """

    observations: Any  # x_t -- current state [B, T, ...]
    actions: Any  # x_{t-1} -- next state (denoised) [B, T, ...]
    rewards: Any  # [B] scalar rewards per sample
    group_ids: Any  # [B] prompt group assignment (for per-prompt normalization)
    extras: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # shared metadata (not stacked)
    trajectory: Any | None = None


__all__ = ["RolloutBatch"]
