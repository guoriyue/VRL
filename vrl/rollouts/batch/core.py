"""Trainer-side batches produced from rollout generation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutBatch:
    """Trainer-ready batch collected from model rollouts.

    The collector fills in observations, actions, rewards, group IDs,
    and replay extras after the generation engine has returned a GenerationOutput.
    """

    observations: Any   # x_t -- current state [B, T, ...]
    actions: Any         # x_{t-1} -- next state (denoised) [B, T, ...]
    rewards: Any         # [B] scalar rewards per sample
    dones: Any           # [B] episode termination flags
    group_ids: Any       # [B] prompt group assignment (for per-prompt normalization)
    extras: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # shared metadata (not stacked)
    videos: Any | None = None      # [B, C, T, H, W] decoded frames (for reward scoring)
    prompts: list[str] | None = None
    trajectory: Any | None = None
    training_view: Any | None = None


__all__ = ["RolloutBatch"]
