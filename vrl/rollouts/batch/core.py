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


def _stack_extra_values(values: list[Any]) -> Any:
    """Stack nested rollout extras along the sample dimension."""
    import torch

    if not values:
        raise ValueError("values must be non-empty")
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        keys = set(first)
        for value in values[1:]:
            if not isinstance(value, dict) or set(value) != keys:
                raise ValueError("nested rollout extras must have matching dict keys")
        return {
            key: _stack_extra_values([value[key] for value in values])
            for key in first
        }
    if isinstance(first, list) and all(isinstance(value, list) for value in values):
        out: list[Any] = []
        for value in values:
            out.extend(value)
        return out
    if isinstance(first, tuple) and all(value == first for value in values):
        return first
    return first


def _stack_training_views(values: list[Any]) -> Any:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("cannot stack mixed training_view and non-training_view batches")
    first = values[0]
    if all(value == first for value in values):
        return first
    raise ValueError("training_view must match across stacked batches")


__all__ = ["RolloutBatch"]
