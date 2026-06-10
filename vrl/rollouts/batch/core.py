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


def stack_batches(batches: list[RolloutBatch]) -> RolloutBatch:
    """Concatenate multiple RolloutBatch instances along batch dim.

    Tensor fields are ``torch.cat``-ed; list fields are concatenated;
    extras are merged from the first batch (non-tensor extras are kept as-is).
    """
    import torch

    if len(batches) == 1:
        return batches[0]

    observations = torch.cat([b.observations for b in batches], dim=0)
    actions = torch.cat([b.actions for b in batches], dim=0)
    rewards = torch.cat([b.rewards for b in batches], dim=0)
    dones = torch.cat([b.dones for b in batches], dim=0)
    group_ids = torch.cat([b.group_ids for b in batches], dim=0)

    if all(b.videos is not None for b in batches):
        videos = torch.cat([b.videos for b in batches], dim=0)
    else:
        videos = None

    prompts: list[str] = []
    for batch in batches:
        if batch.prompts is not None:
            prompts.extend(batch.prompts)

    # Extras can carry nested segment payloads such as Janus-Pro-R1 replay data.
    extras: dict[str, Any] = {}
    first = batches[0].extras
    for key in first:
        extras[key] = _stack_extra_values([batch.extras[key] for batch in batches])

    from vrl.trajectory.ops import stack_trajectory_batches

    context: dict[str, Any] = dict(batches[0].context)
    trajectory = stack_trajectory_batches([batch.trajectory for batch in batches])
    training_view = _stack_training_views([batch.training_view for batch in batches])

    return RolloutBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        group_ids=group_ids,
        extras=extras,
        context=context,
        videos=videos,
        prompts=prompts or None,
        trajectory=trajectory,
        training_view=training_view,
    )


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


__all__ = ["RolloutBatch", "stack_batches"]
