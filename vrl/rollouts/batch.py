"""Trainer-side batches produced from rollout generation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutBatch:
    """Trainer-ready batch collected from model rollouts.

    The collector/packer fills in observations, actions, rewards, group IDs,
    and replay extras after the generation engine has returned an OutputBatch.
    """

    observations: Any   # x_t — current state [B, T, ...]
    actions: Any         # x_{t-1} — next state (denoised) [B, T, ...]
    rewards: Any         # [B] scalar rewards per sample
    dones: Any           # [B] episode termination flags
    group_ids: Any       # [B] prompt group assignment (for per-prompt normalization)
    extras: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # shared metadata (not stacked)
    videos: Any | None = None      # [B, C, T, H, W] decoded frames (for reward scoring)
    prompts: list[str] | None = None


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

    # Videos: cat if all present
    if all(b.videos is not None for b in batches):
        videos = torch.cat([b.videos for b in batches], dim=0)
    else:
        videos = None

    # Prompts: concatenate lists
    prompts: list[str] = []
    for b in batches:
        if b.prompts is not None:
            prompts.extend(b.prompts)

    # Extras: cat tensor leaves from all batches, including nested segment
    # dicts such as Janus-Pro-R1's per-stage replay payload.
    extras: dict[str, Any] = {}
    first = batches[0].extras
    for key in first:
        extras[key] = _stack_extra_values([b.extras[key] for b in batches])

    # Context: shared metadata — take from first batch (not stacked)
    context: dict[str, Any] = dict(batches[0].context)

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
