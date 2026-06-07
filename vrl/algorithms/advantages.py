"""Shared advantage normalization helpers for policy-gradient algorithms."""

from __future__ import annotations

from typing import Any


def group_relative_advantages(
    rewards: Any,
    group_ids: Any,
    *,
    eps: float,
    adv_clip_max: float,
    global_std: bool,
) -> Any:
    """Normalize rewards within each group using the GRPO advantage contract."""

    import torch

    advantages = torch.zeros_like(rewards)
    global_std_value = (
        rewards.std(unbiased=False) if rewards.numel() > 1 else rewards.new_tensor(0.0)
    )
    for gid in torch.unique(group_ids):
        mask = group_ids == gid
        group_rewards = rewards[mask]
        if group_rewards.numel() <= 1:
            advantages[mask] = 0.0
            continue

        mean = group_rewards.mean()
        std = global_std_value if global_std else group_rewards.std(unbiased=False)
        denom = torch.clamp(std, min=eps)
        group_adv = (group_rewards - mean) / denom
        advantages[mask] = torch.clamp(group_adv, -adv_clip_max, adv_clip_max)
    return advantages


__all__ = ["group_relative_advantages"]
