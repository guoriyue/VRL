"""Shared advantage normalization helpers for policy-gradient algorithms."""

from __future__ import annotations

from typing import Any


def _population_std_across_ranks(rewards: Any) -> Any:
    """Population std of ``rewards`` over **all DDP ranks**, not just the local slice.

    Under DDP each rank holds only its own prompt slice (e.g. 16 of the 32 global
    prompts), so a plain ``rewards.std()`` is a *per-rank* std — when ``global_std``
    is on, each rank would normalize by a different denominator and the
    "global" std would not be global at all. To make ``global_std`` mean what it
    says, all-reduce the sufficient statistics (sum, sum-of-squares, count) and
    derive the std from the cross-rank totals.

    No distributed process group (single-GPU / unit tests) or ``world_size == 1``
    → falls back to the local population std, which is already the true global std
    in those cases. The collective is gated on ``global_std`` at the call site, so
    every rank runs it in lockstep (no mismatched-collective deadlock).
    """

    import torch

    n = rewards.numel()
    if n == 0:
        return rewards.new_tensor(0.0)

    dist = torch.distributed
    distributed = (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    if not distributed:
        return rewards.std(unbiased=False) if n > 1 else rewards.new_tensor(0.0)

    # sum, sum-of-squares, count → reduced across ranks in one collective.
    stats = torch.stack(
        [
            rewards.sum(),
            rewards.mul(rewards).sum(),
            rewards.new_tensor(float(n)),
        ],
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    g_sum, g_sumsq, g_count = stats[0], stats[1], stats[2]
    if g_count <= 1:
        return rewards.new_tensor(0.0)
    g_mean = g_sum / g_count
    g_var = (g_sumsq / g_count) - g_mean * g_mean
    return torch.sqrt(torch.clamp(g_var, min=0.0))


def group_relative_advantages(
    rewards: Any,
    group_ids: Any,
    *,
    eps: float,
    adv_clip_max: float,
    global_std: bool,
) -> Any:
    """Normalize rewards within each group using the GRPO advantage contract.

    ``global_std`` shares one denominator across all groups; under DDP it is the
    true cross-rank std (see ``_population_std_across_ranks``). ``global_std=False``
    uses each group's own std and is fully rank-local (correct without any
    collective). The all-reduce only fires for ``global_std=True``, so both ranks
    take the same branch and the collective stays balanced.
    """

    import torch

    advantages = torch.zeros_like(rewards)
    global_std_value = _population_std_across_ranks(rewards) if global_std else None
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
