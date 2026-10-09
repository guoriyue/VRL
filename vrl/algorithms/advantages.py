"""Shared advantage normalization helpers for policy-gradient algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def nonzero_advantage_mask(advantages: Any) -> Any:
    """Flow-GRPO's mask for samples with non-zero total advantage."""

    adv_abs = advantages.detach().abs()
    if adv_abs.dim() <= 1:
        return adv_abs != 0
    reduce_dims = tuple(range(1, adv_abs.dim()))
    return adv_abs.sum(dim=reduce_dims) != 0


def all_reduce_sufficient_stats(values: Any) -> tuple[Any, Any, Any]:
    """Cross-rank ``(Σx, Σx², n)`` in one SUM collective.

    The one hand-copied reduction under three different theorems
    (population std here, reward mean/std in the trainer, rollout mean in
    Flash-GRPO): stack the sufficient statistics, move to the rank's GPU for
    nccl (gloo handles CPU directly, e.g. in tests), reduce, hand back the
    three tensors for the caller's own derivation. Single-rank returns the
    local statistics so both branches share one formula. An empty local
    tensor must NOT short-circuit before the collective: emptiness is
    rank-local, so an empty rank contributes zeros instead.
    """

    import torch

    stats = torch.stack(
        [
            values.sum(),
            values.mul(values).sum(),
            values.new_tensor(float(values.numel())),
        ],
    )
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if dist.get_backend() == "nccl":
            stats = stats.cuda()
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats[0], stats[1], stats[2]


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

    An empty local ``rewards`` must NOT short-circuit: emptiness is a rank-local
    data condition, and returning early on it would skip the all_reduce that the
    other ranks are already blocking on. An empty rank contributes zeros to the
    reduction instead.
    """

    import torch

    # Reward normalization epsilons such as 1e-8 underflow in fp16, so all
    # sufficient statistics must be accumulated in at least fp32.
    stats_rewards = (
        rewards.float() if rewards.dtype in {torch.float16, torch.bfloat16} else rewards
    )
    n = rewards.numel()
    dist = torch.distributed
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if not distributed:
        if n <= 1:
            return stats_rewards.new_tensor(0.0)
        variance, _mean = torch.var_mean(stats_rewards, correction=0)
        return torch.sqrt(variance)

    g_sum, g_sumsq, g_count = all_reduce_sufficient_stats(stats_rewards)
    if g_count <= 1:
        return stats_rewards.new_tensor(0.0)
    g_mean = g_sum / g_count
    g_var = (g_sumsq / g_count) - g_mean * g_mean
    return torch.sqrt(torch.clamp(g_var, min=0.0)).to(rewards.device)


def _standardize_group_rewards(
    rewards: Any,
    group_ids: Any,
    *,
    eps: float,
    global_std: bool,
) -> Any:
    """Standardize rewards within each group without applying a final clamp.

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

        # Separate mean/std reductions can round a constant decimal group to a
        # non-zero std and create an O(1) fake advantage. One var_mean reduction
        # keeps centering and variance consistent; fp32 also keeps eps representable.
        stats_rewards = (
            group_rewards.float()
            if group_rewards.dtype in {torch.float16, torch.bfloat16}
            else group_rewards
        )
        variance, mean = torch.var_mean(stats_rewards, correction=0)
        std = global_std_value if global_std else torch.sqrt(variance)
        # Additive epsilon, as Flow-GRPO's and Flash-GRPO's PerPromptStatTracker
        # write it (``std + 1e-4``): a max() floor would silently leave a small
        # group std uncorrected below eps.
        denom = std + eps
        advantages[mask] = ((stats_rewards - mean) / denom).to(rewards.dtype)
    return advantages


def group_relative_advantages(
    rewards: Any,
    group_ids: Any,
    *,
    eps: float,
    adv_clip_max: float,
    global_std: bool,
) -> Any:
    """Normalize and clamp rewards within each GRPO prompt group."""

    import torch

    advantages = _standardize_group_rewards(
        rewards,
        group_ids,
        eps=eps,
        global_std=global_std,
    )
    return torch.clamp(advantages, -adv_clip_max, adv_clip_max)


# How a multi-component reward turns into one advantage. ``normalized_sum``
# standardizes each configured component within its group before weighting,
# so no reward's units can dominate the update; ``weighted_sum_raw``
# standardizes the weighted total the reward runtime already produced.
ADVANTAGE_COMBINE_STRATEGIES = ("weighted_sum_raw", "normalized_sum")


@dataclass(slots=True)
class GroupAdvantageConfig:
    """Normalization settings shared by group-relative policy objectives."""

    eps: float = 1e-4
    adv_clip_max: float = 5.0
    global_std: bool = False
    advantage_combine: str = "normalized_sum"

    def __post_init__(self) -> None:
        if self.advantage_combine not in ADVANTAGE_COMBINE_STRATEGIES:
            raise ValueError(
                f"unknown advantage_combine strategy {self.advantage_combine!r}; "
                f"available: {sorted(ADVANTAGE_COMBINE_STRATEGIES)}",
            )


class GroupRelativeObjective:
    """The advantage half of an objective that normalizes rewards per prompt group.

    GRPO, DiffusionNFT and V-GRPO inherit it: ``config`` is the objective's
    own hyper-parameters (the normalization fields are the
    ``GroupAdvantageConfig`` part of it) and ``component_weights`` are the
    reward config's objective weights, bound once at construction.
    """

    def __init__(
        self,
        config: GroupAdvantageConfig,
        *,
        component_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.config = config
        self.component_weights = {
            name: float(weight) for name, weight in (component_weights or {}).items()
        }

    def compute_advantages_from_tensors(self, rewards: Any, group_ids: Any) -> Any:
        """Standardize and clamp the weighted reward total within each group."""

        cfg = self.config
        return group_relative_advantages(
            rewards,
            group_ids,
            eps=cfg.eps,
            adv_clip_max=cfg.adv_clip_max,
            global_std=cfg.global_std,
        )

    def compute_advantages_from_components(
        self,
        rewards: Any,
        component_rewards: Mapping[str, Any],
        group_ids: Any,
    ) -> Any:
        """Advantages from the weighted total and its raw component observations.

        Under ``normalized_sum`` each configured objective is standardized in
        its group, weighted and summed, then clamped once. A reward also
        reports observation axes beside its objectives (``MultiReward``
        namespaces them as ``<component>/<axis>``) for logging; those never
        enter the advantage. With no configured objective in hand (a reward
        that reports only observations) there is exactly one objective, the
        weighted total, so standardizing it is the whole strategy. Some but not
        all configured objectives is a misconfiguration.
        """

        import torch

        cfg = self.config
        objectives = {
            name: values
            for name, values in component_rewards.items()
            if name in self.component_weights
        }
        if cfg.advantage_combine != "normalized_sum" or not objectives:
            return self.compute_advantages_from_tensors(rewards, group_ids)
        missing = sorted(set(self.component_weights) - set(objectives))
        if missing:
            raise ValueError(
                f"reward reports configured components {sorted(objectives)} but not {missing}",
            )

        total = None
        # Stable ordering keeps global-std collectives aligned across DDP ranks.
        for name in sorted(objectives):
            advantage = _standardize_group_rewards(
                objectives[name],
                group_ids,
                eps=cfg.eps,
                global_std=cfg.global_std,
            )
            weighted = self.component_weights[name] * advantage
            total = weighted if total is None else total + weighted
        return torch.clamp(total, -cfg.adv_clip_max, cfg.adv_clip_max)


__all__ = [
    "ADVANTAGE_COMBINE_STRATEGIES",
    "GroupAdvantageConfig",
    "GroupRelativeObjective",
    "group_relative_advantages",
    "nonzero_advantage_mask",
]
