"""Shared advantage normalization helpers for policy-gradient algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar


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
        denom = torch.clamp(std, min=eps)
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


class GroupAdvantageEstimator:
    """Compute scalar or multi-objective group-relative advantages.

    The estimator binds the resolved reward weights to one algorithm instance.
    This keeps reward configuration out of the public algorithm dataclass and
    provides a lifecycle owner for strategies that may gain runtime state.
    """

    __slots__ = (
        "adv_clip_max",
        "component_weights",
        "eps",
        "global_std",
        "strategy",
    )

    DEFAULT_STRATEGY = "weighted_sum_raw"

    def __init__(
        self,
        *,
        eps: float,
        adv_clip_max: float,
        global_std: bool,
        strategy: str = DEFAULT_STRATEGY,
        component_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.validate_strategy(strategy)
        self.eps = float(eps)
        self.adv_clip_max = float(adv_clip_max)
        self.global_std = bool(global_std)
        self.strategy = strategy
        self.component_weights = {
            name: float(weight) for name, weight in (component_weights or {}).items()
        }

    @classmethod
    def validate_strategy(cls, strategy: str) -> None:
        """Reject an unknown public strategy name at configuration time."""

        if strategy not in cls._STRATEGIES:
            raise ValueError(
                f"unknown advantage_combine strategy {strategy!r}; "
                f"available: {sorted(cls._STRATEGIES)}",
            )

    def compute(
        self,
        rewards: Any,
        group_ids: Any,
        *,
        component_rewards: Mapping[str, Any] | None = None,
    ) -> Any:
        """Compute advantages using the configured aggregation strategy."""

        compute_strategy = self._STRATEGIES[self.strategy]
        return compute_strategy(self, rewards, component_rewards, group_ids)

    def _normalize_weighted_rewards(
        self,
        rewards: Any,
        _component_rewards: Mapping[str, Any] | None,
        group_ids: Any,
    ) -> Any:
        """Normalize the weighted total already produced by the reward runtime."""

        return group_relative_advantages(
            rewards,
            group_ids,
            eps=self.eps,
            adv_clip_max=self.adv_clip_max,
            global_std=self.global_std,
        )

    def _normalize_components_then_sum(
        self,
        _rewards: Any,
        component_rewards: Mapping[str, Any] | None,
        group_ids: Any,
    ) -> Any:
        """Normalize each objective, weighted-sum it, then clamp once."""

        import torch

        if not component_rewards:
            raise ValueError("normalized_sum requires per-component rewards")
        component_names = set(component_rewards)
        weight_names = set(self.component_weights)
        if component_names != weight_names:
            raise ValueError(
                "reward component keys must match configured weights; "
                f"missing={sorted(weight_names - component_names)}, "
                f"unknown={sorted(component_names - weight_names)}",
            )

        total = None
        # Stable ordering keeps global-std collectives aligned across DDP ranks.
        for name in sorted(component_rewards):
            advantage = _standardize_group_rewards(
                component_rewards[name],
                group_ids,
                eps=self.eps,
                global_std=self.global_std,
            )
            weighted = self.component_weights[name] * advantage
            total = weighted if total is None else total + weighted
        return torch.clamp(total, -self.adv_clip_max, self.adv_clip_max)

    # This table is both dispatch and the source of truth for public validation.
    _STRATEGIES: ClassVar = {
        DEFAULT_STRATEGY: _normalize_weighted_rewards,
        "normalized_sum": _normalize_components_then_sum,
    }


__all__ = ["GroupAdvantageEstimator", "group_relative_advantages"]
