"""Weighted sum of multiple reward functions."""

from __future__ import annotations

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardRollout


class CompositeReward(RewardFunction):
    """Linearly combines multiple reward functions with weights."""

    def __init__(self, rewards: list[tuple[float, RewardFunction]]) -> None:
        self.rewards = rewards

    async def score(self, rollout: RewardRollout) -> float:
        total = 0.0
        for weight, fn in self.rewards:
            total += weight * await fn.score(rollout)
        return total

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        # Accumulate weighted scores from each sub-function
        totals = [0.0] * len(rollouts)
        for weight, fn in self.rewards:
            sub_scores = await fn.score_batch(rollouts)
            for i, s in enumerate(sub_scores):
                totals[i] += weight * s
        return totals
