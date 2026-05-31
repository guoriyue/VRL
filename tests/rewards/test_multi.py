"""Tests for vrl.rewards.functions.registry."""

from __future__ import annotations

import pytest

from vrl.rewards.base import RewardFunction
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.types import RewardRollout, RewardTrajectory


def _make_rollout(prompt: str) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt=prompt, seed=0, steps=[], output=None),
    )


class _QueuedBatchReward(RewardFunction):
    def __init__(self, batches: list[list[float]]) -> None:
        self.batches = list(batches)

    async def score(self, rollout: RewardRollout) -> float:
        return self.batches.pop(0)[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        scores = self.batches.pop(0)
        assert len(scores) == len(rollouts)
        return scores


@pytest.mark.asyncio
async def test_multi_reward_accumulates_components_until_reset() -> None:
    reward = MultiReward(
        [
            ("ocr", 1.0, _QueuedBatchReward([[0.1, 0.2], [0.3]])),
            ("aesthetic", 0.5, _QueuedBatchReward([[1.0, 2.0], [3.0]])),
        ]
    )

    first = await reward.score_batch([_make_rollout("a"), _make_rollout("b")])
    second = await reward.score_batch([_make_rollout("c")])

    assert first == pytest.approx([0.6, 1.2])
    assert second == pytest.approx([1.8])
    assert set(reward.last_components) == {"ocr", "aesthetic"}
    assert reward.last_components["ocr"] == pytest.approx([0.1, 0.2, 0.3])
    assert reward.last_components["aesthetic"] == pytest.approx([1.0, 2.0, 3.0])

    reward.reset_components()

    assert reward.last_components == {}
