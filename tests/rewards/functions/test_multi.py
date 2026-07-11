"""Tests for vrl.rewards.functions.registry."""

from __future__ import annotations

import pytest

from vrl.rewards.base import RewardFunction
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.types import RewardRollout, RewardTrajectory


def _make_rollout(prompt: str) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt=prompt, output=None),
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


class _TimedBatchReward(RewardFunction):
    def __init__(self, scores: list[float], timing_ms: dict[str, float], tag: str) -> None:
        self.scores = list(scores)
        self.timing_ms = dict(timing_ms)
        self.tag = tag
        self.last_results: list[str] = []
        self.last_timing_ms: dict[str, float] = {}

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        assert len(rollouts) == len(self.scores)
        self.last_results = [self.tag]
        self.last_timing_ms = dict(self.timing_ms)
        return list(self.scores)


@pytest.mark.asyncio
async def test_multi_reward_accumulates_components_until_reset() -> None:
    """Checks multi reward accumulates components until reset."""
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


@pytest.mark.asyncio
async def test_zero_weight_component_is_scored_without_changing_total() -> None:
    """Checks observation-only scores are logged but excluded from the reward."""
    reward = MultiReward(
        [
            ("train", 1.0, _QueuedBatchReward([[2.0, 3.0]])),
            ("observe", 0.0, _QueuedBatchReward([[0.7, 0.8]])),
        ],
    )

    scores = await reward.score_batch([_make_rollout("a"), _make_rollout("b")])

    assert scores == pytest.approx([2.0, 3.0])
    assert reward.last_components == pytest.approx(
        {"train": [2.0, 3.0], "observe": [0.7, 0.8]},
    )


@pytest.mark.asyncio
async def test_multi_reward_aggregates_inference_observations() -> None:
    """Checks multi reward exposes child reward inference timings."""
    reward = MultiReward(
        [
            (
                "first",
                1.0,
                _TimedBatchReward(
                    [0.1, 0.2],
                    {
                        "latency_ms": 5.0,
                        "queue_wait_ms": 1.0,
                        "inference_ms": 4.0,
                    },
                    "first-result",
                ),
            ),
            (
                "second",
                2.0,
                _TimedBatchReward(
                    [1.0, 2.0],
                    {
                        "latency_ms": 7.0,
                        "queue_wait_ms": 3.0,
                        "inference_ms": 6.0,
                    },
                    "second-result",
                ),
            ),
        ],
    )

    scores = await reward.score_batch([_make_rollout("a"), _make_rollout("b")])

    assert scores == pytest.approx([2.1, 4.2])
    assert reward.last_results == ["first-result", "second-result"]
    assert reward.last_timing_ms == pytest.approx(
        {
            "latency_ms": 12.0,
            "queue_wait_ms": 4.0,
            "inference_ms": 10.0,
        },
    )
