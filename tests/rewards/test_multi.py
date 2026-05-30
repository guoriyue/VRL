"""Tests for vrl.rewards.functions.registry."""

from __future__ import annotations

import pytest

from vrl.config.builders import build_reward_config
from vrl.config.loading import load_config
from vrl.rewards.base import RewardFunction
from vrl.rewards.functions.codex_image_qa import CodexImageQAReward
from vrl.rewards.functions.registry import MultiReward, get_reward
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


def test_codex_image_qa_reward_config_builds_primary_entry_without_legacy_alias() -> None:
    cfg = load_config("reward/codex_image_qa")
    weights, kwargs = build_reward_config(cfg)

    reward = MultiReward.from_dict(weights, device="cpu", reward_kwargs=kwargs)

    assert [name for name, _, _ in reward.rewards] == ["codex_image_qa"]
    assert get_reward("codex_image_qa") is CodexImageQAReward

    with pytest.raises(KeyError, match="image_qa_cli"):
        get_reward("image_qa_cli")


def test_claude_anatomy_config_builds_claude_image_qa_backend() -> None:
    cfg = load_config("reward/claude_anatomy")
    weights, kwargs = build_reward_config(cfg)

    reward = MultiReward.from_dict(weights, device="cpu", reward_kwargs=kwargs)

    assert [name for name, _, _ in reward.rewards] == ["claude_image_qa"]
