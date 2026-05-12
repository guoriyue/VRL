from __future__ import annotations

import pytest
import torch

from vrl.algorithms.types import Rollout, Trajectory
from vrl.rewards.remote_video import extract_scores
from vrl.rewards.video_reward import VideoReward


def _rollout(prompt: str, value: float) -> Rollout:
    return Rollout(
        request=None,
        trajectory=Trajectory(
            prompt=prompt,
            seed=0,
            steps=[],
            output=torch.full((3, 2, 4, 4), value),
        ),
        metadata={"video_fps": 16.0},
    )


@pytest.mark.asyncio
async def test_video_reward_stub_is_explicit_and_non_flat() -> None:
    reward = VideoReward(
        backend="stub",
        reward_name="dance_grpo",
        score_key="overall_reward",
        stub_scale=0.1,
    )

    scores = await reward.score_batch([_rollout("a", 0.0), _rollout("b", 0.0)])

    assert scores == pytest.approx([0.1, 0.2])


def test_remote_video_extracts_composite_score_key() -> None:
    response = {"scores": {"vq_reward": [1.0, 2.0], "mq_reward": [0.5, 0.25]}}

    assert extract_scores(response, "vq_reward+mq_reward") == pytest.approx([1.5, 2.25])


def test_remote_video_missing_score_key_fails_fast() -> None:
    response = {"scores": {"vq_reward": [1.0]}}

    with pytest.raises(KeyError, match="missing score keys"):
        extract_scores(response, "overall_reward")
