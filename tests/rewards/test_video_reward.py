"""Tests for the current VideoReward boundary."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.validation import validate_reward_config
from vrl.rewards.types import RewardRollout, RewardTrajectory
from vrl.rewards.video_reward import VideoReward


def _rollout(output: torch.Tensor) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt="prompt", seed=0, steps=[], output=output),
    )


def _video_reward_config(**video_kwargs: object):
    kwargs = {
        "backend": "stub",
        "reward_name": "dance_grpo",
        "score_key": "overall_reward",
    }
    kwargs.update(video_kwargs)
    return OmegaConf.create(
        {
            "reward": {
                "components": {"video_reward": 1.0},
                "kwargs": {"video_reward": kwargs},
            },
        },
    )


@pytest.mark.asyncio
async def test_video_reward_stub_scores_tensor_means() -> None:
    reward = VideoReward(
        backend="stub",
        reward_name="dance_grpo",
        score_key="overall_reward",
        stub_scale=0.1,
    )

    scores = await reward.score_batch(
        [
            _rollout(torch.ones(1, 2, 2)),
            _rollout(torch.full((1, 2, 2), 2.0)),
        ],
    )

    assert scores == pytest.approx([1.1, 2.2])


@pytest.mark.parametrize("backend", ["remote", "local", "ray"])
def test_video_reward_rejects_non_stub_backend_until_ray_runtime_exists(backend: str) -> None:
    with pytest.raises(NotImplementedError, match="Ray reward inference runtime"):
        VideoReward(
            backend=backend,
            reward_name="dance_grpo",
            score_key="overall_reward",
        )


def test_video_reward_config_rejects_remote_backend() -> None:
    cfg = _video_reward_config(backend="remote")

    with pytest.raises(ValueError, match=r"video_reward\.backend only supports 'stub'"):
        validate_reward_config(cfg)


def test_video_reward_config_rejects_removed_endpoint_fields() -> None:
    cfg = _video_reward_config(enqueue_url="/removed")

    with pytest.raises(ValueError, match="external reward endpoint fields"):
        validate_reward_config(cfg)
