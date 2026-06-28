"""Tests for target_video_similarity reward."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.functions.target_video_similarity import TargetVideoSimilarityReward
from vrl.rewards.types import RewardRollout, RewardTrajectory


def _rollout(target_image: Path, value: float) -> RewardRollout:
    video = torch.full((3, 1, 4, 4), value, dtype=torch.float32)
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt="match target", output=video),
        metadata={"target_image": str(target_image), "sample_id": "unit"},
    )


@pytest.mark.asyncio
async def test_target_video_similarity_scores_matching_target_higher(tmp_path: Path) -> None:
    """Matching generated video should score above a mismatched target."""
    black = tmp_path / "black.png"
    white = tmp_path / "white.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(black)
    Image.new("RGB", (4, 4), (255, 255, 255)).save(white)
    reward = TargetVideoSimilarityReward(device="cpu")

    matched = await reward.score(_rollout(black, 0.0))
    mismatched = await reward.score(_rollout(white, 0.0))

    assert matched == pytest.approx(1.0)
    assert mismatched < matched


@pytest.mark.asyncio
async def test_target_video_similarity_is_registered(tmp_path: Path) -> None:
    """MultiReward can instantiate the target-aware reward by name."""
    target = tmp_path / "target.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(target)
    reward = MultiReward.from_dict(
        {"target_video_similarity": 1.0},
        device="cpu",
        reward_kwargs={"target_video_similarity": {"worker_config": {"feature_size": 8}}},
    )

    scores = await reward.score_batch([_rollout(target, 0.0)])

    assert scores == pytest.approx([1.0])
    assert reward.last_components["target_video_similarity"] == pytest.approx([1.0])
