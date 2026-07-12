"""Constructor defaults owned by concrete disk-artifact rewards."""

from __future__ import annotations

import inspect

import pytest

from vrl.rewards.functions.cosmos3_reasoner import Cosmos3ReasonerReward
from vrl.rewards.functions.kling_video_reward import KlingVideoReward
from vrl.rewards.functions.phymotion import PhyMotionReward
from vrl.rewards.functions.unified_reward_video import UnifiedRewardVideoReward
from vrl.rewards.functions.videocon_physics import VideoConPhysicsReward
from vrl.rewards.functions.videoscore2 import VideoScore2Reward


@pytest.mark.parametrize(
    ("reward_cls", "reward_name", "score_key"),
    [
        (Cosmos3ReasonerReward, "nvidia/Cosmos3-Nano", "task_success"),
        (VideoScore2Reward, "TIGER-Lab/VideoScore2@main", "physical_common_sense"),
        (
            VideoConPhysicsReward,
            "videophysics/videocon_physics@main",
            "physical_commonsense",
        ),
        (
            UnifiedRewardVideoReward,
            "CodeGoat24/UnifiedReward-2.0-qwen-7b@main",
            "overall",
        ),
        (PhyMotionReward, "phymotion", "overall"),
    ],
)
def test_concrete_reward_signature_owns_its_defaults(
    reward_cls: type,
    reward_name: str,
    score_key: str,
) -> None:
    parameters = inspect.signature(reward_cls).parameters

    assert parameters["reward_name"].default == reward_name
    assert parameters["score_key"].default == score_key
    assert parameters["artifact_format"].default == "mp4"


def test_kling_signature_owns_its_mp4_default() -> None:
    parameters = inspect.signature(KlingVideoReward).parameters

    assert parameters["artifact_format"].default == "mp4"


def test_explicit_empty_values_are_not_replaced_by_truthiness_defaults(tmp_path) -> None:
    reward = VideoScore2Reward(
        reward_name="",
        score_key="",
        artifact_format="tensor",
        artifact_dir=str(tmp_path),
        runtime=object(),
    )

    assert reward.reward_name == ""
    assert reward.score_key == ""
    assert reward.artifact_store.artifact_format == "tensor"
