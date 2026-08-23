"""Constructor defaults owned by concrete disk-artifact rewards."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.cosmos3_reasoner import Cosmos3ReasonerReward
from vrl.rewards.functions.hpsv3 import HPSv3Reward
from vrl.rewards.functions.kling_video_reward import KlingVideoReward
from vrl.rewards.functions.phymotion import PhyMotionReward
from vrl.rewards.functions.robotics_video_reward import RoboticsVideoReward
from vrl.rewards.functions.unified_reward_video import UnifiedRewardVideoReward
from vrl.rewards.functions.videocon_physics import VideoConPhysicsReward
from vrl.rewards.functions.videoscore2 import VideoScore2Reward


class _Runtime:
    scoring_is_nonblocking = False
    external_accelerator_isolation_verified = False

    async def score_batch(self, request):
        return []

    async def shutdown(self) -> None:
        return None


@pytest.mark.parametrize(
    ("reward_cls", "reward_name", "score_key"),
    [
        (HPSv3Reward, "MizzenAI/HPSv3@main", "top_frame_mean"),
        (KlingVideoReward, "kling_video_reward", "overall_reward"),
        (RoboticsVideoReward, "robotics_video_reward", "robotics_blend"),
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
def test_concrete_reward_class_owns_its_defaults(
    reward_cls: type,
    reward_name: str,
    score_key: str,
) -> None:
    """Each disk reward is a declaration block: the class attributes ARE the
    defaults (the forwarding __init__ copies were the rule-3 disease)."""

    assert reward_cls.default_reward_name == reward_name
    assert reward_cls.default_score_key == score_key
    assert reward_cls.default_artifact_format == "mp4"
    assert reward_cls.model_factory
    assert reward_cls.request_prefix and reward_cls.debug_basename


@pytest.mark.parametrize("field", ["reward_name", "score_key"])
def test_explicit_empty_request_identity_is_rejected(tmp_path, field: str) -> None:
    kwargs = {
        "reward_name": "videoscore2",
        "score_key": "physical_common_sense",
        field: "",
    }

    with pytest.raises(ValueError, match=field):
        VideoScore2Reward(
            artifact_dir=str(tmp_path),
            scorer=_Runtime(),
            **kwargs,
        )
