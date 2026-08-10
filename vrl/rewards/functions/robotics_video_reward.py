"""Disk-artifact boundary for the robotics video reward service."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_ROBOTICS_VIDEO_REWARD_MODEL = "vrl.rewards.models.robotics_video_reward:RoboticsVideoRewardModel"


class RoboticsVideoReward(DiskArtifactRewardFunction):
    """Robotics blend scored from integrity-checked video artifacts."""

    def __init__(
        self,
        *,
        reward_name: str = "robotics_video_reward",
        score_key: str = "robotics_blend",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        self._init_disk_artifact_reward(
            model_factory=_ROBOTICS_VIDEO_REWARD_MODEL,
            request_prefix="robotics-video-reward",
            debug_basename="robotics_video_reward",
            artifact_format=artifact_format,
            reward_name=reward_name,
            score_key=score_key,
            **kwargs,
        )


__all__ = ["RoboticsVideoReward"]
