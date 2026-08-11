"""Disk-artifact boundary for the robotics video reward service.

Thin function-layer binding registered as ``robotics_video_reward``: the
registry builds it from YAML, and this file only pins the model factory and the
``robotics_blend`` score key. The composite model (DINOv2 target anchor + RAFT
motion floor + Kling text alignment, and why exactly those three) lives in
``vrl.rewards.models.robotics_video_reward``; it is typically served behind the
HTTP transport on its own GPU (``robotics_video_reward_http.yaml`` preset),
which is why media is materialized to disk before scoring.
"""

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
