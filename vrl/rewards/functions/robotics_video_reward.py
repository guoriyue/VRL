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

from vrl.rewards.base import DiskArtifactRewardFunction


class RoboticsVideoReward(DiskArtifactRewardFunction):
    """Robotics blend scored from integrity-checked video artifacts."""

    model_factory = "vrl.rewards.models.robotics_video_reward:RoboticsVideoRewardModel"
    request_prefix = "robotics-video-reward"
    debug_basename = "robotics_video_reward"
    default_reward_name = "robotics_video_reward"
    default_score_key = "robotics_blend"


__all__ = ["RoboticsVideoReward"]
