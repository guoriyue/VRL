"""Kling VideoReward entry point for world-model RL.

``KlingVideoReward`` writes each sample's media to disk and scores it through
the configured in-process or HTTP runtime. ``DiskArtifactRewardFunction`` is the
transport capability boundary; this file only pins the Kling video-reward model
factory and its defaults.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_KLING_VIDEO_REWARD_MODEL = "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel"


class KlingVideoReward(DiskArtifactRewardFunction):
    """Kling VideoReward scored from disk artifacts."""

    def __init__(
        self,
        *,
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        # mp4, not .pt: decord can't read torch.save tensors, so a video
        # reward that omits artifact_format must still get a real container.
        self._init_disk_artifact_reward(
            model_factory=_KLING_VIDEO_REWARD_MODEL,
            request_prefix="kling-video-reward",
            debug_basename="kling_video_reward",
            artifact_format=artifact_format,
            **kwargs,
        )


__all__ = ["KlingVideoReward"]
