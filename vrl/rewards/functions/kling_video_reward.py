"""Kling VideoReward entry point for world-model RL.

``KlingVideoReward`` writes each rollout's media to disk and scores it through a
in-process reward runtime (the model may still need its own GPU). It is a
``CumemRewardFunction`` configured for the disk-artifact path via
``_init_disk_artifact_reward`` — there is no separate "ray reward" type; transport
is the ``make_reward_runtime`` adapter and disk-vs-in-memory is just which
artifact builder the reward is initialized with. This file only pins the Kling
video-reward model factory and its defaults.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import CumemRewardFunction

_KLING_VIDEO_REWARD_MODEL = "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel"


class KlingVideoReward(CumemRewardFunction):
    """Kling VideoReward scored in-process from disk artifacts."""

    def __init__(self, **kwargs: Any) -> None:
        self._init_disk_artifact_reward(
            model_factory=_KLING_VIDEO_REWARD_MODEL,
            config_key="kling_video_reward",
            request_prefix="kling-video-reward",
            debug_basename="kling_video_reward",
            # mp4, not .pt: decord can't read torch.save tensors, so a video
            # reward that omits artifact_format must still get a real container.
            default_artifact_format="mp4",
            **kwargs,
        )


__all__ = ["KlingVideoReward"]
