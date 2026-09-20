"""Kling VideoReward entry point for world-model RL.

``KlingVideoReward`` forwards sample media to the configured inference runtime.
The file-only Kling model receives a scorer-local MP4. ``ModelRewardFunction`` is the
transport capability boundary; this file only pins the Kling video-reward model
factory and its defaults.
"""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class KlingVideoReward(ModelRewardFunction):
    """Kling VideoReward scored through the configured runtime."""

    model_factory = "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel"
    name = "kling_video_reward"
    default_score_key = "overall_reward"


__all__ = ["KlingVideoReward"]
