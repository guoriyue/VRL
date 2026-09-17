"""VideoScore2 reward function.

A ``ModelRewardFunction`` whose runtime loads ``TIGER-Lab/VideoScore2`` and
returns ``visual_quality`` / ``text_alignment`` / ``physical_common_sense`` /
``overall`` per artifact. This file only pins the model factory and the
VideoScore2 defaults; transport and scorer-local media adaptation are shared.

The default ``score_key`` is ``physical_common_sense`` (see
``vrl/config/presets/reward/videoscore2.yaml``) so a motion/physics compound gets the
naturalness-and-plausibility axis without also pulling in text alignment.
"""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class VideoScore2Reward(ModelRewardFunction):
    """VideoScore2 reward scored through the configured runtime."""

    model_factory = "vrl.rewards.models.videoscore2:VideoScore2Model"
    name = "videoscore2"
    default_reward_name = "TIGER-Lab/VideoScore2@main"
    default_score_key = "physical_common_sense"


__all__ = ["VideoScore2Reward"]
