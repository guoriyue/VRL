"""VideoScore2 reward function (disk artifacts + in-process runtime).

A ``DiskArtifactRewardFunction`` on the disk-artifact path whose runtime loads ``TIGER-Lab/VideoScore2`` and
returns ``visual_quality`` / ``text_alignment`` / ``physical_common_sense`` /
``overall`` per artifact. This file only pins the model factory and the
VideoScore2 defaults; transport and disk-vs-in-memory wiring are shared.

The default ``score_key`` is ``physical_common_sense`` (see
``vrl/config/presets/reward/videoscore2.yaml``) so a motion/physics compound gets the
naturalness-and-plausibility axis without also pulling in text alignment.
"""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class VideoScore2Reward(DiskArtifactRewardFunction):
    """VideoScore2 reward scored from disk artifacts."""

    model_factory = "vrl.rewards.models.videoscore2:VideoScore2Model"
    request_prefix = "videoscore2"
    debug_basename = "videoscore2"
    default_reward_name = "TIGER-Lab/VideoScore2@main"
    default_score_key = "physical_common_sense"


__all__ = ["VideoScore2Reward"]
