"""VideoScore2 reward function (disk artifacts + in-process runtime).

A ``CumemRewardFunction`` on the disk-artifact path (see
``_init_disk_artifact_reward``) whose runtime loads ``TIGER-Lab/VideoScore2`` and
returns ``visual_quality`` / ``text_alignment`` / ``physical_common_sense`` /
``overall`` per artifact. This file only pins the model factory and the
VideoScore2 defaults; transport and disk-vs-in-memory wiring are shared.

The default ``score_key`` is ``physical_common_sense`` (see
``vrl/config/presets/reward/videoscore2.yaml``) so a motion/physics compound gets the
naturalness-and-plausibility axis without also pulling in text alignment.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import CumemRewardFunction

_VIDEOSCORE2_MODEL = "vrl.rewards.models.videoscore2:VideoScore2Model"


class VideoScore2Reward(CumemRewardFunction):
    """VideoScore2 reward scored in-process from disk artifacts."""

    def __init__(self, **kwargs: Any) -> None:
        self._init_disk_artifact_reward(
            model_factory=_VIDEOSCORE2_MODEL,
            config_key="videoscore2",
            request_prefix="videoscore2",
            debug_basename="videoscore2",
            default_reward_name="TIGER-Lab/VideoScore2@main",
            default_score_key="physical_common_sense",
            default_artifact_format="mp4",
            **kwargs,
        )


__all__ = ["VideoScore2Reward"]
