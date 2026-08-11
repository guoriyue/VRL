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

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_VIDEOSCORE2_MODEL = "vrl.rewards.models.videoscore2:VideoScore2Model"


class VideoScore2Reward(DiskArtifactRewardFunction):
    """VideoScore2 reward scored from disk artifacts."""

    def __init__(
        self,
        *,
        reward_name: str = "TIGER-Lab/VideoScore2@main",
        score_key: str = "physical_common_sense",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_factory=_VIDEOSCORE2_MODEL,
            request_prefix="videoscore2",
            debug_basename="videoscore2",
            reward_name=reward_name,
            score_key=score_key,
            artifact_format=artifact_format,
            **kwargs,
        )


__all__ = ["VideoScore2Reward"]
