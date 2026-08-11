"""UnifiedReward-2.0 video reward function (disk artifacts + in-process runtime).

A ``DiskArtifactRewardFunction`` on the disk-artifact path whose runtime loads
``CodeGoat24/UnifiedReward-2.0-qwen-7b`` and returns ``alignment`` / ``physics``
/ ``style`` / ``overall`` per artifact. This file only pins the model factory and
the UnifiedReward defaults; transport and disk-vs-in-memory wiring are shared.

Default ``score_key`` is ``overall``; switch to ``physics`` for the sprint's
skirt/cloth plausibility compound, optionally with a task rubric via
``worker_config.rubric_path``.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_UNIFIED_REWARD_VIDEO_MODEL = "vrl.rewards.models.unified_reward_video:UnifiedRewardVideoModel"


class UnifiedRewardVideoReward(DiskArtifactRewardFunction):
    """UnifiedReward-2.0 video judge scored from disk artifacts."""

    def __init__(
        self,
        *,
        reward_name: str = "CodeGoat24/UnifiedReward-2.0-qwen-7b@main",
        score_key: str = "overall",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_factory=_UNIFIED_REWARD_VIDEO_MODEL,
            request_prefix="unified-reward-video",
            debug_basename="unified_reward_video",
            reward_name=reward_name,
            score_key=score_key,
            artifact_format=artifact_format,
            **kwargs,
        )


__all__ = ["UnifiedRewardVideoReward"]
