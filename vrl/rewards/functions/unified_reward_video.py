"""UnifiedReward-2.0 video reward function (disk artifacts + in-process runtime).

A plain ``RewardFunction`` on the disk-artifact path whose runtime loads
``CodeGoat24/UnifiedReward-2.0-qwen-7b`` and returns ``alignment`` / ``physics``
/ ``style`` / ``overall`` per artifact. This file only pins the model factory and
the UnifiedReward defaults; transport and disk-vs-in-memory wiring are shared.

Default ``score_key`` is ``overall``; switch to ``physics`` for the sprint's
skirt/cloth plausibility compound, optionally with a task rubric via
``worker_config.rubric_path``.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction

_UNIFIED_REWARD_VIDEO_MODEL = "vrl.rewards.models.unified_reward_video:UnifiedRewardVideoModel"


class UnifiedRewardVideoReward(RewardFunction):
    """UnifiedReward-2.0 video judge scored in-process from disk artifacts."""

    def __init__(self, **kwargs: Any) -> None:
        self._init_disk_artifact_reward(
            model_factory=_UNIFIED_REWARD_VIDEO_MODEL,
            config_key="unified_reward_video",
            request_prefix="unified-reward-video",
            debug_basename="unified_reward_video",
            default_reward_name="CodeGoat24/UnifiedReward-2.0-qwen-7b@main",
            default_score_key="overall",
            default_artifact_format="mp4",
            **kwargs,
        )


__all__ = ["UnifiedRewardVideoReward"]
