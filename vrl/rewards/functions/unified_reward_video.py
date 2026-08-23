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

from vrl.rewards.base import DiskArtifactRewardFunction


class UnifiedRewardVideoReward(DiskArtifactRewardFunction):
    """UnifiedReward-2.0 video judge scored from disk artifacts."""

    model_factory = "vrl.rewards.models.unified_reward_video:UnifiedRewardVideoModel"
    request_prefix = "unified-reward-video"
    debug_basename = "unified_reward_video"
    default_reward_name = "CodeGoat24/UnifiedReward-2.0-qwen-7b@main"
    default_score_key = "overall"


__all__ = ["UnifiedRewardVideoReward"]
