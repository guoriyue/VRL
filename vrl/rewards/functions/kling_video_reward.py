"""Kling VideoReward entry point for world-model RL.

``KlingVideoReward`` writes each sample's media to disk and scores it through
the configured in-process or HTTP runtime. ``DiskArtifactRewardFunction`` is the
transport capability boundary; this file only pins the Kling video-reward model
factory and its defaults.
"""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class KlingVideoReward(DiskArtifactRewardFunction):
    """Kling VideoReward scored from disk artifacts."""

    model_factory = "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel"
    request_prefix = "kling-video-reward"
    debug_basename = "kling_video_reward"
    default_reward_name = "kling_video_reward"
    default_score_key = "overall_reward"


# Production configs must name the reward model directly; the loader keys with
# live reader machinery (model_factory here, import_path in the generic
# module:function loader) are locked out by the production contract gate
# (vrl/config/validation.py consumes this). The former seven-key set also
# locked five names with zero readers anywhere in the repo — a lock that
# cannot protect anything misleads more than it protects, so only live knobs
# stay.
PRODUCTION_LOCKED_WORKER_CONFIG_KEYS = frozenset({"model_factory", "import_path"})

__all__ = ["PRODUCTION_LOCKED_WORKER_CONFIG_KEYS", "KlingVideoReward"]
