"""Kling VideoReward entry point for world-model RL.

``KlingVideoReward`` writes each sample's media to disk and scores it through
the configured in-process or HTTP runtime. ``DiskArtifactRewardFunction`` is the
transport capability boundary; this file only pins the Kling video-reward model
factory and its defaults.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_KLING_VIDEO_REWARD_MODEL = "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel"


class KlingVideoReward(DiskArtifactRewardFunction):
    """Kling VideoReward scored from disk artifacts."""

    def __init__(
        self,
        *,
        reward_name: str = "kling_video_reward",
        score_key: str = "overall_reward",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        # mp4, not .pt: decord can't read torch.save tensors, so a video
        # reward that omits artifact_format must still get a real container.
        super().__init__(
            model_factory=_KLING_VIDEO_REWARD_MODEL,
            request_prefix="kling-video-reward",
            debug_basename="kling_video_reward",
            artifact_format=artifact_format,
            reward_name=reward_name,
            score_key=score_key,
            **kwargs,
        )


# Production configs must name the reward model directly; the loader keys with
# live reader machinery (model_factory here, import_path in the generic
# module:function loader) are locked out by the production contract gate
# (vrl/config/validation.py consumes this). The former seven-key set also
# locked five names with zero readers anywhere in the repo — a lock that
# cannot protect anything misleads more than it protects, so only live knobs
# stay.
PRODUCTION_LOCKED_WORKER_CONFIG_KEYS = frozenset({"model_factory", "import_path"})

__all__ = ["PRODUCTION_LOCKED_WORKER_CONFIG_KEYS", "KlingVideoReward"]
