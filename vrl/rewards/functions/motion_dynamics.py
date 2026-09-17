"""Optical-flow motion-dynamics reward over a video."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class MotionDynamicsReward(ModelRewardFunction):
    """Optical-flow motion-dynamics reward over a video."""

    model_factory = "vrl.rewards.models.motion_dynamics:MotionDynamicsModel"
    name = "motion_dynamics"
    default_score_key = "motion_dynamics"
    default_artifact_format = "tensor"
    default_media_type = "video"
    eager_model = True
    worker_config_only = True


__all__ = ["MotionDynamicsReward"]
