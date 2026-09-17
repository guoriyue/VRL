"""Optical-flow motion-dynamics reward over a video."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class MotionDynamicsReward(ModelRewardFunction):
    """Optical-flow motion-dynamics reward over a video.

    ``worker_config`` is the model's own vocabulary; the resolved device is
    stamped through the ceiling check. In-process the model is built eagerly
    and media rides the request in memory; ``inference.kind=ray`` hands
    the same worker_config to a placement-owned Ray actor.
    """

    model_factory = "vrl.rewards.models.motion_dynamics:MotionDynamicsModel"
    request_prefix = "motion_dynamics"
    debug_basename = "motion_dynamics"
    default_reward_name = "motion_dynamics"
    default_score_key = "motion_dynamics"
    default_artifact_format = "tensor"
    default_media_type = "video"
    eager_model = True
    worker_config_only = True


__all__ = ["MotionDynamicsReward"]
