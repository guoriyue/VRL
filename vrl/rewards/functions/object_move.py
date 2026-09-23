"""Object-move edit reward (OWLv2 geometry x DINOv2 consistency)."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class ObjectMoveReward(ModelRewardFunction):
    """Did the named object move as instructed without being copied, lost or redrawn."""

    model_factory = "vrl.rewards.models.object_move:ObjectMoveRewardModel"
    name = "object_move"
    default_score_key = "object_move"
    default_artifact_format = "tensor"
    default_media_type = "image"
    worker_config_only = True


__all__ = ["ObjectMoveReward"]
