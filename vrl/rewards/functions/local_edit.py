"""Local-edit reward (EditReward execution; DINOv2 keep outside the edit box as an observation)."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class LocalEditReward(ModelRewardFunction):
    """Was the instruction carried out with nothing outside its region changed."""

    model_factory = "vrl.rewards.models.local_edit:LocalEditRewardModel"
    name = "local_edit"
    default_score_key = "local_edit"
    default_artifact_format = "tensor"
    default_media_type = "image"
    worker_config_only = True


__all__ = ["LocalEditReward"]
