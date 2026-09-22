"""Reference-conditioned EditReward adapter for the standard reward transports."""

from vrl.rewards.base import ModelRewardFunction


class EditReward(ModelRewardFunction):
    model_factory = "vrl.rewards.models.editreward:EditRewardModel"
    name = "editreward"
    default_score_key = "editreward"
    default_artifact_format = "tensor"
    default_media_type = "image"
    worker_config_only = True
