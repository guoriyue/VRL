"""Framework binding for text-conditioned CountGD exact-object-count reward."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class CountGDReward(ModelRewardFunction):
    """Count ``metadata.object_class`` against ``metadata.expected_count``."""

    model_factory = "vrl.rewards.models.countgd:CountGDModel"
    name = "countgd"
    default_score_key = "countgd"
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["CountGDReward"]
