"""Framework binding for text-conditioned CountGD exact-object-count reward."""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class CountGDReward(DiskArtifactRewardFunction):
    """Count ``metadata.object_class`` against ``metadata.expected_count``."""

    model_factory = "vrl.rewards.models.countgd:CountGDModel"
    request_prefix = "countgd"
    debug_basename = "countgd"
    default_reward_name = "countgd"
    default_score_key = "countgd"
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["CountGDReward"]
