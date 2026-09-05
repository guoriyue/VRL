"""Framework binding for deterministic CountGD exact-person-count reward."""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class CountGDPersonCountReward(DiskArtifactRewardFunction):
    """Score exact adherence to ``metadata.expected_people`` for one image."""

    model_factory = "vrl.rewards.models.countgd_person_count:CountGDPersonCountModel"
    request_prefix = "countgd-person-count"
    debug_basename = "countgd_person_count"
    default_reward_name = "countgd-person-count"
    default_score_key = "countgd_person_count"
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["CountGDPersonCountReward"]
