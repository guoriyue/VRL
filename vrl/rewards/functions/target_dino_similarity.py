"""DINOv2 similarity between generated frames and a target image."""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class TargetDinoSimilarityReward(DiskArtifactRewardFunction):
    """DINOv2 similarity between generated frames and a target image.

    ``worker_config`` is the model's own vocabulary; the resolved device is
    stamped through the ceiling check. In-process the model is built eagerly
    and media rides the request in memory; ``inference.kind=service`` hands
    the same worker_config to a driver-launched service.
    """

    model_factory = "vrl.rewards.models.target_dino_similarity:TargetDinoSimilarityModel"
    request_prefix = "target_dino_similarity"
    debug_basename = "target_dino_similarity"
    default_reward_name = "target_dino_similarity"
    default_score_key = "target_dino_similarity"
    default_artifact_format = "tensor"
    default_media_type = "video"
    in_process_media = "memory"
    eager_model = True
    worker_config_only = True


__all__ = ["TargetDinoSimilarityReward"]
