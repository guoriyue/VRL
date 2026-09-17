"""DINOv2 similarity between generated frames and a target image."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class TargetDinoSimilarityReward(ModelRewardFunction):
    """DINOv2 similarity between generated frames and a target image."""

    model_factory = "vrl.rewards.models.target_dino_similarity:TargetDinoSimilarityModel"
    name = "target_dino_similarity"
    default_score_key = "target_dino_similarity"
    default_artifact_format = "tensor"
    default_media_type = "video"
    eager_model = True
    worker_config_only = True


__all__ = ["TargetDinoSimilarityReward"]
