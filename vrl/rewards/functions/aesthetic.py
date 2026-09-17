"""Aesthetic score (CLIP ViT-L/14 + MLP head)."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class AestheticReward(ModelRewardFunction):
    """Aesthetic score (CLIP ViT-L/14 + MLP head)."""

    model_factory = "vrl.rewards.models.aesthetic:AestheticRewardModel"
    name = "aesthetic"
    default_score_key = "aesthetic"
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["AestheticReward"]
