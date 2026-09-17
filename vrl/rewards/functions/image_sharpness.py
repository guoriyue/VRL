"""Model-free image sharpness (Laplacian variance) reward on CPU."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction


class ImageSharpnessReward(ModelRewardFunction):
    """Model-free image sharpness (Laplacian variance) reward on CPU."""

    model_factory = "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel"
    name = "image_sharpness"
    default_score_key = "image_sharpness"
    default_artifact_format = "tensor"
    default_media_type = "image"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """CPU-only compute; never claim the resource-resolved GPU."""
        return "cpu"


__all__ = ["ImageSharpnessReward"]
