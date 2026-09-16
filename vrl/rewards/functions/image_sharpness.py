"""Model-free image sharpness (Laplacian variance) reward on CPU."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction


class ImageSharpnessReward(DiskArtifactRewardFunction):
    """Model-free image sharpness (Laplacian variance) reward on CPU.

    In-process the model is built here and media rides the request in memory;
    ``inference.kind=service`` hands the same kwargs to a driver-launched
    service that scores this reward's ``.pt`` artifacts.
    """

    model_factory = "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel"
    request_prefix = "image_sharpness"
    debug_basename = "image_sharpness"
    default_reward_name = "image_sharpness"
    default_score_key = "image_sharpness"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """CPU-only compute; never claim the resource-resolved GPU."""
        return "cpu"


__all__ = ["ImageSharpnessReward"]
