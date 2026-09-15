"""Model-free image sharpness (Laplacian variance) reward on CPU."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer


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

    def __init__(
        self,
        device: str = "cpu",
        *,
        score_key: str = "image_sharpness",
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            reward_name="image_sharpness",
            score_key=score_key,
            worker_config=kwargs,
            device=device,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )


__all__ = ["ImageSharpnessReward"]
