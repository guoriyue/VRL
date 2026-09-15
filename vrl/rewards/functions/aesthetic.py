"""Aesthetic score (CLIP ViT-L/14 + MLP head)."""

from __future__ import annotations

from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer


class AestheticReward(DiskArtifactRewardFunction):
    """Aesthetic score (CLIP ViT-L/14 + MLP head).

    Model paths and dtype come from YAML; this binding pins the factory and
    the transport: in-process the runtime builds the model on the resolved
    device (CuMem-pooled under a shared GPU), ``inference.kind=service`` hands
    the same worker_config to a driver-launched service.
    """

    model_factory = "vrl.rewards.models.aesthetic:AestheticRewardModel"
    request_prefix = "aesthetic"
    debug_basename = "aesthetic"
    default_reward_name = "aesthetic"
    default_score_key = "aesthetic"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        model_name: str = "openai/clip-vit-large-patch14",
        score_key: str = "aesthetic",
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
        **kwargs: Any,
    ) -> None:
        worker_config = {
            "dtype": dtype,
            "model_name": model_name,
            **kwargs,
        }
        super().__init__(
            reward_name="aesthetic",
            score_key=score_key,
            worker_config=worker_config,
            device=device,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )


__all__ = ["AestheticReward"]
