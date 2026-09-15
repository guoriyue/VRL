"""PickScore (CLIP-H) prompt-image preference reward."""

from __future__ import annotations

from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer


class PickScoreReward(DiskArtifactRewardFunction):
    """PickScore (CLIP-H) prompt-image preference reward.

    Model paths and dtype come from YAML; this binding pins the factory and
    the transport: in-process the runtime builds the model on the resolved
    device (CuMem-pooled under a shared GPU), ``inference.kind=service`` hands
    the same worker_config to a driver-launched service.
    """

    model_factory = "vrl.rewards.models.pickscore:PickScoreRewardModel"
    request_prefix = "pickscore"
    debug_basename = "pickscore"
    default_reward_name = "pickscore"
    default_score_key = "pickscore"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_name: str = "yuvalkirstain/PickScore_v1",
        score_key: str = "pickscore",
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
            "processor_name": processor_name,
            "model_name": model_name,
            **kwargs,
        }
        super().__init__(
            reward_name="pickscore",
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


__all__ = ["PickScoreReward"]
