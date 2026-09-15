"""DINOv2 similarity between generated frames and a target image."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer


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

    def __init__(
        self,
        device: str = "",
        *,
        reward_name: str = "target_dino_similarity",
        score_key: str = "target_dino_similarity",
        worker_config: Mapping[str, Any] | None = None,
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
    ) -> None:
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            worker_config=dict(worker_config or {}),
            device=device or None,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )


__all__ = ["TargetDinoSimilarityReward"]
