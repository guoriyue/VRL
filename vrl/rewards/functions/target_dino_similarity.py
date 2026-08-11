"""DINOv2 perceptual Video2World target-similarity reward (local, CPU/GPU)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction, reward_worker_config_with_device
from vrl.rewards.models.target_dino_similarity import TargetDinoSimilarityModel
from vrl.rewards.runtime import InProcessRewardInferenceRuntime


class TargetDinoSimilarityReward(InferenceRewardFunction):
    """Local reward comparing generated video to target media by DINOv2 cosine."""

    def __init__(
        self,
        device: str = "",
        reward_name: str = "target_dino_similarity",
        score_key: str = "target_dino_similarity",
        worker_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = reward_worker_config_with_device(worker_config, device=str(device))
        model = TargetDinoSimilarityModel(cfg)
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            inference_runtime=InProcessRewardInferenceRuntime(model=model),
        )


__all__ = ["TargetDinoSimilarityReward"]
