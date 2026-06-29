"""DINOv2 perceptual Video2World target-similarity reward (local, CPU/GPU)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.target_dino_similarity import TargetDinoSimilarityModel
from vrl.rewards.runtime import LocalRewardRuntime


class TargetDinoSimilarityReward(RewardFunction):
    """Local reward comparing generated video to target media by DINOv2 cosine."""

    default_execution = "inline"

    def __init__(
        self,
        device: str = "",
        reward_name: str = "target_dino_similarity",
        score_key: str = "target_dino_similarity",
        worker_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        cfg = dict(worker_config or {})
        if device:
            cfg.setdefault("device", device)
        model = TargetDinoSimilarityModel(cfg)
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts,
                media_type="video",
            ),
        )


__all__ = ["TargetDinoSimilarityReward"]
