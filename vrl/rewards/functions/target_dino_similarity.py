"""DINOv2 perceptual Video2World target-similarity reward (local, CPU/GPU)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction, resolve_reward_component_device
from vrl.rewards.models.target_dino_similarity import TargetDinoSimilarityModel
from vrl.rewards.runtime import InProcessRewardRuntime


class TargetDinoSimilarityReward(RewardFunction):
    """Local reward comparing generated video to target media by DINOv2 cosine."""

    def __init__(
        self,
        device: str = "",
        reward_name: str = "target_dino_similarity",
        score_key: str = "target_dino_similarity",
        worker_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = dict(worker_config or {})
        if device:
            configured_device = str(cfg.get("device", "")).strip()
            cfg["device"] = resolve_reward_component_device(
                resolved_device=str(device),
                overrides=[("worker_config.device", configured_device)],
            )
        model = TargetDinoSimilarityModel(cfg)
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            runtime=InProcessRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts,
                media_type="video",
            ),
        )


__all__ = ["TargetDinoSimilarityReward"]
