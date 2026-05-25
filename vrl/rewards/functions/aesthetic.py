"""Aesthetic reward (model-backed; local or ray transport)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.runtime import make_reward_runtime

_MODEL_FACTORY = "vrl.rewards.models.aesthetic_model:aesthetic_reward_model"


class AestheticReward(RewardFunction):
    """Aesthetic score (CLIP ViT-L/14 + MLP head)."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        model_name: str = "openai/clip-vit-large-patch14",
        inference_runtime: str = "local",
        **kwargs: Any,
    ) -> None:
        worker_config = {
            "device": device,
            "dtype": dtype,
            "model_name": model_name,
            **kwargs,
        }
        super().__init__(
            reward_name="aesthetic",
            score_key="aesthetic",
            runtime=make_reward_runtime(
                inference_runtime, model_factory=_MODEL_FACTORY, worker_config=worker_config,
            ),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )


__all__ = ["AestheticReward"]
