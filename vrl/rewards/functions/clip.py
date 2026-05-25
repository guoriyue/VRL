"""CLIP text-image similarity reward (model-backed; local or ray transport)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.runtime import make_reward_runtime

_MODEL_FACTORY = "vrl.rewards.models.clip:clip_score_reward_model"


class CLIPScoreReward(RewardFunction):
    """CLIP text-image cosine similarity / 30 (normalised to ~[0, 1])."""

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "openai/clip-vit-large-patch14",
        inference_runtime: str = "local",
        **kwargs: Any,
    ) -> None:
        worker_config = {
            "device": device,
            "model_name": model_name,
            **kwargs,
        }
        super().__init__(
            reward_name="clipscore",
            score_key="clipscore",
            runtime=make_reward_runtime(
                inference_runtime, model_factory=_MODEL_FACTORY, worker_config=worker_config,
            ),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )


__all__ = ["CLIPScoreReward"]
