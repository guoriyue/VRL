"""CLIP text-image similarity reward (model-backed; local or ray transport)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction


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
        self._init_reward_model(
            reward_name="clipscore",
            score_key="clipscore",
            model_factory="vrl.rewards.models.clip:clip_score_reward_model",
            worker_config=worker_config,
            inference_runtime=inference_runtime,
        )


__all__ = ["CLIPScoreReward"]
