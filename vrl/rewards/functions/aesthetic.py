"""Aesthetic reward (model-backed; local or ray transport)."""

from __future__ import annotations

from typing import Any, Literal

from vrl.rewards.base import RewardFunction


class AestheticReward(RewardFunction):
    """Aesthetic score (CLIP ViT-L/14 + MLP head)."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        model_name: str = "openai/clip-vit-large-patch14",
        execution: Literal["inline", "pool"] = "inline",
        **kwargs: Any,
    ) -> None:
        worker_config = {
            "device": device,
            "dtype": dtype,
            "model_name": model_name,
            **kwargs,
        }
        self._init_reward_model(
            reward_name="aesthetic",
            score_key="aesthetic",
            model_factory="vrl.rewards.models.aesthetic:aesthetic_reward_model",
            worker_config=worker_config,
            execution=execution,
        )


__all__ = ["AestheticReward"]
