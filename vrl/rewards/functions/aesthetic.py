"""Aesthetic reward (model-backed, scored in-process)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import CumemRewardFunction


class AestheticReward(CumemRewardFunction):
    """Aesthetic score (CLIP ViT-L/14 + MLP head)."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        model_name: str = "openai/clip-vit-large-patch14",
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
            model_factory="vrl.rewards.models.aesthetic:AestheticRewardModel",
            worker_config=worker_config,
        )


__all__ = ["AestheticReward"]
