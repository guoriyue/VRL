"""Aesthetic reward (model-backed, scored in-process).

Thin function-layer binding registered as ``aesthetic``: the registry builds it
from YAML, and this file only pins the model factory plus the CLIP backbone
defaults. The scoring network — the LAION aesthetic predictor MLP head
(``sac+logos+ava1-l14-linearMSE.pth``, packaged in ``vrl.rewards.assets``) over
CLIP ViT-L/14 image embeddings — lives in ``vrl.rewards.models.aesthetic`` and
is built lazily by the scorer from the ``model_factory`` dotted path.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import CumemRewardFunction
from vrl.rewards.runtime import build_reward_scorer


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
        super().__init__(
            reward_name="aesthetic",
            score_key="aesthetic",
            scorer=build_reward_scorer(
                {
                    **worker_config,
                    "model_factory": "vrl.rewards.models.aesthetic:AestheticRewardModel",
                },
            ),
        )


__all__ = ["AestheticReward"]
