"""PickScore preference reward (model-backed; local or ray transport)."""

from __future__ import annotations

from typing import Any, Literal

from vrl.rewards.base import RewardFunction


class PickScoreReward(RewardFunction):
    """PickScore v1 (CLIP ViT-H/14), normalised by /26 to roughly [0, 1]."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_name: str = "yuvalkirstain/PickScore_v1",
        execution: Literal["inline"] = "inline",
        **kwargs: Any,
    ) -> None:
        worker_config = {
            "device": device,
            "dtype": dtype,
            "processor_name": processor_name,
            "model_name": model_name,
            **kwargs,
        }
        self._init_reward_model(
            reward_name="pickscore",
            score_key="pickscore",
            model_factory="vrl.rewards.models.pickscore:pickscore_reward_model",
            worker_config=worker_config,
            execution=execution,
        )


__all__ = ["PickScoreReward"]
