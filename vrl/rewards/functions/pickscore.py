"""PickScore preference reward (model-backed; local or ray transport)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.runtime import make_reward_runtime

_MODEL_FACTORY = "vrl.rewards.models.pickscore_model:pickscore_reward_model"


class PickScoreReward(RewardFunction):
    """PickScore v1 (CLIP ViT-H/14), normalised by /26 to roughly [0, 1]."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_name: str = "yuvalkirstain/PickScore_v1",
        inference_runtime: str = "local",
        **kwargs: Any,
    ) -> None:
        worker_config = {
            "device": device,
            "dtype": dtype,
            "processor_name": processor_name,
            "model_name": model_name,
            **kwargs,
        }
        super().__init__(
            reward_name="pickscore",
            score_key="pickscore",
            runtime=make_reward_runtime(
                inference_runtime, model_factory=_MODEL_FACTORY, worker_config=worker_config,
            ),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )


__all__ = ["PickScoreReward"]
