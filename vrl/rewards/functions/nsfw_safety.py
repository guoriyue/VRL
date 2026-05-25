"""NSFW safety penalty reward (model-backed over the local transport)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.nsfw_safety_model import NSFWSafetyRewardModel
from vrl.rewards.runtime import LocalRewardRuntime


class NSFWSafetyReward(RewardFunction):
    """Non-positive NSFW penalty; aggregates per-rollout via the model's batch hook."""

    def __init__(
        self,
        device: str = "cuda",
        inference_runtime: str = "local",
        **kwargs: Any,
    ) -> None:
        if str(inference_runtime) != "local":
            raise ValueError(
                "nsfw_safety reward currently supports inference_runtime='local' only",
            )
        # Build eagerly so config validation (threshold/penalty_scale/...) fires now.
        model = NSFWSafetyRewardModel({"device": device, **kwargs})
        self._model = model
        super().__init__(
            reward_name="nsfw_safety",
            score_key="nsfw_safety",
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )

    def probability_batch(self, images: list[Any]) -> list[float]:
        """Raw NSFW probabilities for image-level safety audits."""

        return self._model.probability_batch(images)

    def _probability_from_classifier_result(self, result: Any) -> float:
        return self._model._probability_from_classifier_result(result)


__all__ = ["NSFWSafetyReward"]
