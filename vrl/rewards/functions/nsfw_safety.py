"""NSFW safety penalty reward (model-backed over the local transport)."""

from __future__ import annotations

from typing import Any, Literal

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.nsfw_safety import NSFWSafetyRewardModel
from vrl.rewards.runtime import LocalRewardRuntime


class NSFWSafetyReward(RewardFunction):
    """Non-positive NSFW penalty; aggregates per-rollout via the model's batch hook."""

    def __init__(
        self,
        device: str = "cuda",
        execution: Literal["inline", "pool"] = "inline",
        **kwargs: Any,
    ) -> None:
        if str(execution) != "inline":
            raise ValueError(
                "nsfw_safety reward currently supports execution='inline' only",
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


__all__ = ["NSFWSafetyReward"]
