"""NSFW safety penalty reward (model-backed over the in-process transport)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.nsfw_safety import NSFWSafetyRewardModel
from vrl.rewards.runtime import InProcessRewardRuntime


class NSFWSafetyReward(RewardFunction):
    """Non-positive NSFW penalty; aggregates per-rollout via the model's batch hook."""

    device_config_key = "classifier_device"

    def __init__(
        self,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        # Build eagerly so config validation (threshold/penalty_scale/...) fires now.
        model = NSFWSafetyRewardModel({"device": device, **kwargs})
        super().__init__(
            reward_name="nsfw_safety",
            score_key="nsfw_safety",
            runtime=InProcessRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts,
                media_type="image",
            ),
        )


__all__ = ["NSFWSafetyReward"]
