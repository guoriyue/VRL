"""NSFW safety penalty reward (model-backed over the in-process transport)."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.nsfw_safety import NSFWSafetyRewardModel
from vrl.rewards.runtime import InProcessRewardScorer


class NSFWSafetyReward(InferenceRewardFunction):
    """Non-positive NSFW penalty; aggregates per-sample via the model's batch hook."""

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
            scorer=InProcessRewardScorer(model=model),
        )


__all__ = ["NSFWSafetyReward"]
