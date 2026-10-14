"""NSFW safety penalty from a Falconsai image classifier over sampled frames."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class NSFWSafetyReward(ModelRewardFunction):
    """NSFW safety penalty from a Falconsai image classifier over sampled frames."""

    model_factory = "vrl.rewards.models.nsfw_safety:NSFWSafetyRewardModel"
    name = "nsfw_safety"
    default_score_key = "nsfw_safety"
    eager_model = True

    # The classifier's device key differs from the generic ceiling key.
    device_config_key = "classifier_device"


__all__ = ["NSFWSafetyReward"]
