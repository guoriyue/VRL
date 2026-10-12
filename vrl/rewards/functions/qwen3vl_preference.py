"""Qwen3-VL preference reward (per-criterion answer probabilities from a Qwen3-VL judge)."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class Qwen3VLPreferenceReward(ModelRewardFunction):
    """Weighted mean of a Qwen3-VL judge's P(yes) over aesthetic, structure, cleanliness and alignment."""

    model_factory = "vrl.rewards.models.qwen3vl_preference:Qwen3VLPreferenceRewardModel"
    name = "qwen3vl_preference"
    default_score_key = "qwen3vl_preference"
    worker_config_only = True


__all__ = ["Qwen3VLPreferenceReward"]
