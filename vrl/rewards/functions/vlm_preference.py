"""Generative VLM preference reward (per-criterion yes-probabilities from a large judge)."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class VLMPreferenceReward(ModelRewardFunction):
    """Weighted mean of a VLM judge's P(yes) over aesthetic, structure, cleanliness and alignment."""

    model_factory = "vrl.rewards.models.vlm_preference:VLMPreferenceRewardModel"
    name = "vlm_preference"
    default_score_key = "vlm_preference"
    default_artifact_format = "tensor"
    default_media_type = "image"
    worker_config_only = True


__all__ = ["VLMPreferenceReward"]
