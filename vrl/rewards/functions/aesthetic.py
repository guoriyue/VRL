"""Aesthetic score (SigLIP Aesthetic Predictor V2.5)."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class AestheticReward(ModelRewardFunction):
    """Aesthetic score (SigLIP Aesthetic Predictor V2.5)."""

    model_factory = "vrl.rewards.models.aesthetic:AestheticRewardModel"
    name = "aesthetic"
    default_score_key = "aesthetic"


__all__ = ["AestheticReward"]
