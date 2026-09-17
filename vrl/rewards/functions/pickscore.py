"""PickScore (CLIP-H) prompt-image preference reward."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class PickScoreReward(ModelRewardFunction):
    """PickScore (CLIP-H) prompt-image preference reward."""

    model_factory = "vrl.rewards.models.pickscore:PickScoreRewardModel"
    name = "pickscore"
    default_score_key = "pickscore"
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["PickScoreReward"]
