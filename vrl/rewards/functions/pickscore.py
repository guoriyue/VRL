"""PickScore (CLIP-H) prompt-image preference reward."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class PickScoreReward(ModelRewardFunction):
    """PickScore (CLIP-H) prompt-image preference reward.

    Model paths and dtype come from YAML; this binding pins the factory and
    the transport: in-process the runtime builds the model on the resolved
    device (CuMem-pooled under a shared GPU), ``inference.kind=ray`` hands
    the same worker_config to a placement-owned Ray actor.
    """

    model_factory = "vrl.rewards.models.pickscore:PickScoreRewardModel"
    request_prefix = "pickscore"
    debug_basename = "pickscore"
    default_reward_name = "pickscore"
    default_score_key = "pickscore"
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["PickScoreReward"]
