"""PickScore (CLIP-H) prompt-image preference reward."""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class PickScoreReward(DiskArtifactRewardFunction):
    """PickScore (CLIP-H) prompt-image preference reward.

    Model paths and dtype come from YAML; this binding pins the factory and
    the transport: in-process the runtime builds the model on the resolved
    device (CuMem-pooled under a shared GPU), ``inference.kind=service`` hands
    the same worker_config to a driver-launched service.
    """

    model_factory = "vrl.rewards.models.pickscore:PickScoreRewardModel"
    request_prefix = "pickscore"
    debug_basename = "pickscore"
    default_reward_name = "pickscore"
    default_score_key = "pickscore"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"


__all__ = ["PickScoreReward"]
