"""Aesthetic score (CLIP ViT-L/14 + MLP head)."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class AestheticReward(ModelRewardFunction):
    """Aesthetic score (CLIP ViT-L/14 + MLP head).

    Model paths and dtype come from YAML; this binding pins the factory and
    the transport: in-process the runtime builds the model on the resolved
    device (CuMem-pooled under a shared GPU), ``inference.kind=ray`` hands
    the same worker_config to a placement-owned Ray actor.
    """

    model_factory = "vrl.rewards.models.aesthetic:AestheticRewardModel"
    request_prefix = "aesthetic"
    debug_basename = "aesthetic"
    default_reward_name = "aesthetic"
    default_score_key = "aesthetic"
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["AestheticReward"]
