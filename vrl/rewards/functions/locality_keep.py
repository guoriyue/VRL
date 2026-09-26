"""Training/service adapter for the outside-the-boxes pixel keep measurement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction


class LocalityKeepReward(ModelRewardFunction):
    model_factory = "vrl.rewards.models.locality_keep:LocalityKeepRewardModel"
    name = "locality_keep"
    default_score_key = "locality_keep"
    default_media_type = "image"
    default_artifact_format = "tensor"
    worker_config_only = True
    score_keys = ("locality_keep", "locality_psnr", "locality_hf_ratio")

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """Pure numpy; never claims the trainer GPU."""
        return "cpu"


__all__ = ["LocalityKeepReward"]
