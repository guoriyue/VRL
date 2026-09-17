"""WD tagger (onnxruntime, CPU) tag-adherence reward."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction


class WDTaggerReward(ModelRewardFunction):
    """WD tagger (onnxruntime, CPU) tag-adherence reward."""

    model_factory = "vrl.rewards.models.wd_tagger:WDTaggerRewardModel"
    name = "wd_tagger"
    default_score_key = "wd_tagger_dense"
    score_keys = ("wd_tagger_dense", "wd_tagger_recall")
    default_artifact_format = "tensor"
    default_media_type = "image"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """CPU-only compute; never claim the resource-resolved GPU."""
        return "cpu"


__all__ = ["WDTaggerReward"]
