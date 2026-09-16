"""WD tagger (onnxruntime, CPU) tag-adherence reward."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction


class WDTaggerReward(DiskArtifactRewardFunction):
    """WD tagger (onnxruntime, CPU) tag-adherence reward.

    In-process the model is built here and media rides the request in memory;
    ``inference.kind=service`` hands the same kwargs to a driver-launched
    service that scores this reward's ``.pt`` artifacts.
    """

    model_factory = "vrl.rewards.models.wd_tagger:WDTaggerRewardModel"
    request_prefix = "wd_tagger"
    debug_basename = "wd_tagger"
    default_reward_name = "wd_tagger"
    default_score_key = "wd_tagger_dense"
    score_keys = ("wd_tagger_dense", "wd_tagger_recall")
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """CPU-only compute; never claim the resource-resolved GPU."""
        return "cpu"


__all__ = ["WDTaggerReward"]
