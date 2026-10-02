"""Model-free exposure/saturation anchor against the prompt's reference image, on CPU."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction


class TonalAnchorReward(ModelRewardFunction):
    """Penalty for mean-luma / saturation drift from ``metadata.judge_reference`` (CPU)."""

    model_factory = "vrl.rewards.models.tonal_anchor:TonalAnchorRewardModel"
    name = "tonal_anchor"
    default_score_key = "tonal_anchor"
    default_artifact_format = "tensor"
    default_media_type = "image"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """CPU-only compute; never claim the resource-resolved GPU."""
        return "cpu"


__all__ = ["TonalAnchorReward"]
