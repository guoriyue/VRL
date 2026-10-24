"""Layer-decomposition reward against a row's object masks and truth layers, on CPU."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction


class LayeredGroupingReward(ModelRewardFunction):
    """Layer-decomposition reward against a row's object masks and truth layers, on CPU."""

    model_factory = "vrl.rewards.models.layered_grouping:LayeredGroupingRewardModel"
    name = "layered_grouping"
    default_score_key = "layered_grouping"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """Mask and pixel arithmetic only; never claim the resource-resolved GPU."""
        return "cpu"


__all__ = ["LayeredGroupingReward"]
