"""Training adapters for the known-layer decomposition measurement.

One CPU model computes every axis; each registered component selects one so a
recipe can weight hidden completion, object matching, visible retention and
edit holes separately (``advantage_combine=normalized_sum`` standardizes each).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction
from vrl.rewards.models.layered_reference import SCORE_KEYS


class LayeredReferenceReward(ModelRewardFunction):
    """Aggregate key: the product of every known-layer quality axis."""

    model_factory = "vrl.rewards.models.layered_reference:LayeredReferenceRewardModel"
    name = "layered_reference"
    default_score_key = "layered_reference"
    default_media_type = "image"
    default_artifact_format = "tensor"
    worker_config_only = True
    score_keys = SCORE_KEYS

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """Pure numpy; never claims the trainer GPU."""
        return "cpu"


class LayeredHiddenCompletionReward(LayeredReferenceReward):
    name = "layered_hidden_completion"
    default_score_key = "layered_hidden_completion"


class LayeredObjectMatchReward(LayeredReferenceReward):
    name = "layered_object_match"
    default_score_key = "layered_object_match"


class LayeredVisibleRetentionReward(LayeredReferenceReward):
    name = "layered_visible_retention"
    default_score_key = "layered_visible_retention"


class LayeredEditHolesReward(LayeredReferenceReward):
    name = "layered_edit_holes"
    default_score_key = "layered_edit_holes"


class LayeredCompositeCoverageReward(LayeredReferenceReward):
    name = "layered_composite_coverage"
    default_score_key = "layered_composite_coverage"


class LayeredInferableCompletionReward(LayeredReferenceReward):
    name = "layered_inferable_completion"
    default_score_key = "layered_inferable_completion"


class LayeredUnknowablePlausibilityReward(LayeredReferenceReward):
    name = "layered_unknowable_plausibility"
    default_score_key = "layered_unknowable_plausibility"


__all__ = [
    "LayeredCompositeCoverageReward",
    "LayeredEditHolesReward",
    "LayeredHiddenCompletionReward",
    "LayeredInferableCompletionReward",
    "LayeredObjectMatchReward",
    "LayeredReferenceReward",
    "LayeredUnknowablePlausibilityReward",
    "LayeredVisibleRetentionReward",
]
