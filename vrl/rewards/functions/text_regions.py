"""Standard training/service adapter for explicit text-region diagnostics."""

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction


class TextRegionsReward(ModelRewardFunction):
    model_factory = "vrl.rewards.models.text_regions:TextRegionsRewardModel"
    name = "text_regions"
    default_score_key = "text_similarity"
    default_media_type = "image"
    default_artifact_format = "tensor"
    worker_config_only = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """Use the existing CPU OCR backend, including in isolated Ray workers."""
        return "cpu"
