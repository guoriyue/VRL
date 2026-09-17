"""Codex CLI subprocess judge scoring an image against its prompt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import ModelRewardFunction
from vrl.rewards.models.codex_image_qa import (
    DEFAULT_PROMPT_TEMPLATE,
    _extract_score_from_text,
    _render_prompt_template,
)


class CodexImageQAReward(ModelRewardFunction):
    """Codex CLI subprocess judge scoring an image against its prompt."""

    model_factory = "vrl.rewards.models.codex_image_qa:CodexImageQARewardModel"
    name = "codex_image_qa"
    default_score_key = "codex_image_qa"
    default_artifact_format = "tensor"
    default_media_type = "image"
    eager_model = True

    @classmethod
    def resolve_execution_device(cls, *, device: str, kwargs: Mapping[str, Any]) -> str:
        """CPU-only compute; never claim the resource-resolved GPU."""
        return "cpu"


__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "CodexImageQAReward",
    "_extract_score_from_text",
    "_render_prompt_template",
]
