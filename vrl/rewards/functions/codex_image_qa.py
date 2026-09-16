"""Codex CLI subprocess judge scoring an image against its prompt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.models.codex_image_qa import (
    DEFAULT_PROMPT_TEMPLATE,
    _extract_score_from_text,
    _render_prompt_template,
)


class CodexImageQAReward(DiskArtifactRewardFunction):
    """Codex CLI subprocess judge scoring an image against its prompt.

    In-process the model is built here and media rides the request in memory;
    ``inference.kind=service`` hands the same kwargs to a driver-launched
    service that scores this reward's ``.pt`` artifacts.
    """

    model_factory = "vrl.rewards.models.codex_image_qa:CodexImageQARewardModel"
    request_prefix = "codex_image_qa"
    debug_basename = "codex_image_qa"
    default_reward_name = "codex_image_qa"
    default_score_key = "codex_image_qa"
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"
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
