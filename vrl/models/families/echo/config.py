"""Lightweight public config schema for the JoyAI-Echo model family."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata

if TYPE_CHECKING:
    from vrl.models.interfaces.runtime import ModelBuild

ECHO_VIDEO_HEIGHT = 512
ECHO_VIDEO_WIDTH = 768


class EchoModelSection(ModelSection):
    """JoyAI-Echo checkpoint and Gemma text-encoder keys."""

    gemma_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="gemma",
            revision_field="gemma_revision",
        ),
    )
    gemma_revision: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source_revision",
            source="gemma",
        ),
    )
    video_height: int = Field(
        default=ECHO_VIDEO_HEIGHT,
        gt=0,
        multiple_of=32,
        json_schema_extra=checkpoint_identity_metadata(
            "value",
            default=ECHO_VIDEO_HEIGHT,
        ),
    )
    video_width: int = Field(
        default=ECHO_VIDEO_WIDTH,
        gt=0,
        multiple_of=32,
        json_schema_extra=checkpoint_identity_metadata(
            "value",
            default=ECHO_VIDEO_WIDTH,
        ),
    )


def resolve_echo_video_dimensions(build: ModelBuild) -> tuple[int, int]:
    """Resolve the fixed latent-grid dimensions shared by rollout and replay."""

    model_config = getattr(build, "model_config", None) or {}
    sampling_config = getattr(build, "sampling_config", None) or {}
    video_height = int(model_config.get("video_height", ECHO_VIDEO_HEIGHT))
    video_width = int(model_config.get("video_width", ECHO_VIDEO_WIDTH))

    for model_name, sampling_name, construction_value in (
        ("video_height", "height", video_height),
        ("video_width", "width", video_width),
    ):
        if construction_value < 1 or construction_value % 32:
            raise ValueError(
                f"model.{model_name} must be a positive multiple of 32, got {construction_value}",
            )
        requested = sampling_config.get(sampling_name)
        if requested is not None and int(requested) != construction_value:
            raise ValueError(
                f"sampling.{sampling_name}={int(requested)} must equal "
                f"model.{model_name}={construction_value} for Echo's fixed latent grid",
            )
    return video_height, video_width


__all__ = [
    "ECHO_VIDEO_HEIGHT",
    "ECHO_VIDEO_WIDTH",
    "EchoModelSection",
    "resolve_echo_video_dimensions",
]
