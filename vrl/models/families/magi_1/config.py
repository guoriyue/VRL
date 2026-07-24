"""Lightweight public config schema for the MAGI-1 model family."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class Magi1ModelSection(ModelSection):
    """MAGI-1 isolated runtime and checkpoint component paths."""

    checkpoint_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    config_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    python_executable: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    source_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    source_revision: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    t5_pretrained_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    timeout_seconds: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    vae_pretrained_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )


__all__ = ["Magi1ModelSection"]
