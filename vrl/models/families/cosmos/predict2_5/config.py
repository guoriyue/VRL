"""Lightweight public config schema for Cosmos Predict2.5."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class CosmosPredict25ModelSection(ModelSection):
    """Cosmos Predict2.5 public model keys."""

    skip_text_encoder: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=False),
    )


__all__ = ["CosmosPredict25ModelSection"]
