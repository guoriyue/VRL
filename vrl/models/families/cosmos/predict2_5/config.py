"""Lightweight public config schema for Cosmos Predict2.5."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vrl.models.checkpoint_identity import checkpoint_identity_metadata
from vrl.models.families.cosmos.config import CosmosVideoModelSection


class CosmosPredict25ModelSection(CosmosVideoModelSection):
    """Cosmos Predict2.5 public model keys."""

    skip_text_encoder: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=False),
    )


__all__ = ["CosmosPredict25ModelSection"]
