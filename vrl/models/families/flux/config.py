"""Lightweight public config schema for the FLUX model family."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class FluxModelSection(ModelSection):
    """FLUX public model keys."""

    # DiffusionNFT's frozen ``previous`` adapter switch.
    nft_previous_adapter: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=False),
    )


__all__ = ["FluxModelSection"]
