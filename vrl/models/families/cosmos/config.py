"""Lightweight model schema for frame-conditioned Cosmos video models."""

from __future__ import annotations

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class CosmosVideoModelSection(ModelSection):
    """Predict2 and Predict2.5 use per-frame timesteps; Anima does not."""

    # Both rollout and replay share the frame-level conditioning computation.
    # This changes kernels, not weights, so it is excluded from checkpoint identity.
    frame_shared_adaln: bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
