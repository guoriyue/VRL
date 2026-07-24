"""Lightweight public config schema for the CausVid model family."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata

# Immutable upstream protocol values shared by schema defaults and runtime
# integrity checks. Keeping them in this lightweight module lets config
# identity resolve defaults without importing the model implementation.
CAUSVID_SOURCE_REVISION = "adb6a5ecd07666b4d0290042915c8406e6d5ce22"
CAUSVID_CHECKPOINT_FILE = "autoregressive_checkpoint/model.pt"


class CausVidModelSection(ModelSection):
    """CausVid pinned source, Wan base, and released checkpoint keys."""

    accept_noncommercial_license: bool = Field(
        default=False,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    base_model_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="base_model",
            revision_field="base_model_revision",
        ),
    )
    base_model_revision: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source_revision",
            source="base_model",
        ),
    )
    causvid_source_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    causvid_source_revision: Any = Field(
        default=CAUSVID_SOURCE_REVISION,
        json_schema_extra=checkpoint_identity_metadata("value"),
    )
    checkpoint_file: Any = Field(
        default=CAUSVID_CHECKPOINT_FILE,
        json_schema_extra=checkpoint_identity_metadata(
            "member",
            source="main",
            omit_for_source_file=True,
        ),
    )
    checkpoint_sha256: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )


__all__ = [
    "CAUSVID_CHECKPOINT_FILE",
    "CAUSVID_SOURCE_REVISION",
    "CausVidModelSection",
]
