"""Lightweight public config schema for Cosmos Predict2 Anima."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class CosmosAnimaModelSection(ModelSection):
    """Cosmos Anima single-file artifact paths and scheduler key."""

    path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="main",
            revision_field="revision",
            members_only=True,
        ),
    )
    qwen_tokenizer_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="qwen_tokenizer",
            revision_field="qwen_tokenizer_revision",
        ),
    )
    qwen_tokenizer_revision: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source_revision",
            source="qwen_tokenizer",
        ),
    )
    scheduler_shift: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=3.0),
    )
    t5_tokenizer_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="t5_tokenizer",
            revision_field="t5_tokenizer_revision",
        ),
    )
    t5_tokenizer_revision: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source_revision",
            source="t5_tokenizer",
        ),
    )
    text_encoder_file: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "member",
            source="main",
            overridden_by="text_encoder_path",
        ),
    )
    text_encoder_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="text_encoder",
        ),
    )
    transformer_file: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "member",
            source="main",
            overridden_by="transformer_path",
        ),
    )
    transformer_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="transformer",
        ),
    )
    vae_file: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "member",
            source="main",
            overridden_by="vae_path",
        ),
    )
    vae_path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="vae",
        ),
    )


__all__ = ["CosmosAnimaModelSection"]
