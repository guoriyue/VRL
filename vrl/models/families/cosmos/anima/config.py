"""Lightweight public config schema for Cosmos Predict2 Anima."""

from __future__ import annotations

from typing import Any

from vrl.config.model_schema import ModelSection


class CosmosAnimaModelSection(ModelSection):
    """Cosmos Anima single-file artifact paths and scheduler key."""

    qwen_tokenizer_path: Any = None
    qwen_tokenizer_revision: Any = None
    scheduler_shift: Any = None
    t5_tokenizer_path: Any = None
    t5_tokenizer_revision: Any = None
    text_encoder_file: Any = None
    text_encoder_path: Any = None
    transformer_file: Any = None
    transformer_path: Any = None
    vae_file: Any = None
    vae_path: Any = None


__all__ = ["CosmosAnimaModelSection"]
