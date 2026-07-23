"""Lightweight public config schema for Cosmos Predict2.5."""

from __future__ import annotations

from typing import Any

from vrl.config.model_schema import ModelSection


class CosmosPredict25ModelSection(ModelSection):
    """Cosmos Predict2.5 public model keys."""

    skip_text_encoder: Any = None


__all__ = ["CosmosPredict25ModelSection"]
