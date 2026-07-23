"""Lightweight public config schema for the JoyAI-Echo model family."""

from __future__ import annotations

from typing import Any

from vrl.config.model_schema import ModelSection


class EchoModelSection(ModelSection):
    """JoyAI-Echo checkpoint and Gemma text-encoder keys."""

    gemma_path: Any = None
    gemma_revision: Any = None


__all__ = ["EchoModelSection"]
