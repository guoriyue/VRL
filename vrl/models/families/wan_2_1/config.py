"""Lightweight public config schema for the Wan model family."""

from __future__ import annotations

from typing import Any, Literal

from vrl.config.model_schema import ModelSection


class WanModelSection(ModelSection):
    """Wan-specific public model keys."""

    boundary_ratio: Any = None
    expert_lifecycle_profiling: bool = False
    offload_mode: Literal["none", "model", "sequential"] = "none"
    trainable_transformers: Any = None


__all__ = ["WanModelSection"]
