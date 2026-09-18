"""Lightweight public config schema for the FLUX model family."""

from __future__ import annotations

from vrl.config.model_schema import ModelSection


class FluxModelSection(ModelSection):
    """FLUX public model keys."""


__all__ = ["FluxModelSection"]
