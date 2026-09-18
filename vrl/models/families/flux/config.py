"""Lightweight public config schema for the FLUX model family."""

from __future__ import annotations

from typing import ClassVar

from vrl.config.model_schema import ModelSection


class FluxModelSection(ModelSection):
    """FLUX public model keys."""

    supports_previous_adapter: ClassVar[bool] = True


__all__ = ["FluxModelSection"]
