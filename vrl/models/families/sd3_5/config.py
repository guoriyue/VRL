"""Lightweight public config schema for the SD 3.5 model family."""

from __future__ import annotations

from typing import ClassVar

from vrl.config.model_schema import ModelSection


class SD3_5ModelSection(ModelSection):
    """SD 3.5 public model keys."""

    supports_previous_adapter: ClassVar[bool] = True


__all__ = ["SD3_5ModelSection"]
