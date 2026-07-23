"""Shared Pydantic contract for public configuration sections."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    """Typed public section whose unknown keys are reported by the tree walker."""

    model_config = ConfigDict(extra="ignore")


__all__ = ["ConfigBase"]
