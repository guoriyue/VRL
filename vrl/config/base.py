"""Shared Pydantic contract for public configuration sections."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    """Typed public section whose unknown keys are reported by the tree walker."""

    model_config = ConfigDict(extra="ignore")


class ClosedConfigBase(ConfigBase):
    """Fully typed section: every key is a declared field, so pydantic itself
    rejects unknown ones (the tree walker still names them first)."""

    model_config = ConfigDict(extra="forbid")


__all__ = ["ClosedConfigBase", "ConfigBase"]
