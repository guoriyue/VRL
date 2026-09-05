"""Shared Pydantic contract for public configuration sections."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    """Typed public section: every key is a declared field.

    ``extra="forbid"`` is the whole unknown-key mechanism — a typo, a removed
    key, and a never-seen key all fail at ``parse_config`` with the same
    ``unknown <dotted.path>`` message (see ``_extract_error_message``).
    """

    model_config = ConfigDict(extra="forbid")


__all__ = ["ConfigBase"]
