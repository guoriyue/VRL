"""Lightweight public config schema for the FLUX model family."""

from __future__ import annotations

from typing import Any

from vrl.config.model_schema import ModelSection


class FluxModelSection(ModelSection):
    """FLUX public model keys."""

    # DiffusionNFT's frozen ``previous`` adapter switch.
    nft_previous_adapter: Any = None


__all__ = ["FluxModelSection"]
