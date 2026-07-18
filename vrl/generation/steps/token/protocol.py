"""Model-facing inputs and outputs for one scheduled token step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TokenLoopInit:
    """Model-provided state and cache lanes for a token loop."""

    state: Any
    cache_lanes: Mapping[str, Any] = field(default_factory=dict)
    row_lanes: Mapping[str, Any] = field(default_factory=dict)
    cache_lane_owners: Mapping[str, str] = field(default_factory=dict)
    row_lane_owners: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TokenStepBatch:
    """One scheduled token step built from a composition-owned envelope."""

    row_indices: list[int]
    positions: list[int]
    position: int
    cache_lanes: dict[str, Any]
    row_lanes: dict[str, Any]


@dataclass(slots=True)
class TokenStepOutput:
    """Model updates produced by one scheduled token step."""

    updated_cache_lanes: Mapping[str, Any] = field(default_factory=dict)
    updated_row_lanes: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["TokenLoopInit", "TokenStepBatch", "TokenStepOutput"]
