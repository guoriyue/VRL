"""Model-facing inputs and outputs for one scheduled token step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TokenLoopInit:
    """Model-provided state and row lanes for a token loop."""

    state: Any
    row_lanes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenStepBatch:
    """One scheduled token step built from a composition-owned envelope."""

    row_indices: list[int]
    position: int
    row_lanes: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.row_indices:
            raise ValueError("TokenStepBatch.row_indices must be non-empty")
        if any(index < 0 for index in self.row_indices):
            raise ValueError("TokenStepBatch.row_indices must be non-negative")
        if len(set(self.row_indices)) != len(self.row_indices):
            raise ValueError("TokenStepBatch.row_indices must be unique")
        if self.position < 0:
            raise ValueError("TokenStepBatch.position must be non-negative")


@dataclass(slots=True)
class TokenStepOutput:
    """Model updates produced by one scheduled token step."""

    updated_row_lanes: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["TokenLoopInit", "TokenStepBatch", "TokenStepOutput"]
