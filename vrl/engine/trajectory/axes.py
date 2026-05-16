"""Axis metadata for generated training trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AxisKind = Literal[
    "sample",
    "discrete_token",
    "continuous_token",
    "denoise_step",
    "text_token",
    "segment",
    "frame",
    "media",
    "custom",
]


@dataclass(frozen=True, slots=True)
class TrajectoryAxis:
    """Named logical axis used by trajectory tensors."""

    name: str
    kind: AxisKind
    length: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TrajectoryAxis.name must be non-empty")
        if self.length is not None and self.length < 0:
            raise ValueError("TrajectoryAxis.length must be >= 0 when set")


__all__ = ["AxisKind", "TrajectoryAxis"]
