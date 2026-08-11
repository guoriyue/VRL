"""Reward scoring data containers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RewardSample:
    """One generated sample at the reward-domain boundary."""

    prompt: str
    output: Any  # Final generated media (frames / latents)
    sample_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str):
            raise TypeError("RewardSample.sample_id must be a str")
        if not self.sample_id:
            raise ValueError("RewardSample.sample_id must be non-empty")


@dataclass(frozen=True, slots=True)
class RewardOutput:
    """Sample-aligned scores and observations from one reward call."""

    scores: tuple[float, ...]
    components: dict[str, tuple[float, ...]] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scores = tuple(float(score) for score in self.scores)
        components = {
            str(name): tuple(float(value) for value in values)
            for name, values in self.components.items()
        }
        timing_ms = {str(name): float(value) for name, value in self.timing_ms.items()}
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("RewardOutput.scores must contain only finite values")
        for name, values in components.items():
            if len(values) != len(scores):
                raise ValueError(
                    "RewardOutput component/score mismatch: "
                    f"component={name!r}, values={len(values)}, scores={len(scores)}",
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"RewardOutput component {name!r} must contain only finite values",
                )
        for name, value in timing_ms.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"RewardOutput timing {name!r} must be finite and non-negative",
                )
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "timing_ms", timing_ms)


__all__ = ["RewardOutput", "RewardSample"]
