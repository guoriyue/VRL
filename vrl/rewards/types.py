"""Reward scoring data containers.

The leaf of the reward import graph — the reward-side dual of the payload
dataclasses in vrl/generation/types.py. It lives apart from protocols.py and
runtime.py so the collector seam (vrl/rollouts) and the contract layer stay
importable without torch, CUDA utilities, or aiohttp. ``RewardSample.output``
stays ``Any`` deliberately: the media's shape (frames / latents) is owned by
whichever generation family produced it, and rewards must not constrain it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Internal schema key joining rollout prompt groups across the reward boundary.
# The collector is the only writer; batch-capable rewards consume the opaque
# value by equality and must never reconstruct it from sample ids or prompt text.
REWARD_GROUP_ID_METADATA_KEY = "reward_group_id"


@dataclass(slots=True)
class RewardSample:
    """One generated sample at the reward-domain boundary."""

    prompt: str
    output: Any  # Final generated media (frames / latents)
    sample_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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


__all__ = [
    "REWARD_GROUP_ID_METADATA_KEY",
    "RewardOutput",
    "RewardSample",
]
