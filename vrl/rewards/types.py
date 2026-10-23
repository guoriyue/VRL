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


@dataclass(slots=True)
class RewardSample:
    """One generated sample at the reward-domain boundary.

    ``output`` is decoded media or a boxed object-store sample reference.
    Transport selection and scorer-local files do not change this contract.
    """

    prompt: str
    output: Any
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

    def __post_init__(self) -> None:
        scores = tuple(float(score) for score in self.scores)
        components = {
            str(name): tuple(float(value) for value in values)
            for name, values in self.components.items()
        }
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
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "components", components)


__all__ = [
    "RewardOutput",
    "RewardSample",
]
