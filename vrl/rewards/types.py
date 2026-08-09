"""Reward scoring data containers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RewardSample:
    """One generated sample with its canonical generation lineage."""

    prompt: str
    output: Any  # Final generated media (frames / latents)
    source_request_id: str
    sample_id: str
    group_id: str
    trajectory_id: str
    policy_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_request_id",
            "sample_id",
            "group_id",
            "trajectory_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"RewardSample.{name} must be a str")
            if not value:
                raise ValueError(f"RewardSample.{name} must be non-empty")
        if self.policy_version is not None and int(self.policy_version) < 0:
            raise ValueError("RewardSample.policy_version must be >= 0")


@dataclass(frozen=True, slots=True)
class RewardRequest:
    """One logical scoring operation over ordered generated samples."""

    request_id: str
    samples: tuple[RewardSample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError("RewardRequest.request_id must be a str")
        if not self.request_id:
            raise ValueError("RewardRequest.request_id must be non-empty")
        normalized = tuple(self.samples)
        if not normalized:
            raise ValueError("RewardRequest.samples must be non-empty")
        if not all(isinstance(sample, RewardSample) for sample in normalized):
            raise TypeError("RewardRequest.samples must contain RewardSample values")
        sample_ids = [sample.sample_id for sample in normalized]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("RewardRequest.sample_id values must be unique")
        object.__setattr__(self, "samples", normalized)


@dataclass(frozen=True, slots=True)
class RewardOutput:
    """Sample-aligned result of one complete reward request."""

    request_id: str
    sample_ids: tuple[str, ...]
    scores: tuple[float, ...]
    components: dict[str, tuple[float, ...]] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError("RewardOutput.request_id must be a str")
        if not self.request_id:
            raise ValueError("RewardOutput.request_id must be non-empty")
        sample_ids = tuple(self.sample_ids)
        scores = tuple(float(score) for score in self.scores)
        components = {
            str(name): tuple(float(value) for value in values)
            for name, values in self.components.items()
        }
        timing_ms = {str(name): float(value) for name, value in self.timing_ms.items()}
        if not all(isinstance(sample_id, str) for sample_id in sample_ids):
            raise TypeError("RewardOutput.sample_ids must contain str values")
        if not sample_ids or any(not sample_id for sample_id in sample_ids):
            raise ValueError("RewardOutput.sample_ids must be non-empty")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("RewardOutput.sample_ids must be unique")
        if len(scores) != len(sample_ids):
            raise ValueError(
                "RewardOutput score/sample mismatch: "
                f"scores={len(scores)}, samples={len(sample_ids)}",
            )
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("RewardOutput.scores must contain only finite values")
        for name, values in components.items():
            if len(values) != len(sample_ids):
                raise ValueError(
                    "RewardOutput component/sample mismatch: "
                    f"component={name!r}, scores={len(values)}, "
                    f"samples={len(sample_ids)}",
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
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "timing_ms", timing_ms)

    def validate(self, request: RewardRequest) -> None:
        """Validate request identity and exact sample ordering at the runtime seam."""

        if self.request_id != request.request_id:
            raise ValueError(
                "reward output belongs to a different request: "
                f"expected={request.request_id!r}, actual={self.request_id!r}",
            )
        expected_sample_ids = tuple(sample.sample_id for sample in request.samples)
        if self.sample_ids != expected_sample_ids:
            raise ValueError(
                "reward output sample order mismatch: "
                f"expected={expected_sample_ids!r}, actual={self.sample_ids!r}",
            )


__all__ = ["RewardOutput", "RewardRequest", "RewardSample"]
