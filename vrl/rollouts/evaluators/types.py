"""Training signal types for evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SegmentSignal:
    """Trajectory-native training signals for one logical segment."""

    name: str
    distribution: str
    log_prob: Any
    old_log_prob: Any
    mask: Any
    ref_log_prob: Any | None = None
    entropy: Any | None = None
    prev_sample_mean: Any | None = None
    ref_prev_sample_mean: Any | None = None
    # Rollout-time (behavior policy) reverse-SDE proposal mean, captured at
    # generation and replayed back unchanged. Distinct from prev_sample_mean
    # (current replay policy) and ref_prev_sample_mean (frozen ref): trust-region
    # algorithms (Flow-DPPO / GRPO-Guard) measure the current-vs-rollout drift,
    # so they need this third mean. None unless generation stored it.
    old_prev_sample_mean: Any | None = None
    std_dev_t: Any | None = None
    dt: Any | None = None
    aux: dict[str, Any] = field(default_factory=dict)

@dataclass
class TrajectorySignalBatch:
    """First-class signal schema derived from a TrajectoryBatch."""

    segments: dict[str, SegmentSignal]
    group_ids: Any
    context: dict[str, Any] = field(default_factory=dict)
    primary_segment: str | None = None

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("TrajectorySignalBatch.segments must be non-empty")
        if self.primary_segment is not None and self.primary_segment not in self.segments:
            raise ValueError(
                f"TrajectorySignalBatch.primary_segment={self.primary_segment!r} "
                "is not present in segments",
            )
        for name, segment in self.segments.items():
            if not isinstance(segment, SegmentSignal):
                raise TypeError(f"trajectory signal {name!r} must be a SegmentSignal")
            if not segment.name:
                raise ValueError(f"trajectory signal {name!r} must have a non-empty name")
            if segment.name != name:
                raise ValueError(
                    f"trajectory signal key {name!r} must match segment name {segment.name!r}",
                )
            if not segment.distribution:
                raise ValueError(
                    f"trajectory signal {name!r} must have a non-empty distribution",
                )
            _require_same_shape(
                segment.log_prob,
                segment.old_log_prob,
                label=f"trajectory signal {name!r} log_prob/old_log_prob",
            )
            if segment.mask is None:
                raise ValueError(f"trajectory signal {name!r} must have a mask")
            _require_same_shape(
                segment.log_prob,
                segment.mask,
                label=f"trajectory signal {name!r} log_prob/mask",
            )

    @property
    def primary(self) -> SegmentSignal:
        """Return the primary segment signal."""

        if self.primary_segment is not None:
            return self.segments[self.primary_segment]
        return next(iter(self.segments.values()))


@dataclass
class SignalRequest:
    """What the algorithm needs the evaluator to compute."""

    need_ref: bool = False
    need_entropy: bool = False
    need_kl_intermediates: bool = False  # for latent-space KL


def _require_same_shape(left: Any, right: Any, *, label: str) -> None:
    left_shape = getattr(left, "shape", None)
    right_shape = getattr(right, "shape", None)
    if left_shape is None or right_shape is None:
        return
    if tuple(left_shape) != tuple(right_shape):
        raise ValueError(
            f"{label} shape mismatch: left={tuple(left_shape)} right={tuple(right_shape)}",
        )
