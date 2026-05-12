"""Training signal types for evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalBatch:
    """Distribution-family-specific training signals.

    Produced by an ``Evaluator`` from model forward results.
    Consumed by ``Algorithm.compute_loss()``.
    """

    log_prob: Any                               # [B, T] log pi(a|s)
    ref_log_prob: Any | None = None             # [B, T] log pi_ref(a|s)
    entropy: Any | None = None
    # Flow-matching specific (for latent-space KL)
    prev_sample_mean: Any | None = None
    ref_prev_sample_mean: Any | None = None
    std_dev_t: Any | None = None
    dt: Any | None = None
    dist_family: str = "flow_matching"          # or "categorical", etc.
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentSignal:
    """Trajectory-native training signals for one logical segment."""

    name: str
    segment: str
    axis: str
    axes: tuple[str, ...]
    distribution: str
    log_prob: Any
    old_log_prob: Any
    mask: Any
    ref_log_prob: Any | None = None
    entropy: Any | None = None
    prev_sample_mean: Any | None = None
    ref_prev_sample_mean: Any | None = None
    std_dev_t: Any | None = None
    dt: Any | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    @property
    def dist_family(self) -> str:
        """Legacy algorithm name for the segment distribution."""

        return self.distribution


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
