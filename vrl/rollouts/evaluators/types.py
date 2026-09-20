"""Training signal types for evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(kw_only=True)
class SegmentSignal:
    """Trajectory-native training signals for one logical segment.

    Every replay produces the three policy values. ``ref_log_prob`` is the one
    field decided at startup rather than by the replay: the trainer requests it
    exactly when the objective's KL term is on, and refuses that configuration
    without a reference model.
    """

    name: str
    distribution: str
    log_prob: Any
    old_log_prob: Any
    mask: Any
    ref_log_prob: Any | None = None


@dataclass(kw_only=True)
class FlowSDESignal(SegmentSignal):
    """Signals of a flow-matching reverse-SDE replay step.

    The SDE step always yields the proposal mean, its standard deviation, the
    step's ``sqrt(-dt)`` and flow-domain sigma, so objectives that read them
    (the KL in latent space, Flash-GRPO's rectification, the trust-region
    losses) take this type and read the fields. The two optional means follow
    startup decisions, not the replay: ``ref_prev_sample_mean`` accompanies
    ``ref_log_prob``; ``old_prev_sample_mean`` is the rollout-time proposal
    mean that generation stores only when the recipe sets
    ``rollout.return_prev_sample_mean`` (the factory requires it for
    trust-region objectives).
    """

    prev_sample_mean: Any
    std_dev_t: Any
    dt: Any
    sigma: Any
    ref_prev_sample_mean: Any | None = None
    old_prev_sample_mean: Any | None = None


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
                hint="Evaluators must pass step-level old_log_prob for per-step signals.",
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
    """What the trainer asks the evaluator to compute beyond the replay itself."""

    need_ref: bool = False


def _require_same_shape(left: Any, right: Any, *, label: str, hint: str = "") -> None:
    left_shape = getattr(left, "shape", None)
    right_shape = getattr(right, "shape", None)
    if left_shape is None or right_shape is None:
        return
    if tuple(left_shape) != tuple(right_shape):
        message = f"{label} shape mismatch: left={tuple(left_shape)} right={tuple(right_shape)}"
        if hint:
            message = f"{message}. {hint}"
        raise ValueError(message)
