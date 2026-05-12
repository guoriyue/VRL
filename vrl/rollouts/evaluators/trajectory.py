"""Adapters between legacy SignalBatch and trajectory-native signals."""

from __future__ import annotations

from typing import Any

import torch

from vrl.engine.trajectory import TrainingView, TrajectoryBatch
from vrl.engine.trajectory.views import LossUnit
from vrl.rollouts.evaluators.types import (
    SegmentSignal,
    SignalBatch,
    TrajectorySignalBatch,
)


def to_trajectory_signals(
    signals: SignalBatch | TrajectorySignalBatch,
    *,
    trajectory: TrajectoryBatch | None = None,
    training_view: TrainingView | None = None,
    old_log_probs: Any | None = None,
    group_ids: Any | None = None,
    context: dict[str, Any] | None = None,
    primary_segment: str | None = None,
    mask_key: str = "token_mask",
) -> TrajectorySignalBatch:
    """Project legacy evaluator signals into the trajectory signal schema."""

    if isinstance(signals, TrajectorySignalBatch):
        return signals

    if isinstance((signals.aux or {}).get("segments"), dict):
        return _multi_segment_to_trajectory(
            signals,
            trajectory=trajectory,
            training_view=training_view,
            group_ids=group_ids,
            context=context,
            primary_segment=primary_segment,
            mask_key=mask_key,
        )

    segment_name = _primary_segment_name(
        trajectory=trajectory,
        training_view=training_view,
        fallback=primary_segment,
    )
    segment = trajectory.segments.get(segment_name) if trajectory is not None else None
    unit = _loss_unit(training_view, segment_name)
    old_log_prob = old_log_probs
    if old_log_prob is None and segment is not None:
        old_log_prob = _role_value(segment, "old_log_prob")
    if old_log_prob is None:
        raise RuntimeError("old_log_probs is required for legacy SignalBatch conversion")
    mask = _mask_from_signal_or_trajectory(signals, segment, mask_key=mask_key)
    _validate_signal_shapes(
        log_prob=signals.log_prob,
        old_log_prob=old_log_prob,
        mask=mask,
        segment=segment_name,
    )
    axis = unit.axis if unit is not None else _axis_from_segment_or_signal(segment, old_log_prob)
    axes = _axes_from_value(old_log_prob, axis=axis)

    return TrajectorySignalBatch(
        segments={
            segment_name: SegmentSignal(
                name=segment_name,
                segment=segment_name,
                axis=axis,
                axes=axes,
                distribution=segment.distribution if segment is not None else signals.dist_family,
                log_prob=signals.log_prob,
                old_log_prob=old_log_prob,
                mask=mask,
                ref_log_prob=signals.ref_log_prob,
                entropy=signals.entropy,
                prev_sample_mean=signals.prev_sample_mean,
                ref_prev_sample_mean=signals.ref_prev_sample_mean,
                std_dev_t=signals.std_dev_t,
                dt=signals.dt,
                aux=dict(signals.aux),
            )
        },
        group_ids=group_ids if group_ids is not None else _group_ids_from_trajectory(trajectory),
        context=context if context is not None else _context_from_trajectory(trajectory),
        primary_segment=segment_name,
    )


def trajectory_signals_to_signal_batch(
    signals: TrajectorySignalBatch,
    *,
    mask_key: str = "token_mask",
) -> SignalBatch:
    """Project trajectory-native signals back to the legacy algorithm API."""

    if len(signals.segments) > 1:
        segment_signals: dict[str, SignalBatch] = {}
        old_by_segment: dict[str, Any] = {}
        for name, segment in signals.segments.items():
            segment_signals[name] = _segment_to_signal_batch(segment, mask_key=mask_key)
            old_by_segment[name] = segment.old_log_prob
        primary = signals.primary
        aux: dict[str, Any] = {
            "segments": segment_signals,
            "old_log_probs": old_by_segment,
            "segment_order": tuple(signals.segments),
            "primary_segment": primary.name,
        }
        aux.update(primary.aux)
        if primary.mask is not None:
            aux[mask_key] = primary.mask
        return SignalBatch(
            log_prob=primary.log_prob,
            ref_log_prob=primary.ref_log_prob,
            entropy=primary.entropy,
            dist_family=f"multisegment_{primary.distribution}",
            aux=aux,
        )
    return _segment_to_signal_batch(signals.primary, mask_key=mask_key)


def old_log_probs_from_trajectory_signals(signals: TrajectorySignalBatch) -> Any:
    """Return the legacy old_log_probs argument for an algorithm call."""

    if len(signals.segments) == 1:
        return signals.primary.old_log_prob
    return {name: segment.old_log_prob for name, segment in signals.segments.items()}


def _multi_segment_to_trajectory(
    signals: SignalBatch,
    *,
    trajectory: TrajectoryBatch | None,
    training_view: TrainingView | None,
    group_ids: Any | None,
    context: dict[str, Any] | None,
    primary_segment: str | None,
    mask_key: str,
) -> TrajectorySignalBatch:
    raw_segments = (signals.aux or {})["segments"]
    raw_old = (signals.aux or {}).get("old_log_probs") or {}
    out: dict[str, SegmentSignal] = {}

    for name, segment_signal in raw_segments.items():
        if not isinstance(segment_signal, SignalBatch):
            raise TypeError(f"segment signal {name!r} must be a SignalBatch")
        segment = trajectory.segments.get(name) if trajectory is not None else None
        unit = _loss_unit(training_view, name)
        old_log_prob = raw_old.get(name)
        if old_log_prob is None and segment is not None:
            old_log_prob = _role_value(segment, "old_log_prob")
        if old_log_prob is None:
            raise RuntimeError(f"old log-prob is required for segment {name!r}")
        mask = _mask_from_signal_or_trajectory(segment_signal, segment, mask_key=mask_key)
        _validate_signal_shapes(
            log_prob=segment_signal.log_prob,
            old_log_prob=old_log_prob,
            mask=mask,
            segment=name,
        )
        axis = unit.axis if unit is not None else _axis_from_segment_or_signal(segment, old_log_prob)
        out[name] = SegmentSignal(
            name=name,
            segment=name,
            axis=axis,
            axes=_axes_from_value(old_log_prob, axis=axis),
            distribution=segment.distribution if segment is not None else segment_signal.dist_family,
            log_prob=segment_signal.log_prob,
            old_log_prob=old_log_prob,
            mask=mask,
            ref_log_prob=segment_signal.ref_log_prob,
            entropy=segment_signal.entropy,
            prev_sample_mean=segment_signal.prev_sample_mean,
            ref_prev_sample_mean=segment_signal.ref_prev_sample_mean,
            std_dev_t=segment_signal.std_dev_t,
            dt=segment_signal.dt,
            aux=dict(segment_signal.aux),
        )

    primary = (
        primary_segment
        or (signals.aux or {}).get("primary_segment")
        or (training_view.primary_segment if training_view is not None else None)
        or next(iter(out))
    )
    return TrajectorySignalBatch(
        segments=out,
        group_ids=group_ids if group_ids is not None else _group_ids_from_trajectory(trajectory),
        context=context if context is not None else _context_from_trajectory(trajectory),
        primary_segment=primary,
    )


def _segment_to_signal_batch(segment: SegmentSignal, *, mask_key: str) -> SignalBatch:
    aux = dict(segment.aux)
    if segment.mask is not None:
        aux.setdefault(mask_key, segment.mask)
    return SignalBatch(
        log_prob=segment.log_prob,
        ref_log_prob=segment.ref_log_prob,
        entropy=segment.entropy,
        prev_sample_mean=segment.prev_sample_mean,
        ref_prev_sample_mean=segment.ref_prev_sample_mean,
        std_dev_t=segment.std_dev_t,
        dt=segment.dt,
        dist_family=segment.distribution,
        aux=aux,
    )


def _primary_segment_name(
    *,
    trajectory: TrajectoryBatch | None,
    training_view: TrainingView | None,
    fallback: str | None,
) -> str:
    if fallback:
        return fallback
    if training_view is not None and training_view.primary_segment is not None:
        return training_view.primary_segment
    if trajectory is not None:
        for name, segment in trajectory.segments.items():
            if segment.trainable:
                return name
    return "default"


def _loss_unit(training_view: TrainingView | None, segment_name: str) -> LossUnit | None:
    if training_view is None:
        return None
    for unit in training_view.loss_units:
        if unit.segment == segment_name:
            return unit
    return None


def _role_value(segment: Any, role: str) -> Any:
    matches = [tensor for tensor in segment.tensors.values() if tensor.role == role]
    if len(matches) != 1:
        raise RuntimeError(
            f"segment {segment.name!r} requires exactly one role {role!r}, "
            f"found {len(matches)}",
        )
    return matches[0].value


def _mask_from_signal_or_trajectory(
    signals: SignalBatch,
    segment: Any | None,
    *,
    mask_key: str,
) -> Any:
    if signals.aux:
        if mask_key in signals.aux:
            return signals.aux[mask_key]
        if "mask" in signals.aux:
            return signals.aux["mask"]
    if segment is not None:
        mask = _role_value(segment, "mask")
        if _same_shape(mask, signals.log_prob):
            return mask
    return torch.ones_like(signals.log_prob)


def _validate_signal_shapes(
    *,
    log_prob: Any,
    old_log_prob: Any,
    mask: Any,
    segment: str,
) -> None:
    if not _same_shape(log_prob, old_log_prob):
        raise RuntimeError(
            "trajectory signal shape mismatch for segment "
            f"{segment!r}: log_prob shape={_shape(log_prob)}, "
            f"old_log_prob shape={_shape(old_log_prob)}. Pass step-level "
            "old_log_probs when adapting per-step legacy evaluator signals.",
        )
    if mask is not None and not _same_shape(log_prob, mask):
        raise RuntimeError(
            "trajectory signal mask shape mismatch for segment "
            f"{segment!r}: log_prob shape={_shape(log_prob)}, mask shape={_shape(mask)}",
        )


def _same_shape(left: Any, right: Any) -> bool:
    left_shape = getattr(left, "shape", None)
    right_shape = getattr(right, "shape", None)
    if left_shape is None or right_shape is None:
        return True
    return tuple(left_shape) == tuple(right_shape)


def _shape(value: Any) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "<unknown>"
    return tuple(int(dim) for dim in shape)


def _axis_from_segment_or_signal(segment: Any | None, value: Any) -> str:
    if segment is not None:
        for tensor in segment.tensors.values():
            if tensor.role == "old_log_prob":
                for axis in tensor.axes:
                    if axis != "sample":
                        return axis
    return "step" if getattr(value, "ndim", 1) <= 1 else "token"


def _axes_from_value(value: Any, *, axis: str) -> tuple[str, ...]:
    ndim = int(getattr(value, "ndim", 1))
    if ndim <= 1:
        return ("sample",)
    return ("sample", axis)


def _group_ids_from_trajectory(trajectory: TrajectoryBatch | None) -> Any:
    if trajectory is None:
        return None
    return trajectory.group_ids


def _context_from_trajectory(trajectory: TrajectoryBatch | None) -> dict[str, Any]:
    if trajectory is None:
        return {}
    return dict(trajectory.context)


__all__ = [
    "old_log_probs_from_trajectory_signals",
    "to_trajectory_signals",
    "trajectory_signals_to_signal_batch",
]
