"""Operations over trajectory batches.

These helpers keep sample-axis slicing, stacking, and device movement in one
place so batching, rollout packing, and trainer utilities do not each invent a
slightly different trajectory convention.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vrl.engine.core.types import GenerationRequest, GenerationSampleSpec
from vrl.engine.trajectory.types import (
    TrajectoryBatch,
    TrajectoryMetrics,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.engine.trajectory.validation import validate_trajectory_batch


def slice_trajectory_batch(
    data: Any,
    *,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    offset: int,
    count: int,
    total: int,
) -> Any:
    """Slice a TrajectoryBatch along the sample axis, preserving bridge safety."""

    if data is None:
        return None
    if not isinstance(data, TrajectoryBatch):
        return data

    return _rebuild_trajectory(
        data,
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_specs=list(sample_specs),
        group_ids=_slice_value(data.group_ids, offset, count, total),
        tensor_value_fn=lambda tensor: _slice_value(tensor.value, offset, count, total)
        if tensor.axes and tensor.axes[0] == "sample"
        else tensor.value,
        axes_sample_length=count,
        metrics_sample_count=count,
        metrics_values=_slice_value(data.metrics.values, offset, count, total),
        context=_slice_value(data.context, offset, count, total),
    )


def select_trajectory_batch(data: Any, selector: Any) -> Any:
    """Select TrajectoryBatch rows by boolean mask or integer indices."""

    if data is None:
        return None
    if not isinstance(data, TrajectoryBatch):
        return data

    positions = _selector_positions(selector)
    count = len(positions)
    return _rebuild_trajectory(
        data,
        request_id=data.request_id,
        family=data.family,
        task=data.task,
        sample_specs=[data.sample_specs[i] for i in positions],
        group_ids=_select_value(data.group_ids, selector, len(data.sample_specs)),
        tensor_value_fn=lambda tensor: _select_value(tensor.value, selector, len(data.sample_specs))
        if tensor.axes and tensor.axes[0] == "sample"
        else tensor.value,
        axes_sample_length=count,
        metrics_sample_count=count,
        metrics_values=_select_value(data.metrics.values, selector, len(data.sample_specs)),
        context=_select_value(data.context, selector, len(data.sample_specs)),
    )


def stack_trajectory_batches(batches: list[TrajectoryBatch | None]) -> TrajectoryBatch | None:
    """Concatenate compatible trajectory batches along the sample axis."""

    if not batches:
        raise ValueError("batches must be non-empty")
    if all(batch is None for batch in batches):
        return None
    if any(batch is None for batch in batches):
        raise ValueError("cannot stack mixed trajectory and non-trajectory batches")

    typed = [batch for batch in batches if batch is not None]
    first = typed[0]
    _validate_stack_compatible(typed)

    sample_specs: list[GenerationSampleSpec] = []
    for batch in typed:
        sample_specs.extend(batch.sample_specs)
    sample_count = len(sample_specs)

    return _rebuild_trajectory(
        first,
        request_id=first.request_id,
        family=first.family,
        task=first.task,
        sample_specs=sample_specs,
        group_ids=_stack_values([batch.group_ids for batch in typed]),
        tensor_value_fn=lambda tensor: _stack_values(
            [
                batch.segments[_segment_name_for_tensor(first, tensor)]
                .tensors[tensor.name]
                .value
                for batch in typed
            ]
        )
        if tensor.axes and tensor.axes[0] == "sample"
        else tensor.value,
        axes_sample_length=sample_count,
        metrics_sample_count=sample_count,
        metrics_values=dict(first.metrics.values),
        context=dict(first.context),
    )


def move_trajectory_batch(data: Any, device: Any) -> Any:
    """Move tensor leaves in a TrajectoryBatch to a target device."""

    if data is None:
        return None
    if not isinstance(data, TrajectoryBatch):
        return data

    return _rebuild_trajectory(
        data,
        request_id=data.request_id,
        family=data.family,
        task=data.task,
        sample_specs=list(data.sample_specs),
        group_ids=_move_value(data.group_ids, device),
        tensor_value_fn=lambda tensor: _move_value(tensor.value, device),
        axes_sample_length=data.axes["sample"].length,
        metrics_sample_count=data.metrics.num_samples,
        metrics_values=_move_value(data.metrics.values, device),
        context=_move_value(data.context, device),
    )


def _rebuild_trajectory(
    data: TrajectoryBatch,
    *,
    request_id: str,
    family: str,
    task: str,
    sample_specs: list[GenerationSampleSpec],
    group_ids: Any,
    tensor_value_fn: Any,
    axes_sample_length: int | None,
    metrics_sample_count: int | None,
    metrics_values: dict[str, Any],
    context: dict[str, Any],
) -> TrajectoryBatch:
    axes = {
        name: replace(axis, length=axes_sample_length) if name == "sample" else axis
        for name, axis in data.axes.items()
    }
    segments = {
        name: TrajectorySegment(
            name=segment.name,
            modality=segment.modality,
            trainable=segment.trainable,
            distribution=segment.distribution,
            tensors={
                tensor_name: TrajectoryTensor(
                    name=tensor.name,
                    value=tensor_value_fn(tensor),
                    axes=tensor.axes,
                    role=tensor.role,
                    metadata=dict(tensor.metadata),
                )
                for tensor_name, tensor in segment.tensors.items()
            },
            reward_view=segment.reward_view,
            advantage_scope=segment.advantage_scope,
            replay_inputs=dict(segment.replay_inputs),
            metadata=dict(segment.metadata),
        )
        for name, segment in data.segments.items()
    }
    axis_lengths = dict(data.metrics.axis_lengths)
    if axes_sample_length is not None and "sample" in axis_lengths:
        axis_lengths["sample"] = axes_sample_length

    out = TrajectoryBatch(
        request_id=request_id,
        family=family,
        task=task,
        sample_specs=sample_specs,
        group_ids=group_ids,
        axes=axes,
        segments=segments,
        reward_views=dict(data.reward_views),
        metrics=TrajectoryMetrics(
            num_samples=metrics_sample_count,
            axis_lengths=axis_lengths,
            values=metrics_values,
        ),
        context=context,
    )
    return validate_trajectory_batch(out)


def _slice_value(value: Any, offset: int, count: int, total: int) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == total:
        return value[offset : offset + count]
    if isinstance(value, list) and len(value) == total:
        return value[offset : offset + count]
    if isinstance(value, tuple) and len(value) == total:
        return value[offset : offset + count]
    if isinstance(value, dict):
        return {key: _slice_value(inner, offset, count, total) for key, inner in value.items()}
    return value


def _select_value(value: Any, selector: Any, batch_size: int) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == batch_size:
        return value[selector.to(value.device)] if hasattr(selector, "to") else value[selector]
    if isinstance(value, list) and len(value) == batch_size:
        return [value[i] for i in _selector_positions(selector)]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[i] for i in _selector_positions(selector))
    if isinstance(value, dict):
        return {key: _select_value(inner, selector, batch_size) for key, inner in value.items()}
    return value


def _stack_values(values: list[Any]) -> Any:
    if not values:
        raise ValueError("values must be non-empty")
    first = values[0]
    shape = getattr(first, "shape", None)
    if shape is not None:
        import torch

        return torch.cat(values, dim=0)
    if isinstance(first, list):
        out: list[Any] = []
        for value in values:
            out.extend(value)
        return out
    if isinstance(first, tuple):
        out: list[Any] = []
        for value in values:
            out.extend(list(value))
        return tuple(out)
    if isinstance(first, dict):
        keys = set(first)
        for value in values[1:]:
            if not isinstance(value, dict) or set(value) != keys:
                raise ValueError("trajectory dict leaves must have matching keys")
        return {key: _stack_values([value[key] for value in values]) for key in first}
    return first


def _move_value(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        try:
            return value.to(device)
        except TypeError:
            return value
    if isinstance(value, dict):
        return {key: _move_value(inner, device) for key, inner in value.items()}
    if isinstance(value, list):
        return [_move_value(inner, device) for inner in value]
    if isinstance(value, tuple):
        return tuple(_move_value(inner, device) for inner in value)
    return value


def _selector_positions(selector: Any) -> list[int]:
    if hasattr(selector, "detach"):
        selector_cpu = selector.detach().cpu()
        if str(selector_cpu.dtype) == "torch.bool":
            return [int(i) for i in selector_cpu.nonzero(as_tuple=False).flatten().tolist()]
        return [int(i) for i in selector_cpu.reshape(-1).tolist()]
    if isinstance(selector, slice):
        raise TypeError("slice selectors are not supported for trajectory selection")
    return [int(i) for i in selector]


def _validate_stack_compatible(batches: list[TrajectoryBatch]) -> None:
    first = batches[0]
    for batch in batches[1:]:
        if batch.family != first.family or batch.task != first.task:
            raise ValueError("cannot stack trajectories from different family/task")
        if set(batch.axes) != set(first.axes):
            raise ValueError("cannot stack trajectories with different axes")
        for name, axis in first.axes.items():
            other = batch.axes[name]
            if name == "sample":
                continue
            if axis != other:
                raise ValueError(f"cannot stack trajectories with different axis {name!r}")
        if set(batch.segments) != set(first.segments):
            raise ValueError("cannot stack trajectories with different segments")
        for name, segment in first.segments.items():
            other = batch.segments[name]
            if set(other.tensors) != set(segment.tensors):
                raise ValueError(f"cannot stack segment {name!r} with different tensors")


def _segment_name_for_tensor(data: TrajectoryBatch, target: TrajectoryTensor) -> str:
    for name, segment in data.segments.items():
        if target.name in segment.tensors and segment.tensors[target.name] is target:
            return name
    raise RuntimeError(f"tensor {target.name!r} does not belong to trajectory")


__all__ = [
    "move_trajectory_batch",
    "select_trajectory_batch",
    "slice_trajectory_batch",
    "stack_trajectory_batches",
]
