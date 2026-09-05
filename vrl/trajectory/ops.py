"""Operations over trajectory batches.

These helpers keep sample-axis slicing and device movement in one place so
batching, rollout packing, and trainer utilities do not each invent a slightly
different trajectory convention.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vrl.generation.types import GenerationSampleRow
from vrl.trajectory.device import move_value_to_device
from vrl.trajectory.types import (
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.trajectory.validation import TrajectoryValidator


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
        sample_rows=[data.sample_rows[i] for i in positions],
        tensor_value_fn=lambda tensor: (
            _select_value(tensor.value, selector, len(data.sample_rows))
            if tensor.axes and tensor.axes[0] == "sample"
            else tensor.value
        ),
        axes_sample_length=count,
        context=_select_value(data.context, selector, len(data.sample_rows)),
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
        sample_rows=list(data.sample_rows),
        tensor_value_fn=lambda tensor: move_value_to_device(tensor.value, device),
        axes_sample_length=data.axes["sample"].length,
        context=move_value_to_device(data.context, device),
    )


def _rebuild_trajectory(
    data: TrajectoryBatch,
    *,
    request_id: str,
    family: str,
    task: str,
    sample_rows: list[GenerationSampleRow],
    tensor_value_fn: Any,
    axes_sample_length: int | None,
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
                )
                for tensor_name, tensor in segment.tensors.items()
            },
            reward_view=segment.reward_view,
            replay_inputs=dict(segment.replay_inputs),
            metadata=dict(segment.metadata),
        )
        for name, segment in data.segments.items()
    }

    out = TrajectoryBatch(
        request_id=request_id,
        family=family,
        task=task,
        sample_rows=sample_rows,
        axes=axes,
        segments=segments,
        primary_segment=data.primary_segment,
        reward_views=dict(data.reward_views),
        context=context,
    )
    return TrajectoryValidator(out).validate_batch()


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


def _selector_positions(selector: Any) -> list[int]:
    if hasattr(selector, "detach"):
        selector_cpu = selector.detach().cpu()
        if str(selector_cpu.dtype) == "torch.bool":
            return [int(i) for i in selector_cpu.nonzero(as_tuple=False).flatten().tolist()]
        return [int(i) for i in selector_cpu.reshape(-1).tolist()]
    if isinstance(selector, slice):
        raise TypeError("slice selectors are not supported for trajectory selection")
    return [int(i) for i in selector]


__all__ = [
    "move_trajectory_batch",
    "select_trajectory_batch",
]
