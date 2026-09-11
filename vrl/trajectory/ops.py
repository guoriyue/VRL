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
)
from vrl.trajectory.validation import TrajectoryValidator


def select_trajectory_batch(data: Any, selector: Any) -> Any:
    """Select TrajectoryBatch rows by boolean mask or integer indices."""

    if data is None:
        return None
    if not isinstance(data, TrajectoryBatch):
        return data

    positions = _selector_positions(selector, len(data.sample_rows))
    count = len(positions)
    return _rebuild_trajectory(
        data,
        sample_rows=[data.sample_rows[i] for i in positions],
        tensor_value_fn=lambda tensor: (
            _select_value(tensor.value, positions, tensor.axes.index("sample"))
            if "sample" in tensor.axes
            else tensor.value
        ),
        axes_sample_length=count,
        context=dict(data.context),
    )


def move_trajectory_batch(data: Any, device: Any) -> Any:
    """Move tensor leaves in a TrajectoryBatch to a target device."""

    if data is None:
        return None
    if not isinstance(data, TrajectoryBatch):
        return data

    return _rebuild_trajectory(
        data,
        sample_rows=list(data.sample_rows),
        tensor_value_fn=lambda tensor: move_value_to_device(tensor.value, device),
        axes_sample_length=data.axes["sample"].length,
        context=move_value_to_device(data.context, device),
    )


def _rebuild_trajectory(
    data: TrajectoryBatch,
    *,
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
        name: replace(
            segment,
            tensors={
                tensor_name: replace(tensor, value=tensor_value_fn(tensor))
                for tensor_name, tensor in segment.tensors.items()
            },
            replay_inputs=dict(segment.replay_inputs),
            metadata=dict(segment.metadata),
        )
        for name, segment in data.segments.items()
    }

    out = replace(
        data,
        sample_rows=sample_rows,
        axes=axes,
        segments=segments,
        reward_views=dict(data.reward_views),
        context=context,
    )
    return TrajectoryValidator(out).validate_batch()


def _select_value(value: Any, positions: list[int], axis_dim: int) -> Any:
    """Select the declared sample dimension, including nested Python payloads."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        selected = (
            [value[i] for i in positions]
            if axis_dim == 0
            else [_select_value(inner, positions, axis_dim - 1) for inner in value]
        )
        return tuple(selected) if isinstance(value, tuple) else selected
    if isinstance(value, dict):
        return {key: _select_value(inner, positions, axis_dim) for key, inner in value.items()}
    key = [slice(None)] * (axis_dim + 1)
    key[axis_dim] = positions
    return value[tuple(key)]


def _selector_positions(selector: Any, batch_size: int) -> list[int]:
    """Normalize one-dimensional masks/indices once for every sample-aligned value."""
    is_boolean_mask = False
    if hasattr(selector, "detach"):
        if selector.is_floating_point() or selector.is_complex():
            raise ValueError("trajectory selector must contain only booleans or only integers")
        is_boolean_mask = str(selector.dtype) == "torch.bool"
        if selector.ndim != 1:
            raise ValueError("trajectory selector must be one-dimensional")
        values = selector.detach().cpu().tolist()
    else:
        values = list(selector)
    if is_boolean_mask or (values and all(isinstance(value, bool) for value in values)):
        if len(values) != batch_size:
            raise ValueError("trajectory boolean selector must match the sample count")
        return [index for index, selected in enumerate(values) if selected]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("trajectory selector must contain only booleans or only integers")
    return values


__all__ = [
    "move_trajectory_batch",
    "select_trajectory_batch",
]
