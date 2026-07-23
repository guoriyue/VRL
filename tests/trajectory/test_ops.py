"""Structural invariants for trajectory select, stack, and move operations."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.types import GenerationRequest
from vrl.trajectory import TrajectoryBatch, build_ar_discrete_trajectory
from vrl.trajectory.ops import (
    move_trajectory_batch,
    select_trajectory_batch,
    stack_trajectory_batches,
)


def test_select_derives_sample_structure_from_selected_rows_and_axis() -> None:
    trajectory = _trajectory(samples=3)

    selected = select_trajectory_batch(trajectory, torch.tensor([2, 0]))

    assert selected.num_samples == 2
    assert selected.axis_lengths == {"sample": 2, "token": 2}
    assert [row.sample_index for row in selected.sample_rows] == [2, 0]
    assert trajectory.num_samples == 3
    assert trajectory.axis_lengths == {"sample": 3, "token": 2}


def test_select_rejects_unsupported_slice_without_partial_rebuild() -> None:
    trajectory = _trajectory(samples=3)

    with pytest.raises(TypeError, match="slice selectors are not supported"):
        select_trajectory_batch(trajectory, slice(0, 2))

    assert trajectory.num_samples == 3
    assert trajectory.axis_lengths["sample"] == 3


def test_stack_derives_combined_sample_structure() -> None:
    first = _trajectory(samples=1, request_id="first")
    second = _trajectory(samples=2, request_id="second")

    stacked = stack_trajectory_batches([first, second])

    assert stacked is not None
    assert stacked.num_samples == 3
    assert stacked.axis_lengths == {"sample": 3, "token": 2}
    assert stacked.metrics.values == {"source": "fixture"}


def test_stack_rejects_different_non_sample_axis_lengths() -> None:
    first = _trajectory(samples=1, token_count=2)
    second = _trajectory(samples=1, token_count=3)

    with pytest.raises(ValueError, match="different axis 'token'"):
        stack_trajectory_batches([first, second])


def test_move_preserves_derived_structure_and_provenance() -> None:
    trajectory = _trajectory(samples=2)

    moved = move_trajectory_batch(trajectory, torch.device("cpu"))

    assert moved is not trajectory
    assert moved.num_samples == 2
    assert moved.axis_lengths == {"sample": 2, "token": 2}
    assert moved.metrics.values == {"source": "fixture"}
    for tensor in moved.segments["image_tokens"].tensors.values():
        assert tensor.value.device.type == "cpu"


def _trajectory(
    *,
    samples: int,
    token_count: int = 2,
    request_id: str = "ops",
) -> TrajectoryBatch:
    request = GenerationRequest(
        request_id=request_id,
        family="test",
        task="t2i",
        inputs=["draw"],
        samples_per_prompt=samples,
    )
    sample_rows = build_sample_rows(request)
    token_ids = torch.arange(samples * token_count, dtype=torch.long).reshape(
        samples,
        token_count,
    )
    prompt_input_ids = torch.ones(samples, 3, dtype=torch.long)
    trajectory = build_ar_discrete_trajectory(
        request=request,
        sample_rows=sample_rows,
        token_ids=token_ids,
        token_log_probs=torch.zeros(samples, token_count),
        token_mask=torch.ones(samples, token_count),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=torch.ones_like(prompt_input_ids),
        uncond_input_ids=torch.zeros_like(prompt_input_ids),
        uncond_attention_mask=torch.ones_like(prompt_input_ids),
        context={},
    )
    trajectory.metrics.values = {"source": "fixture"}
    return trajectory
