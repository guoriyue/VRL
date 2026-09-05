"""Structural invariants for trajectory select and move operations."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.types import GenerationRequest
from vrl.trajectory import TrajectoryBatch, build_ar_discrete_trajectory
from vrl.trajectory.ops import move_trajectory_batch, select_trajectory_batch


def _axis_lengths(trajectory: TrajectoryBatch) -> dict[str, int]:
    return {name: axis.length for name, axis in trajectory.axes.items() if axis.length is not None}


def test_select_derives_sample_structure_from_selected_rows_and_axis() -> None:
    trajectory = _trajectory(samples=3)

    selected = select_trajectory_batch(trajectory, torch.tensor([2, 0]))

    assert len(selected.sample_rows) == 2
    assert _axis_lengths(selected) == {"sample": 2, "token": 2}
    assert [row.sample_index for row in selected.sample_rows] == [2, 0]
    assert len(trajectory.sample_rows) == 3
    assert _axis_lengths(trajectory) == {"sample": 3, "token": 2}


def test_select_rejects_unsupported_slice_without_partial_rebuild() -> None:
    trajectory = _trajectory(samples=3)

    with pytest.raises(TypeError, match="slice selectors are not supported"):
        select_trajectory_batch(trajectory, slice(0, 2))

    assert len(trajectory.sample_rows) == 3
    assert _axis_lengths(trajectory)["sample"] == 3


def test_move_preserves_derived_structure_and_provenance() -> None:
    trajectory = _trajectory(samples=2)

    moved = move_trajectory_batch(trajectory, torch.device("cpu"))

    assert moved is not trajectory
    assert len(moved.sample_rows) == 2
    assert _axis_lengths(moved) == {"sample": 2, "token": 2}
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
    sample_rows = request.sample_rows()
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
    return trajectory
