"""Structural invariants for trajectory select and move operations."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.types import GenerationRequest
from vrl.trajectory import TrajectoryBatch, build_ar_discrete_trajectory


def _axis_lengths(trajectory: TrajectoryBatch) -> dict[str, int]:
    return {name: axis.length for name, axis in trajectory.axes.items() if axis.length is not None}


def test_select_derives_sample_structure_from_selected_rows_and_axis() -> None:
    trajectory = _trajectory(samples=3)

    selected = trajectory.select_samples(torch.tensor([2, 0]))

    assert len(selected.sample_rows) == 2
    assert _axis_lengths(selected) == {"sample": 2, "token": 2}
    assert [row.sample_index for row in selected.sample_rows] == [2, 0]
    assert len(trajectory.sample_rows) == 3
    assert _axis_lengths(trajectory) == {"sample": 3, "token": 2}


def test_move_preserves_derived_structure_and_provenance() -> None:
    trajectory = _trajectory(samples=2)

    moved = trajectory.to_device(torch.device("cpu"))

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


@pytest.mark.parametrize(
    ("selector", "positions"),
    [
        ([True, False, True], [0, 2]),
        (torch.tensor([True, False, True]), [0, 2]),
        ([2, 0], [2, 0]),
        (torch.tensor([2, 0]), [2, 0]),
    ],
)
def test_select_keeps_tensor_rows_and_metadata_aligned(selector, positions) -> None:
    trajectory = _trajectory(samples=3)
    trajectory.context = {"geometry": [4, 8, 16], "schedule": (10, 20, 30)}
    selected = trajectory.select_samples(selector)
    assert [row.sample_index for row in selected.sample_rows] == positions
    assert selected.context == trajectory.context
    assert selected.context is not trajectory.context
    for name, tensor in selected.segments["image_tokens"].tensors.items():
        original = trajectory.segments["image_tokens"].tensors[name]
        if original.axes and original.axes[0] == "sample":
            assert torch.equal(tensor.value, original.value[positions])


@pytest.mark.parametrize(
    "selector",
    [[True, False], [0.5], [True, 0], ["1"], torch.tensor([[0, 1]]), torch.tensor([0.5])],
)
def test_select_rejects_ambiguous_selectors(selector) -> None:
    with pytest.raises(ValueError, match=r"trajectory .*selector"):
        _trajectory(samples=3).select_samples(selector)


def test_sample_selection_preserves_minimax_exported_vae_geometry() -> None:
    from types import SimpleNamespace

    from vrl.models.families.minimax_h3.model import MiniMaxH3Model

    model = SimpleNamespace(_vae_geometry=lambda: (4, 2, 8))
    state = SimpleNamespace(height=32, width=32, num_frames=4, fps=24, timesteps=torch.ones(2))
    trajectory = _trajectory(samples=3)
    trajectory.context = MiniMaxH3Model.export_batch_context(model, state)
    selected = trajectory.select_samples([2, 0])
    assert selected.context["vae_geometry"] == [4, 2, 8]
    assert len(selected.sample_rows) == 2


@pytest.mark.parametrize("positions", [[2, 0, 1], [2, 0]])
@pytest.mark.parametrize("container", ["tensor", "list", "tuple"])
def test_selection_uses_declared_nonleading_sample_axis(positions, container) -> None:
    from vrl.trajectory.types import TrajectoryTensor
    from vrl.trajectory.validation import TrajectoryValidator

    trajectory = _trajectory(samples=3)
    values = torch.arange(6).reshape(2, 3)
    payload = values
    if container == "list":
        payload = values.tolist()
    elif container == "tuple":
        payload = tuple(tuple(row) for row in values.tolist())
    trajectory.segments["image_tokens"].tensors["token_sample_cache"] = TrajectoryTensor(
        "token_sample_cache", payload, ("token", "sample"), "replay_input"
    )
    TrajectoryValidator(trajectory).validate_batch()
    selected = trajectory.select_samples(positions)
    actual = selected.segments["image_tokens"].tensors["token_sample_cache"].value
    assert isinstance(actual, type(payload))
    assert torch.equal(torch.as_tensor(actual), values[:, positions])
    assert [row.sample_index for row in selected.sample_rows] == positions
