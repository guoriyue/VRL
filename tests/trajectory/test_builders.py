"""Behavioral tests for the public trajectory-builder facades."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.types import GenerationRequest
from vrl.trajectory.builders import build_diffusion_trajectory
from vrl.trajectory.types import TrajectoryBatch
from vrl.trajectory.validation import TrajectoryValidationError


def _axis_lengths(trajectory: TrajectoryBatch) -> dict[str, int]:
    return {name: axis.length for name, axis in trajectory.axes.items() if axis.length is not None}


def test_all_builders_derive_structure() -> None:
    for name, trajectory, expected_axis_lengths in _structural_trajectories():
        assert len(trajectory.sample_rows) == expected_axis_lengths["sample"], name
        assert _axis_lengths(trajectory) == expected_axis_lengths, name


def test_builder_rejects_tensor_rows_that_disagree_with_sample_rows() -> None:
    request = _request()
    sample_rows = request.sample_rows()

    with pytest.raises(
        TrajectoryValidationError,
        match=r"observations axis 'sample' has shape 1, expected 2",
    ):
        build_diffusion_trajectory(
            request=request,
            sample_rows=sample_rows,
            observations=torch.zeros(1, 2, 1),
            actions=torch.zeros(2, 2, 1),
            old_log_prob=torch.zeros(2, 2),
            timesteps=torch.zeros(2, 2),
            replay_tensors={},
            context={},
        )


def test_diffusion_replay_extras_only_declare_sample_axis_when_sample_aligned(caplog) -> None:
    import logging

    caplog.set_level(logging.DEBUG, logger="vrl.trajectory.builders")
    request = GenerationRequest(
        request_id="builder-sample-alignment",
        family="test",
        task="t2i",
        inputs=["draw"],
        samples_per_prompt=2,
    )
    sample_rows = request.sample_rows()
    old_log_prob = torch.zeros(2, 2)

    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(2, 2, 1),
        actions=torch.zeros(2, 2, 1),
        old_log_prob=old_log_prob,
        timesteps=torch.zeros(2, 2),
        replay_tensors={
            "per_sample": torch.tensor([1.0, 2.0]),
            "scalar_tensor": torch.tensor(1.0),
            "python_scalar": 1.0,
        },
        context={},
    )

    tensors = trajectory.segments["denoise"].tensors
    assert tensors["per_sample"].axes == ("sample",)
    assert "scalar_tensor" not in tensors
    assert "python_scalar" not in tensors
    assert "['python_scalar', 'scalar_tensor']" in caplog.text  # left-out names are diagnosable


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="derived-structure",
        family="test",
        task="t2i",
        inputs=["draw"],
        samples_per_prompt=2,
    )


def _structural_trajectories() -> list[tuple[str, TrajectoryBatch, dict[str, int]]]:
    request = _request()
    sample_rows = request.sample_rows()
    batch_size = len(sample_rows)

    diffusion_steps = 3
    diffusion_policy_shape = (batch_size, diffusion_steps)
    diffusion = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(*diffusion_policy_shape, 1),
        actions=torch.ones(*diffusion_policy_shape, 1),
        old_log_prob=torch.zeros(diffusion_policy_shape),
        timesteps=torch.zeros(diffusion_policy_shape),
        replay_tensors={},
        context={},
    )

    return [
        (
            "diffusion",
            diffusion,
            {"sample": batch_size, "denoise": diffusion_steps},
        ),
    ]
