"""Behavioral tests for the public trajectory-builder facades."""

from __future__ import annotations

import torch

from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.types import GenerationRequest
from vrl.trajectory import build_diffusion_trajectory


def test_diffusion_replay_extras_only_declare_sample_axis_when_sample_aligned() -> None:
    request = GenerationRequest(
        request_id="builder-sample-alignment",
        family="test",
        task="t2i",
        inputs=["draw"],
        samples_per_prompt=2,
    )
    sample_rows = build_sample_rows(request)
    old_log_prob = torch.zeros(2, 2)

    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(2, 2, 1),
        actions=torch.zeros(2, 2, 1),
        old_log_prob=old_log_prob,
        timesteps=torch.zeros(2, 2),
        kl=torch.zeros_like(old_log_prob),
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
