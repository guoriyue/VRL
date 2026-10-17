"""Tests for trajectory runtime storage policy."""

from __future__ import annotations

import pytest
import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.trajectory.builders import build_diffusion_trajectory
from vrl.trajectory.storage import TrajectoryStoragePolicy


def test_dtype_policy_only_casts_floating_tensors() -> None:
    """A dtype policy casts floating tensors only; integer replay inputs keep ``torch.long``
    and axis lengths are preserved.
    """
    trajectory = _trajectory()

    result = TrajectoryStoragePolicy(dtype="float16").apply_to_trajectory_(trajectory)

    segment = result.segments["denoise"]
    assert segment.tensors["prompt_ids"].value.dtype == torch.long
    assert segment.tensors["old_log_prob"].value.dtype == torch.float16
    assert segment.tensors["mask"].value.dtype == torch.float16
    assert torch.allclose(
        segment.tensors["old_log_prob"].value.float(),
        torch.tensor([[0.125, -0.25], [0.5, -0.75]]),
        atol=1e-3,
    )
    axis_lengths = {n: a.length for n, a in result.axes.items() if a.length is not None}
    assert axis_lengths == {"sample": 2, "denoise": 2}


def test_rollout_schema_parses_the_storage_policy_and_rejects_bad_values() -> None:
    """``rollout.trajectory_storage`` parses a mapping into the policy; an unknown dtype, a bare
    scalar (``trajectory_storage: cpu``) and a misspelled key each fail instead of degrading to
    the default policy.
    """
    from pydantic import ValidationError

    from vrl.config.schema import RolloutConfig

    parsed = RolloutConfig.model_validate({"trajectory_storage": {"device": "cpu"}})
    assert parsed.trajectory_storage == TrajectoryStoragePolicy(device="cpu")
    for bad in ({"dtype": "int8"}, "cpu", {"device": "cpu", "dytpe": "float16"}):
        with pytest.raises(ValidationError):
            RolloutConfig.model_validate({"trajectory_storage": bad})


def _trajectory():
    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["p0"],
        samples_per_prompt=2,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt="p0",
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]
    old_log_prob = torch.tensor([[0.125, -0.25], [0.5, -0.75]])
    return build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(2, 2, 1),
        actions=torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        old_log_prob=old_log_prob,
        timesteps=torch.zeros(2, 2),
        replay_tensors={"prompt_ids": torch.ones(2, 3, dtype=torch.long)},
        context={"model_family": "sd3_5"},
    )
