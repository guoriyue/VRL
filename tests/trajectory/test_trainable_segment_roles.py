"""A trainable segment carries exactly one action, old_log_prob, and mask tensor."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.types import GenerationRequest
from vrl.trajectory.builders import build_diffusion_trajectory
from vrl.trajectory.types import TrajectoryBatch, TrajectoryTensor
from vrl.trajectory.validation import TrajectoryValidationError, TrajectoryValidator


def _trajectory() -> TrajectoryBatch:
    request = GenerationRequest(
        request_id="trainable-roles",
        family="test",
        task="t2i",
        inputs=["draw"],
        samples_per_prompt=2,
    )
    return build_diffusion_trajectory(
        request=request,
        sample_rows=request.sample_rows(),
        observations=torch.zeros(2, 3, 1),
        actions=torch.ones(2, 3, 1),
        old_log_prob=torch.zeros(2, 3),
        timesteps=torch.zeros(2, 3),
        replay_tensors={},
        context={},
    )


def test_trainable_segment_without_a_mask_is_rejected() -> None:
    trajectory = _trajectory()
    del trajectory.segments["denoise"].tensors["mask"]
    with pytest.raises(TrajectoryValidationError, match="missing required roles: mask"):
        TrajectoryValidator(trajectory).validate_batch()


def test_trainable_segment_with_a_second_action_is_rejected() -> None:
    """A second action tensor would make the replayed action ambiguous."""
    trajectory = _trajectory()
    tensors = trajectory.segments["denoise"].tensors
    actions = tensors["actions"]
    tensors["actions_copy"] = TrajectoryTensor(
        "actions_copy", actions.value.clone(), actions.axes, "action"
    )
    with pytest.raises(TrajectoryValidationError, match="has multiple 'action' tensors"):
        TrajectoryValidator(trajectory).validate_batch()
