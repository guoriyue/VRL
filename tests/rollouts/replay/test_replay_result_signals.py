"""Tests for trajectory signal source ownership."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.types import GenerationRequest
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.trajectory.builders import build_diffusion_trajectory


def _denoise_batch() -> tuple[RolloutBatch, torch.Tensor, torch.Tensor]:
    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
    )
    old_log_prob = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=request.sample_rows(),
        observations=torch.zeros(2, 2, 1),
        actions=torch.ones(2, 2, 1),
        old_log_prob=old_log_prob,
        timesteps=torch.zeros(2, 2),
        replay_tensors={},
        context={"model_family": "sd3_5"},
    )
    # The builder records an all-ones mask; the signal tests need a partial one.
    trajectory.segments["denoise"].tensors["mask"].value = mask
    batch = RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
    )
    return batch, old_log_prob, mask


def test_segment_signal_reads_old_logprob_mask_and_distribution_from_trajectory() -> None:
    """The trajectory is the source of ``old_log_prob`` / ``mask`` / ``distribution``."""
    batch, old_log_prob, mask = _denoise_batch()

    signal = TrajectorySignalBuilder(batch).segment_signal(
        segment_name="denoise",
        log_prob=torch.full((2, 2), -1.0),
    )

    assert signal.distribution == "flow_matching"
    assert torch.equal(signal.old_log_prob, old_log_prob)
    assert torch.equal(signal.mask, mask)


def test_signal_builder_requires_a_canonical_trajectory() -> None:
    batch = RolloutBatch(
        rewards=torch.zeros(1),
        group_ids=torch.zeros(1, dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match=r"batch\.trajectory.*TrajectoryBatch"):
        TrajectorySignalBuilder(batch)


def test_signal_builder_rejects_unknown_segment() -> None:
    batch, _, _ = _denoise_batch()

    with pytest.raises(RuntimeError, match="unknown trajectory segment 'missing'"):
        TrajectorySignalBuilder(batch).segment_signal(
            segment_name="missing",
            log_prob=torch.zeros(2, 2),
        )


def test_signal_builder_rejects_mismatched_trajectory_mask() -> None:
    batch, _, _ = _denoise_batch()
    assert batch.trajectory is not None
    batch.trajectory.segments["denoise"].tensors["mask"].value = torch.ones(2, 1)

    with pytest.raises(ValueError, match="log_prob/mask shape mismatch"):
        TrajectorySignalBuilder(batch).single_segment(
            segment_name="denoise",
            log_prob=torch.zeros(2, 2),
        )


@pytest.mark.parametrize("step_dim", [0, 1])
def test_signal_builder_selects_declared_denoise_axis(step_dim: int) -> None:
    from vrl.trajectory.types import TrajectoryAxis

    batch, old_log_prob, mask = _denoise_batch()
    trajectory = batch.trajectory
    assert trajectory is not None
    trajectory.axes["iteration"] = TrajectoryAxis("iteration", "denoise_step", 2)
    segment = trajectory.segments["denoise"]
    for tensor in segment.tensors.values():
        if tensor.role in {"old_log_prob", "mask"}:
            tensor.axes = ("iteration", "sample") if step_dim == 0 else ("sample", "iteration")
            if step_dim == 0:
                tensor.value = tensor.value.transpose(0, 1)

    signal = (
        TrajectorySignalBuilder(batch)
        .single_segment(
            segment_name="denoise",
            log_prob=torch.zeros(2),
            timestep_idx=1,
        )
        .primary
    )

    assert torch.equal(signal.old_log_prob, old_log_prob[:, 1])
    assert torch.equal(signal.mask, mask[:, 1])


def test_signal_builder_does_not_guess_custom_axis_is_denoise_step() -> None:
    """Only an axis declared ``denoise_step`` is sliced by ``timestep_idx``."""
    from vrl.trajectory.types import TrajectoryAxis

    batch, _, _ = _denoise_batch()
    assert batch.trajectory is not None
    batch.trajectory.axes["denoise"] = TrajectoryAxis("denoise", "custom", 2)

    with pytest.raises(ValueError, match="log_prob/old_log_prob shape mismatch"):
        TrajectorySignalBuilder(batch).single_segment(
            segment_name="denoise",
            log_prob=torch.zeros(2),
            timestep_idx=1,
        )
