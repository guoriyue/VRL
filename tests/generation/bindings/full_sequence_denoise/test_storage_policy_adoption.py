"""Tests for diffusion trajectory storage policy adoption."""

from __future__ import annotations

import torch

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionChunkGatherer,
    DiffusionChunkResult,
)
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.types import GenerationRequest
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.trajectory import TrajectoryStoragePolicy


def test_diffusion_rollout_batch_builder_applies_storage_policy() -> None:
    """Checks diffusion rollout batch builder applies storage policy."""
    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["p0"],
        samples_per_prompt=1,
        sampling={"num_steps": 2},
    )
    output = DiffusionChunkGatherer().gather_chunks(
        request,
        build_sample_rows(request),
        [_chunk()],
    )

    batch = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(
            metadata={},
            trajectory_storage_policy=TrajectoryStoragePolicy(device="cpu", dtype="float16"),
        ),
    ).build(torch.tensor([1.0]))

    assert batch.observations.device.type == "cpu"
    assert batch.observations.dtype == torch.float16
    assert batch.actions.dtype == torch.float16
    assert batch.trajectory is output.trajectory


def _chunk() -> DiffusionChunkResult:
    return DiffusionChunkResult(
        prompt_index=0,
        sample_start=0,
        sample_count=1,
        observations=torch.ones(1, 2, 3, dtype=torch.float32),
        actions=torch.ones(1, 2, 3, dtype=torch.float32) * 2,
        log_probs=torch.ones(1, 2, dtype=torch.float32) * 3,
        timesteps=torch.arange(2, dtype=torch.float32).view(1, 2),
        kl=torch.ones(1, 2, dtype=torch.float32) * 4,
        video=torch.ones(1, 3, 4, 4, dtype=torch.float32),
        replay_tensors={},
        context={"guidance_scale": 4.5, "cfg": False, "model_family": "sd3_5"},
    )
