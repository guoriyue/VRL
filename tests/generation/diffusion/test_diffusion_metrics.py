"""Tests for diffusion runtime byte counters."""

from __future__ import annotations

import json

import torch

from vrl.generation.diffusion import DiffusionChunkGatherer, DiffusionChunkResult
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.types import GenerationRequest
from vrl.trajectory import trajectory_tensor_bytes


def test_diffusion_chunk_gatherer_aggregates_json_serializable_counters() -> None:
    """Checks diffusion chunk gatherer aggregates JSON serializable counters."""
    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        prompts=["p0"],
        samples_per_prompt=2,
        sampling={"num_steps": 2},
    )
    chunks = [
        _chunk(value=1.0, sample_start=0, stage_duration=0.25),
        _chunk(value=2.0, sample_start=1, stage_duration=0.5),
    ]

    output = DiffusionChunkGatherer().gather_chunks(
        request,
        build_sample_rows(request),
        chunks,
    )

    assert output.metrics is not None
    counters = output.metrics.engine_counters
    assert counters["diffusion_num_denoise_steps"] == 2
    assert counters["diffusion_sample_batch_size"] == 1
    assert counters["diffusion_observation_bytes"] == trajectory_tensor_bytes(
        torch.cat([chunk.observations for chunk in chunks], dim=0),
    )
    assert counters["diffusion_action_bytes"] == trajectory_tensor_bytes(
        torch.cat([chunk.actions for chunk in chunks], dim=0),
    )
    assert counters["diffusion_video_bytes"] == trajectory_tensor_bytes(
        torch.cat([chunk.video for chunk in chunks], dim=0),
    )
    assert counters["stage_durations_s"] == {"denoise": 0.75}
    assert counters["diffusion_storage_device"] == "preserve"
    assert counters["diffusion_storage_dtype"] == "preserve"
    json.dumps(counters)


def _chunk(
    *,
    value: float,
    sample_start: int,
    stage_duration: float,
) -> DiffusionChunkResult:
    observations = torch.full((1, 2, 3), value)
    actions = torch.full((1, 2, 3), value + 1)
    log_probs = torch.full((1, 2), value + 2)
    timesteps = torch.arange(2, dtype=torch.float32).view(1, 2)
    kl = torch.full((1, 2), value + 3)
    video = torch.full((1, 3, 4, 4), value)
    replay_tensors = {"prompt_embeds": torch.full((1, 4), value)}
    return DiffusionChunkResult(
        prompt_index=0,
        sample_start=sample_start,
        sample_count=1,
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        timesteps=timesteps,
        kl=kl,
        video=video,
        replay_tensors=replay_tensors,
        context={"guidance_scale": 4.5, "cfg": False, "model_family": "sd3_5"},
        stage_durations={"denoise": stage_duration},
    )
