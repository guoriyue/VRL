"""Tests for pure chunk gatherers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vrl.generation.diffusion import DiffusionChunkGatherer, DiffusionChunkResult
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.execution.planner import build_engine_plan
from vrl.generation.protocols import ChunkGatherer
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.models.diffusion.capabilities import diffusion_family_capability


class _PureGatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[Any],
        chunks: Sequence[Any],
    ) -> GenerationOutput:
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


def test_chunk_gatherer_accepts_pure_object_without_forward_chunk_plan() -> None:
    request = _request()
    sample_rows = build_sample_rows(request)
    gatherer = _PureGatherer()

    assert isinstance(gatherer, ChunkGatherer)
    assert not hasattr(gatherer, "forward_chunk_plan")

    output = gatherer.gather_chunks(request, sample_rows, ["chunk"])

    assert output.output == ["chunk"]


def test_diffusion_chunk_gatherer_gathers_without_model_object() -> None:
    request = _request(cfg=False)
    sample_rows = build_sample_rows(request)
    gatherer = DiffusionChunkGatherer()
    context = {
        "guidance_scale": 4.5,
        "cfg": False,
        "model_family": "sd3_5",
    }

    output = gatherer.gather_chunks(request, sample_rows, _diffusion_chunks(context))

    assert output.output.device.type == "cpu"
    assert output.metrics is not None
    assert output.metrics.num_steps == 2
    assert output.metrics.chunks == 2
    assert output.trajectory is not None
    assert "trajectory" not in output.extra
    assert output.trajectory.segments["denoise"].distribution == "flow_matching"
    assert output.trajectory.axes["sample"].length == 2
    assert output.trajectory.axes["timestep"].length == 2
    assert torch.equal(output.output[:, 0, 0, 0], torch.tensor([1.0, 2.0]))
    assert output.metrics.peak_memory_mb == 20.0


def test_diffusion_chunk_gatherer_orders_prompt_major_chunks() -> None:
    request = _request(cfg=False)
    sample_rows = build_sample_rows(request)
    gatherer = DiffusionChunkGatherer()
    context = {
        "guidance_scale": 4.5,
        "cfg": False,
        "model_family": "sd3_5",
    }

    output = gatherer.gather_chunks(
        request,
        sample_rows,
        list(reversed(_diffusion_chunks(context))),
    )

    assert torch.equal(output.output[:, 0, 0, 0], torch.tensor([1.0, 2.0]))


def test_diffusion_chunk_gatherer_keeps_rollout_context() -> None:
    request = _request(family="cosmos", task="v2w", cfg=False)
    sample_rows = build_sample_rows(request)
    gatherer = DiffusionChunkGatherer()
    context = {
        "guidance_scale": 4.5,
        "cfg": True,
        "model_family": "cosmos",
    }

    output = gatherer.gather_chunks(request, sample_rows, _diffusion_chunks(context))

    assert output.trajectory is not None
    assert output.trajectory.context == context
    assert output.trajectory.segments["denoise"].reward_view == "video"


def test_diffusion_engine_plan_uses_generation_profiler_labels() -> None:
    request = _request(cfg=False)
    sample_rows = build_sample_rows(request)
    plan = build_engine_plan(
        request,
        sample_rows,
        capability=diffusion_family_capability("sd3_5", "t2i"),
        max_samples_per_chunk=1,
    )

    labels = set(plan.profiler_labels)
    stage_names = {stage.name for stage in plan.execution_stages}

    assert {
        "generation.prompt_encode",
        "generation.prepare_sampling",
        "generation.denoise_step",
        "generation.decode_latents",
    }.issubset(labels)
    assert "decode_latents" in stage_names
    assert "vq_decode" not in stage_names
    assert "engine.cache_read" not in labels
    assert "engine.cache_write" not in labels
    assert "engine.vq_decode" not in labels
    assert all(not stage.cache_read for stage in plan.execution_stages)
    assert all(not stage.cache_write for stage in plan.execution_stages)


def _request(
    *,
    family: str = "sd3_5",
    task: str = "t2i",
    cfg: bool = True,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family=family,
        task=task,
        prompts=["p0"],
        samples_per_prompt=2,
        sampling={
            "num_steps": 2,
            "guidance_scale": 4.5,
            "cfg": cfg,
            "seed": 1,
        },
    )


def _diffusion_chunks(context: dict[str, Any]) -> list[DiffusionChunkResult]:
    return [
        _diffusion_chunk(1.0, context, sample_start=0, peak_memory_mb=10.0),
        _diffusion_chunk(2.0, context, sample_start=1, peak_memory_mb=20.0),
    ]


def _diffusion_chunk(
    value: float,
    context: dict[str, Any],
    *,
    sample_start: int,
    peak_memory_mb: float,
) -> DiffusionChunkResult:
    return DiffusionChunkResult(
        prompt_index=0,
        sample_start=sample_start,
        sample_count=1,
        observations=torch.full((1, 2, 1), value),
        actions=torch.full((1, 2, 1), value + 1),
        log_probs=torch.full((1, 2), value + 2),
        timesteps=torch.arange(2).view(1, 2),
        kl=torch.full((1, 2), value + 3),
        video=torch.full((1, 3, 4, 4), value),
        replay_tensors={},
        context=context,
        peak_memory_mb=peak_memory_mb,
    )
