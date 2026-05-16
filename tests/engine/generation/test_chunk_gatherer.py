"""Tests for pure chunk gatherers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vrl.engine.core.types import GenerationRequest, OutputBatch
from vrl.engine.diffusion import DiffusionChunkGatherer, DiffusionChunkResult
from vrl.engine.execution.gather import (
    ChunkGatherer,
    gather_pipeline_chunks,
    require_chunk_gatherer,
)
from vrl.engine.execution.ids import GenerationIdFactory


class _PureGatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[Any],
        chunks: Sequence[Any],
    ) -> OutputBatch:
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=list(chunks),
        )


def test_chunk_gatherer_accepts_pure_object_without_forward_chunk_plan() -> None:
    request = _request()
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    gatherer = _PureGatherer()

    assert isinstance(gatherer, ChunkGatherer)
    assert not hasattr(gatherer, "forward_chunk_plan")
    assert require_chunk_gatherer(gatherer) is gatherer

    output = gather_pipeline_chunks(gatherer, request, sample_specs, ["chunk"])

    assert output.output == ["chunk"]


def test_diffusion_chunk_gatherer_gathers_without_model_object() -> None:
    request = _request(cfg=False)
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    gatherer = DiffusionChunkGatherer(model_family="sd3_5")
    context = {
        "guidance_scale": 4.5,
        "cfg": False,
        "model_family": "sd3_5",
    }

    output = gatherer.gather_chunks(request, sample_specs, _diffusion_chunks(context))

    assert output.output.device.type == "cpu"
    assert output.metrics is not None
    assert output.metrics.num_steps == 2
    assert output.metrics.micro_batches == 2
    assert output.trajectory is not None
    assert "trajectory" not in output.extra
    assert output.trajectory.segments["denoise"].distribution == "flow_matching"
    assert output.trajectory.axes["sample"].length == 2
    assert output.trajectory.axes["timestep"].length == 2
    assert torch.equal(output.output[:, 0, 0, 0], torch.tensor([1.0, 2.0]))
    assert output.metrics.peak_memory_mb == 20.0


def test_diffusion_chunk_gatherer_orders_prompt_major_chunks() -> None:
    request = _request(cfg=False)
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    gatherer = DiffusionChunkGatherer(model_family="sd3_5")
    context = {
        "guidance_scale": 4.5,
        "cfg": False,
        "model_family": "sd3_5",
    }

    output = gatherer.gather_chunks(
        request,
        sample_specs,
        list(reversed(_diffusion_chunks(context))),
    )

    assert torch.equal(output.output[:, 0, 0, 0], torch.tensor([1.0, 2.0]))


def test_diffusion_chunk_gatherer_can_ignore_cfg_sampling_flag() -> None:
    request = _request(family="cosmos", task="v2w", cfg=False)
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    gatherer = DiffusionChunkGatherer(
        model_family="cosmos",
        respect_cfg_flag=False,
    )
    context = {
        "guidance_scale": 4.5,
        "cfg": True,
        "model_family": "cosmos",
    }

    output = gatherer.gather_chunks(request, sample_specs, _diffusion_chunks(context))

    assert output.trajectory is not None
    assert output.trajectory.context == context
    assert output.trajectory.segments["denoise"].reward_view == "video"


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
