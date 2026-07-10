"""Tests for pure chunk gatherers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vrl.generation.diffusion import DiffusionChunkGatherer, DiffusionChunkResult
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.protocols import ChunkGatherer
from vrl.generation.types import GenerationOutput, GenerationRequest


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
    """Checks chunk gatherer accepts pure object without forward chunk plan."""
    request = _request()
    sample_rows = build_sample_rows(request)
    gatherer = _PureGatherer()

    assert isinstance(gatherer, ChunkGatherer)
    assert not hasattr(gatherer, "forward_chunk_plan")

    output = gatherer.gather_chunks(request, sample_rows, ["chunk"])

    assert output.output == ["chunk"]


def test_diffusion_chunk_gatherer_gathers_without_model_object() -> None:
    """Checks diffusion chunk gatherer gathers without model object."""
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
    assert output.metrics.engine_counters["diffusion_num_denoise_steps"] == 2
    assert output.metrics.engine_counters["diffusion_video_bytes"] == (
        output.output.numel() * output.output.element_size()
    )
    assert output.trajectory.segments["denoise"].distribution == "flow_matching"
    assert output.trajectory.axes["sample"].length == 2
    assert output.trajectory.axes["denoise"].length == 2
    assert torch.equal(output.output[:, 0, 0, 0], torch.tensor([1.0, 2.0]))
    assert output.metrics.peak_memory_mb == 20.0


def test_diffusion_chunk_gatherer_orders_prompt_major_chunks() -> None:
    """Checks diffusion chunk gatherer orders prompt major chunks."""
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
    """Checks diffusion chunk gatherer keeps rollout context."""
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
