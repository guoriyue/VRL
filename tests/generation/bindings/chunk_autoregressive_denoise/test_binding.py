"""Contract tests for the shared chunk-autoregressive denoise binding."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.generation.bindings.chunk_autoregressive_denoise import (
    ChunkAutoregressiveDenoiseExecutor,
    ChunkAutoregressiveDenoiseGatherer,
    ChunkAutoregressiveDenoiseResult,
)
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.types import GenerationRequest
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)


def test_trainable_trajectory_declares_temporal_chunk_and_transition_axes() -> None:
    request = _request()
    sample_rows = build_sample_rows(request)

    output = ChunkAutoregressiveDenoiseGatherer().gather_chunks(
        request,
        sample_rows,
        [_trainable_result(20.0, sample_start=1), _trainable_result(10.0, sample_start=0)],
    )

    assert output.trajectory is not None
    trajectory = output.trajectory
    assert trajectory.axes["temporal_chunk"].kind == "temporal_chunk"
    assert trajectory.axes["temporal_chunk"].length == 2
    assert trajectory.axes["denoise_transition"].kind == "denoise_transition"
    assert trajectory.axes["denoise_transition"].length == 3
    segment = trajectory.segments["denoise"]
    assert segment.trainable is True
    assert segment.tensors["actions"].axes == (
        "sample",
        "temporal_chunk",
        "denoise_transition",
    )
    assert segment.tensors["finalized_chunk_latents"].axes == (
        "sample",
        "temporal_chunk",
    )
    assert segment.tensors["transition_noise"].axes == (
        "sample",
        "temporal_chunk",
        "denoise_transition",
    )
    assert torch.count_nonzero(segment.tensors["kl"].value) == 0
    assert trajectory.context == {"model_family": "causvid"}


def test_gatherer_orders_transport_chunks_and_concatenates_sample_rows() -> None:
    request = _request()
    sample_rows = build_sample_rows(request)

    output = ChunkAutoregressiveDenoiseGatherer().gather_chunks(
        request,
        sample_rows,
        [_trainable_result(20.0, sample_start=1), _trainable_result(10.0, sample_start=0)],
    )

    assert torch.equal(output.output[:, 0], torch.tensor([10.0, 20.0]))
    assert output.trajectory is not None
    denoise = output.trajectory.segments["denoise"].tensors
    assert torch.equal(denoise["old_log_prob"].value[:, 0, 0], torch.tensor([10.0, 20.0]))
    assert torch.equal(
        denoise["finalized_chunk_latents"].value[:, 0, 0],
        torch.tensor([15.0, 25.0]),
    )


def test_generation_only_result_has_no_fabricated_policy_facts() -> None:
    request = _request()
    sample_rows = build_sample_rows(request)
    chunks = [
        _generation_only_result(20.0, sample_start=1),
        _generation_only_result(10.0, sample_start=0),
    ]

    output = ChunkAutoregressiveDenoiseGatherer().gather_chunks(
        request,
        sample_rows,
        chunks,
    )

    assert output.trajectory is not None
    trajectory = output.trajectory
    assert torch.equal(output.output[:, 0], torch.tensor([10.0, 20.0]))
    assert trajectory.axes["temporal_chunk"].length == 2
    assert "denoise_transition" not in trajectory.axes
    segment = trajectory.segments["generated_chunks"]
    assert segment.trainable is False
    assert trajectory.primary_segment is None
    assert set(segment.tensors) == {"output"}
    assert segment.tensors["output"].role == "replay_input"
    assert trajectory.context == {"model_family": "causvid"}

    builder = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}),
    )
    assert torch.equal(builder.reward_outputs(), output.output)
    with pytest.raises(RuntimeError, match="generation-only trajectory"):
        builder.build(torch.ones(2))


def test_generic_executor_delegates_temporal_generation_to_model() -> None:
    request = _request()
    sample_rows = build_sample_rows(request)
    model = _FakeChunkModel()
    executor = ChunkAutoregressiveDenoiseExecutor(
        model,
        family=request.family,
        task=request.task,
        samples_per_chunk=1,
    )

    plan = executor.plan(request)
    output = executor.forward_plan(request, sample_rows, plan)

    assert model.calls == [(0, 0, 1), (0, 1, 1)]
    assert torch.equal(output.output[:, 0], torch.tensor([0.0, 1.0]))
    assert output.trajectory is not None
    assert output.trajectory.segments["generated_chunks"].trainable is False


class _FakeChunkModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def generate_chunk_autoregressive(
        self,
        *,
        request: GenerationRequest,
        chunk: Any,
    ) -> ChunkAutoregressiveDenoiseResult:
        del request
        self.calls.append((chunk.prompt_index, chunk.sample_start, chunk.sample_count))
        return ChunkAutoregressiveDenoiseResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=torch.full((chunk.sample_count, 1), float(chunk.sample_start)),
            temporal_chunk_count=2,
            context={"model_family": "fake"},
        )


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="causvid",
        task="t2v",
        inputs=["a video prompt"],
        samples_per_prompt=2,
        sampling={},
    )


def _trainable_result(
    value: float,
    *,
    sample_start: int,
) -> ChunkAutoregressiveDenoiseResult:
    return ChunkAutoregressiveDenoiseResult(
        prompt_index=0,
        sample_start=sample_start,
        sample_count=1,
        output=torch.full((1, 1), value),
        temporal_chunk_count=2,
        denoise_transition_count=3,
        observations=torch.full((1, 2, 3, 1), value - 1),
        actions=torch.full((1, 2, 3, 1), value + 1),
        old_log_prob=torch.full((1, 2, 3), value),
        mask=torch.ones((1, 2, 3)),
        timesteps=torch.arange(3).view(1, 1, 3).expand(1, 2, 3),
        finalized_chunk_latents=torch.full((1, 2, 1), value + 5),
        replay_tensors={
            "transition_noise": torch.full((1, 2, 3, 1), value + 6),
            "cache_position": torch.arange(2).view(1, 2),
        },
        context={"model_family": "causvid"},
    )


def _generation_only_result(
    value: float,
    *,
    sample_start: int,
) -> ChunkAutoregressiveDenoiseResult:
    return ChunkAutoregressiveDenoiseResult(
        prompt_index=0,
        sample_start=sample_start,
        sample_count=1,
        output=torch.full((1, 1), value),
        temporal_chunk_count=2,
        denoise_transition_count=3,
        context={"model_family": "causvid"},
    )
