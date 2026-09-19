"""Contract tests for the shared chunk-autoregressive denoise binding."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from vrl.generation.bindings.chunk_autoregressive_denoise import (
    ChunkAutoregressiveDenoiseExecutorBase,
    ChunkAutoregressiveDenoiseGatherer,
    ChunkAutoregressiveDenoiseResult,
)
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.trajectory.types import TrajectoryTensor
from vrl.utils.media_reference import MediaReference


def test_trainable_trajectory_declares_temporal_chunk_and_transition_axes() -> None:
    request = _request()
    sample_rows = request.sample_rows()

    output = ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
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
    assert trajectory.context == {"model_family": "causvid"}


def test_gatherer_orders_transport_chunks_and_concatenates_sample_rows() -> None:
    request = _request()
    sample_rows = request.sample_rows()

    output = ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
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


@pytest.mark.parametrize("trainable", [True, False])
def test_media_references_survive_chunk_gather_in_sample_order(trainable):
    request = replace(_request(), reward_media_refs=True)
    make_result = _trainable_result if trainable else _generation_only_result
    batches = [make_result(0.75, sample_start=1), make_result(0.25, sample_start=0)]
    for batch in batches:
        batch.output = [MediaReference(f"batch-{batch.batch.sample_start}", 0)]
    output = ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
        request, request.sample_rows(), batches
    )
    assert [ref.object_ref for ref in output.output] == ["batch-0", "batch-1"]
    if trainable:
        assert torch.equal(
            output.trajectory.segments["denoise"].tensors["old_log_prob"].value[:, 0, 0],
            torch.tensor([0.25, 0.75]),
        )
    else:
        assert output.trajectory.segments["generated_chunks"].tensors == {}


def test_generation_only_result_has_no_fabricated_policy_facts() -> None:
    request = _request()
    sample_rows = request.sample_rows()
    batches = [
        _generation_only_result(20.0, sample_start=1),
        _generation_only_result(10.0, sample_start=0),
    ]

    output = ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
        request,
        sample_rows,
        batches,
    )

    assert output.trajectory is not None
    trajectory = output.trajectory
    assert torch.equal(output.output[:, 0], torch.tensor([10.0, 20.0]))
    assert trajectory.axes["temporal_chunk"].length == 2
    assert "denoise_transition" not in trajectory.axes
    segment = trajectory.segments["generated_chunks"]
    assert segment.trainable is False
    assert trajectory.primary_segment is None
    assert segment.tensors == {}
    assert trajectory.reward_views["video"].metadata["output_ref"] == "GenerationOutput.output"
    assert trajectory.context == {"model_family": "causvid"}

    builder = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}),
    )
    assert torch.equal(builder.reward_outputs(), output.output)
    with pytest.raises(RuntimeError, match="generation-only trajectory"):
        builder.build(torch.ones(2))


def test_gatherer_rejects_mismatched_batch_context() -> None:
    request = _request()
    batches = [
        _generation_only_result(10.0, sample_start=0),
        _generation_only_result(20.0, sample_start=1),
    ]
    batches[1].context = {"model_family": "different"}

    with pytest.raises(ValueError, match="batch context at ordered index 1 does not match"):
        ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
            request,
            request.sample_rows(),
            batches,
        )


class _GenericChunkExecutor(ChunkAutoregressiveDenoiseExecutorBase):
    """Minimal Base subclass exercising forward_plan delegation."""

    family = "causvid"
    task = "t2v"


def test_generic_executor_delegates_temporal_generation_to_model() -> None:
    request = _request()
    sample_rows = request.sample_rows()
    model = _FakeChunkModel()
    executor = _GenericChunkExecutor(
        model,
        gatherer=ChunkAutoregressiveDenoiseGatherer(),
    )

    plan = EnginePlan.from_request(request, max_samples_per_batch=1)
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
        batch: Any,
    ) -> ChunkAutoregressiveDenoiseResult:
        del request
        self.calls.append((batch.prompt_index, batch.sample_start, batch.sample_count))
        return ChunkAutoregressiveDenoiseResult(
            batch=batch,
            output=torch.full((batch.sample_count, 1), float(batch.sample_start)),
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
        batch=GenerationSampleBatch(prompt_index=0, sample_start=sample_start, sample_count=1),
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
            "transition_noise": TrajectoryTensor(
                "transition_noise",
                torch.full((1, 2, 3, 1), value + 6),
                ("sample", "temporal_chunk", "denoise_transition"),
                "replay_input",
            ),
            "cache_position": TrajectoryTensor(
                "cache_position",
                torch.arange(2).view(1, 2),
                ("sample", "temporal_chunk"),
                "replay_input",
            ),
        },
        context={"model_family": "causvid"},
    )


def _generation_only_result(
    value: float,
    *,
    sample_start: int,
) -> ChunkAutoregressiveDenoiseResult:
    return ChunkAutoregressiveDenoiseResult(
        batch=GenerationSampleBatch(prompt_index=0, sample_start=sample_start, sample_count=1),
        output=torch.full((1, 1), value),
        temporal_chunk_count=2,
        denoise_transition_count=3,
        context={"model_family": "causvid"},
    )


@pytest.mark.parametrize("field_name", ["actions", "finalized_chunk_latents"])
def test_gatherer_rejects_result_with_misaligned_trajectory_axes(field_name: str) -> None:
    request = _request()
    batches = [_trainable_result(10.0, sample_start=0), _trainable_result(20.0, sample_start=1)]
    setattr(batches[1], field_name, torch.zeros(1, 4, 3))

    with pytest.raises(ValueError, match=f"batch {field_name} has leading dimensions"):
        ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
            request, request.sample_rows(), batches
        )


@pytest.mark.parametrize("field_name", ["temporal_chunk_count", "denoise_transition_count"])
@pytest.mark.parametrize("value", [True, 2.0, 2.5, "2", -1])
def test_result_rejects_noninteger_or_negative_axis_counts(field_name, value):
    kwargs = {"temporal_chunk_count": 2, field_name: value}
    with pytest.raises(ValueError, match=field_name):
        ChunkAutoregressiveDenoiseResult(
            batch=GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1),
            output=torch.zeros(1, 1),
            **kwargs,
        )


@pytest.mark.parametrize("transition_count", [None, 0, 3])
def test_generation_only_result_preserves_optional_transition_count(transition_count):
    result = ChunkAutoregressiveDenoiseResult(
        batch=GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1),
        output=torch.zeros(1, 1),
        temporal_chunk_count=2,
        denoise_transition_count=transition_count,
    )
    assert result.denoise_transition_count == transition_count
    assert not result.has_trainable_trajectory


def test_prompt_embedding_dimensions_do_not_become_chunk_axes() -> None:
    request = _request()
    result = _trainable_result(10.0, sample_start=0)
    other = _trainable_result(20.0, sample_start=1)
    # Token count and embedding width happen to equal chunk/transition counts.
    for batch in (result, other):
        batch.replay_tensors["prompt_embeds"] = TrajectoryTensor(
            "prompt_embeds", torch.ones(1, 2, 3), ("sample",), "replay_input"
        )
    output = ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
        request, request.sample_rows(), [result, other]
    )
    from vrl.trajectory.reader import TrajectoryReader

    reader = TrajectoryReader(output.trajectory)
    replay = reader.replay_tensor_dict("denoise", axis="temporal_chunk", axis_index=1)
    assert replay["prompt_embeds"].shape == (2, 2, 3)
    assert replay["transition_noise"].shape == (2, 3, 1)


@pytest.mark.parametrize("invalid_axes", [None, ("sample",)])
def test_gather_rejects_missing_records_or_inconsistent_axes(invalid_axes) -> None:
    request = _request()
    batches = [_trainable_result(10.0, sample_start=0), _trainable_result(20.0, sample_start=1)]
    if invalid_axes is None:
        del batches[1].replay_tensors["transition_noise"]
        match = "keys must match"
    else:
        batches[1].replay_tensors["transition_noise"] = replace(
            batches[1].replay_tensors["transition_noise"], axes=invalid_axes
        )
        match = "must declare the same"
    with pytest.raises(ValueError, match=match):
        ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
            request, request.sample_rows(), batches
        )


def test_serialized_replay_records_preserve_axes_values_and_sample_order() -> None:
    import cloudpickle

    from vrl.trajectory.reader import TrajectoryReader

    request = _request()
    batches = cloudpickle.loads(
        cloudpickle.dumps(
            [
                _trainable_result(20.0, sample_start=1),
                _trainable_result(10.0, sample_start=0),
            ]
        )
    )
    original = batches[0].replay_tensors["transition_noise"]
    output = ChunkAutoregressiveDenoiseGatherer().merge_generation_batches(
        request, request.sample_rows(), batches
    )
    tensor = output.trajectory.segments["denoise"].tensors["transition_noise"]
    assert isinstance(tensor, TrajectoryTensor)
    assert tensor.axes == ("sample", "temporal_chunk", "denoise_transition")
    assert tensor.role == "replay_input"
    assert tensor.value[:, 0, 0, 0].tolist() == [16.0, 26.0]
    assert original.value.shape == (1, 2, 3, 1)
    replay = TrajectoryReader(output.trajectory).replay_tensor_dict(
        "denoise", axis="temporal_chunk", axis_index=1
    )
    assert replay["transition_noise"].shape == (2, 3, 1)
    assert replay["transition_noise"][:, 0, 0].tolist() == [16.0, 26.0]
