"""Tests for pure chunk gatherers."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionChunkGatherer,
    DiffusionChunkResult,
)
from vrl.generation.execution.chunks import (
    SampleAlignedValues,
    gather_replay_tensors,
    require_matching_chunk_context,
)
from vrl.generation.execution.executor_base import ChunkExecutorBase
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkGatherer
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.models.families.cosmos.cosmos3.model import Cosmos3Model


class _PureGatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[Any],
        chunks: Sequence[Any],
    ) -> GenerationOutput:
        return GenerationOutput(
            request_id=request.request_id,
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


class _Executor(ChunkExecutorBase):
    family = "sd3_5"
    task = "t2i"

    def forward_chunk_plan(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_chunk_executor_uses_injected_gatherer() -> None:
    request = _request()
    sample_rows = build_sample_rows(request)
    gatherer = _PureGatherer()
    executor = _Executor(gatherer=gatherer)

    output = executor.gather_chunks(request, sample_rows, ["chunk"])

    assert output.output == ["chunk"]
    assert executor._gatherer is gatherer


def test_chunk_executor_rejects_request_execution_without_gatherer() -> None:
    request = _request()

    with pytest.raises(RuntimeError, match="requires an injected chunk gatherer"):
        _Executor().gather_chunks(request, build_sample_rows(request), ["chunk"])


def test_chunk_gatherer_accepts_pure_object_without_forward_chunk_plan() -> None:
    """Checks chunk gatherer accepts pure object without forward chunk plan."""
    request = _request()
    sample_rows = build_sample_rows(request)
    gatherer = _PureGatherer()

    assert isinstance(gatherer, ChunkGatherer)
    assert not hasattr(gatherer, "forward_chunk_plan")

    output = gatherer.gather_chunks(request, sample_rows, ["chunk"])

    assert output.output == ["chunk"]


def test_worker_core_rejects_invalid_launch_contract_at_process_boundary() -> None:
    with pytest.raises(
        TypeError,
        match="launch_contract must be a GenerationRuntimeLaunchContract, got object",
    ):
        GenerationWorkerCore("worker-0", object(), _PureGatherer())


@pytest.mark.parametrize(
    "gatherer",
    [object(), SimpleNamespace(gather_chunks=object())],
)
def test_worker_core_rejects_invalid_gatherer_at_process_boundary(
    gatherer: Any,
) -> None:
    with pytest.raises(TypeError, match="gatherer must implement ChunkGatherer"):
        GenerationWorkerCore(
            "worker-0",
            GenerationRuntimeLaunchContract(
                family="test",
                model_build={},
                expected_model_identity={"schema": "test"},
            ),
            gatherer,
        )


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
    assert output.trajectory is not None
    assert "trajectory" not in output.extra
    assert output.trajectory.segments["denoise"].distribution == "flow_matching"
    assert output.trajectory.axes["sample"].length == 2
    assert output.trajectory.axes["denoise"].length == 2
    assert torch.equal(output.output[:, 0, 0, 0], torch.tensor([1.0, 2.0]))


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


def test_diffusion_chunk_gatherer_strictly_merges_replay_values() -> None:
    request = _request(cfg=False)
    chunks = _diffusion_chunks(
        {
            "guidance_scale": 4.5,
            "cfg": False,
            "model_family": "sd3_5",
        },
    )
    chunks[0].replay_tensors = {
        "prompt_embeds": torch.tensor([[1.0]]),
        "optional": None,
        "scheduler": "flow",
    }
    chunks[1].replay_tensors = {
        "prompt_embeds": torch.tensor([[2.0]]),
        "optional": None,
        "scheduler": "flow",
    }

    output = DiffusionChunkGatherer().gather_chunks(
        request,
        build_sample_rows(request),
        chunks,
    )

    assert output.trajectory is not None
    prompt_embeds = output.trajectory.segments["denoise"].tensors["prompt_embeds"].value
    assert torch.equal(prompt_embeds, torch.tensor([[1.0], [2.0]]))


def test_replay_gather_treats_plain_sequences_as_static_values() -> None:
    gathered = gather_replay_tensors(
        [
            {"schedule": [1, 2], "shape": (3, 4)},
            {"schedule": [1, 2], "shape": (3, 4)},
        ],
        sample_counts=[1, 1],
    )

    assert gathered == {"schedule": [1, 2], "shape": (3, 4)}


def test_replay_gather_concatenates_explicit_ragged_sample_rows() -> None:
    gathered = gather_replay_tensors(
        [
            {"input_ids": SampleAlignedValues(([1, 2],))},
            {"input_ids": SampleAlignedValues(([3, 4, 5],))},
        ],
        sample_counts=[1, 1],
    )

    assert gathered["input_ids"] == ([1, 2], [3, 4, 5])


def test_replay_gather_validates_explicit_sample_row_count() -> None:
    with pytest.raises(ValueError, match="must match each chunk sample_count"):
        gather_replay_tensors(
            [{"input_ids": SampleAlignedValues(([1], [2]))}],
            sample_counts=[1],
        )


def test_cosmos3_keeps_prompt_ids_out_of_shared_chunk_context() -> None:
    def state(input_ids: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            guidance_scale=7.0,
            do_cfg=True,
            height=64,
            width=64,
            num_frames=5,
            fps=24,
            num_noisy_vision_tokens=8,
            cond_input_ids=input_ids,
            uncond_input_ids=[0],
            latents=torch.zeros(1, 1),
            vision_condition_mask=torch.zeros(1, 1, 1),
        )

    states = [state([1, 2]), state([3, 4, 5])]
    contexts = [Cosmos3Model.export_batch_context(object(), value) for value in states]
    replay = [Cosmos3Model.export_replay_tensors(object(), value) for value in states]

    assert require_matching_chunk_context(contexts) == contexts[0]
    gathered = gather_replay_tensors(replay, sample_counts=[1, 1])
    assert gathered["cond_input_ids"] == ([1, 2], [3, 4, 5])


def test_diffusion_chunk_gatherer_rejects_mixed_none_replay_values() -> None:
    request = _request(cfg=False)
    chunks = _diffusion_chunks({"model_family": "sd3_5"})
    chunks[0].replay_tensors = {"prompt_embeds": None}
    chunks[1].replay_tensors = {"prompt_embeds": torch.ones(1, 1)}

    with pytest.raises(ValueError, match="must be present on all results"):
        DiffusionChunkGatherer().gather_chunks(
            request,
            build_sample_rows(request),
            chunks,
        )


def test_diffusion_chunk_gatherer_rejects_mismatched_static_replay_values() -> None:
    request = _request(cfg=False)
    chunks = _diffusion_chunks({"model_family": "sd3_5"})
    chunks[0].replay_tensors = {"scheduler": "flow"}
    chunks[1].replay_tensors = {"scheduler": "ddim"}

    with pytest.raises(ValueError, match="non-batched replay value 'scheduler' must match"):
        DiffusionChunkGatherer().gather_chunks(
            request,
            build_sample_rows(request),
            chunks,
        )


def test_diffusion_chunk_gatherer_rejects_mismatched_replay_keys() -> None:
    request = _request(cfg=False)
    chunks = _diffusion_chunks({"model_family": "sd3_5"})
    chunks[0].replay_tensors = {"prompt_embeds": torch.ones(1, 1)}
    chunks[1].replay_tensors = {"pooled_prompt_embeds": torch.ones(1, 1)}

    with pytest.raises(ValueError, match="replay_tensors keys must match"):
        DiffusionChunkGatherer().gather_chunks(
            request,
            build_sample_rows(request),
            chunks,
        )


def test_diffusion_chunk_gatherer_rejects_mismatched_context() -> None:
    request = _request(cfg=False)
    chunks = _diffusion_chunks({"model_family": "sd3_5", "cfg": False})
    chunks[1].context = {"model_family": "sd3_5", "cfg": True}

    with pytest.raises(ValueError, match="chunk context at ordered index 1 does not match"):
        DiffusionChunkGatherer().gather_chunks(
            request,
            build_sample_rows(request),
            chunks,
        )


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
        inputs=["p0"],
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
