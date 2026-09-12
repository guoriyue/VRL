"""Tests for pure batch gatherers."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionBatchGatherer,
    DiffusionBatchResult,
)
from vrl.generation.bindings.token_autoregressive.layout import ARRequestLayout
from vrl.generation.execution.executor_base import BatchExecutorBase
from vrl.generation.execution.sample_batches import (
    GenerationSampleBatch,
    SampleAlignedValues,
    gather_batch_context,
    gather_replay_tensors,
)
from vrl.generation.types import GenerationRequest
from vrl.models.families.cosmos.cosmos3.model import Cosmos3Model


class _PureGatherer:
    def merge_generation_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[Any],
        batches: Sequence[Any],
    ) -> Any:
        del request, sample_rows
        return SimpleNamespace(output=list(batches))


class _Executor(BatchExecutorBase):
    family = "sd3_5"
    task = "t2i"

    def forward_batch(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_chunk_executor_uses_injected_gatherer() -> None:
    request = _request()
    sample_rows = request.sample_rows()
    gatherer = _PureGatherer()
    executor = _Executor(gatherer=gatherer)

    output = executor.merge_generation_batches(request, sample_rows, ["batch"])

    assert output.output == ["batch"]
    assert executor._gatherer is gatherer


def test_chunk_executor_rejects_request_execution_without_gatherer() -> None:
    request = _request()

    with pytest.raises(RuntimeError, match="requires an injected batch gatherer"):
        _Executor().merge_generation_batches(request, request.sample_rows(), ["batch"])


def test_diffusion_chunk_gatherer_gathers_without_model_object() -> None:
    """Checks diffusion batch gatherer gathers without model object."""
    request = _request(cfg=False)
    sample_rows = request.sample_rows()
    gatherer = DiffusionBatchGatherer()
    context = {
        "guidance_scale": 4.5,
        "cfg": False,
        "model_family": "sd3_5",
    }

    output = gatherer.merge_generation_batches(request, sample_rows, _diffusion_batches(context))

    assert output.output.device.type == "cpu"
    assert output.trajectory is not None
    assert output.runtime_debug is None
    assert output.trajectory.segments["denoise"].distribution == "flow_matching"
    assert output.trajectory.axes["sample"].length == 2
    assert output.trajectory.axes["denoise"].length == 2
    assert torch.equal(output.output[:, 0, 0, 0], torch.tensor([1.0, 2.0]))


def test_diffusion_chunk_gatherer_orders_prompt_major_chunks() -> None:
    """Checks diffusion batch gatherer orders prompt major batches."""
    request = _request(cfg=False)
    sample_rows = request.sample_rows()
    gatherer = DiffusionBatchGatherer()
    context = {
        "guidance_scale": 4.5,
        "cfg": False,
        "model_family": "sd3_5",
    }

    output = gatherer.merge_generation_batches(
        request,
        sample_rows,
        list(reversed(_diffusion_batches(context))),
    )

    assert torch.equal(output.output[:, 0, 0, 0], torch.tensor([1.0, 2.0]))


def test_diffusion_chunk_gatherer_keeps_rollout_context() -> None:
    """Checks diffusion batch gatherer keeps rollout context."""
    request = _request(family="cosmos", task="v2w", cfg=False)
    sample_rows = request.sample_rows()
    gatherer = DiffusionBatchGatherer()
    context = {
        "guidance_scale": 4.5,
        "cfg": True,
        "model_family": "cosmos",
    }

    output = gatherer.merge_generation_batches(request, sample_rows, _diffusion_batches(context))

    assert output.trajectory is not None
    assert output.trajectory.context == context
    assert output.trajectory.segments["denoise"].reward_view == "video"


def test_diffusion_chunk_gatherer_strictly_merges_replay_values() -> None:
    request = _request(cfg=False)
    batches = _diffusion_batches(
        {
            "guidance_scale": 4.5,
            "cfg": False,
            "model_family": "sd3_5",
        },
    )
    batches[0].replay_tensors = {
        "prompt_embeds": torch.tensor([[1.0]]),
        "optional": None,
        "scheduler": "flow",
    }
    batches[1].replay_tensors = {
        "prompt_embeds": torch.tensor([[2.0]]),
        "optional": None,
        "scheduler": "flow",
    }

    output = DiffusionBatchGatherer().merge_generation_batches(
        request,
        request.sample_rows(),
        batches,
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
    with pytest.raises(ValueError, match="must match each batch sample_count"):
        gather_replay_tensors(
            [{"input_ids": SampleAlignedValues(([1], [2]))}],
            sample_counts=[1],
        )


def test_cosmos3_keeps_prompt_ids_out_of_shared_batch_context() -> None:
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

    assert gather_batch_context(contexts) == contexts[0]
    gathered = gather_replay_tensors(replay, sample_counts=[1, 1])
    assert gathered["cond_input_ids"] == ([1, 2], [3, 4, 5])


def test_diffusion_chunk_gatherer_rejects_mixed_none_replay_values() -> None:
    request = _request(cfg=False)
    batches = _diffusion_batches({"model_family": "sd3_5"})
    batches[0].replay_tensors = {"prompt_embeds": None}
    batches[1].replay_tensors = {"prompt_embeds": torch.ones(1, 1)}

    with pytest.raises(ValueError, match="must be present on all results"):
        DiffusionBatchGatherer().merge_generation_batches(
            request,
            request.sample_rows(),
            batches,
        )


def test_diffusion_chunk_gatherer_rejects_mismatched_static_replay_values() -> None:
    request = _request(cfg=False)
    batches = _diffusion_batches({"model_family": "sd3_5"})
    batches[0].replay_tensors = {"scheduler": "flow"}
    batches[1].replay_tensors = {"scheduler": "ddim"}

    with pytest.raises(ValueError, match="non-batched replay value 'scheduler' must match"):
        DiffusionBatchGatherer().merge_generation_batches(
            request,
            request.sample_rows(),
            batches,
        )


def test_diffusion_chunk_gatherer_rejects_mismatched_replay_keys() -> None:
    request = _request(cfg=False)
    batches = _diffusion_batches({"model_family": "sd3_5"})
    batches[0].replay_tensors = {"prompt_embeds": torch.ones(1, 1)}
    batches[1].replay_tensors = {"pooled_prompt_embeds": torch.ones(1, 1)}

    with pytest.raises(ValueError, match="replay_tensors keys must match"):
        DiffusionBatchGatherer().merge_generation_batches(
            request,
            request.sample_rows(),
            batches,
        )


def test_diffusion_chunk_gatherer_rejects_mismatched_context() -> None:
    request = _request(cfg=False)
    batches = _diffusion_batches({"model_family": "sd3_5", "cfg": False})
    batches[1].context = {"model_family": "sd3_5", "cfg": True}

    with pytest.raises(ValueError, match="batch context at ordered index 1 does not match"):
        DiffusionBatchGatherer().merge_generation_batches(
            request,
            request.sample_rows(),
            batches,
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


def _diffusion_batches(context: dict[str, Any]) -> list[DiffusionBatchResult]:
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
) -> DiffusionBatchResult:
    return DiffusionBatchResult(
        batch=GenerationSampleBatch(
            prompt_index=0,
            sample_start=sample_start,
            sample_count=1,
        ),
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


def test_replay_gather_rejects_per_batch_misalignment_even_when_total_matches():
    with pytest.raises(ValueError, match=r"replay_tensors\.prompt_embeds has 1 rows, expected 2"):
        gather_replay_tensors(
            [{"prompt_embeds": torch.ones(1, 4)}, {"prompt_embeds": torch.ones(3, 4)}],
            sample_counts=[2, 2],
        )


def test_replay_gather_rejects_scalar_tensor_without_sample_axis():
    with pytest.raises(ValueError, match="leading batch dimension"):
        gather_replay_tensors([{"scale": torch.tensor(1.0)}], sample_counts=[1])


@pytest.mark.parametrize("sample_count", [True, 1.0, 1.5, "1", 0, -1])
@pytest.mark.parametrize("replay", [{}, {"schedule": [1, 2]}, {"tokens": torch.ones(1, 2)}])
def test_replay_gather_requires_integer_counts_for_every_payload(sample_count, replay):
    with pytest.raises(ValueError, match=r"sample_counts\[0\]"):
        gather_replay_tensors([replay], sample_counts=[sample_count])


@pytest.mark.parametrize("dtype", [torch.int64, torch.bfloat16, torch.float32])
def test_replay_gather_preserves_tensor_dtype_and_values(dtype) -> None:
    first = torch.tensor([[16777217]], dtype=dtype)
    second = torch.zeros(1, 1, dtype=dtype)
    merged = gather_replay_tensors([{"value": first}, {"value": second}], sample_counts=[1, 1])
    assert merged["value"].dtype == dtype
    assert torch.equal(merged["value"][:1], first)
    assert torch.equal(merged["value"][1:], second)


def test_replay_gather_rejects_dtype_promotion_that_changes_integer_values() -> None:
    with pytest.raises(ValueError, match=r"replay_tensors.ids.*index 1.*dtypes must match"):
        gather_replay_tensors(
            [{"ids": torch.tensor([[16777217]])}, {"ids": torch.zeros(1, 1)}],
            sample_counts=[1, 1],
        )


@pytest.mark.parametrize(
    "field", ["observations", "actions", "log_probs", "timesteps", "kl", "video"]
)
def test_diffusion_gather_rejects_mixed_field_dtypes(field) -> None:
    request = _request(cfg=False)
    batches = _diffusion_batches({"model_family": "sd3_5"})
    setattr(batches[1], field, getattr(batches[1], field).double())
    with pytest.raises(ValueError, match=rf"{field!r}.*index 1.*dtypes must match"):
        DiffusionBatchGatherer().merge_generation_batches(request, request.sample_rows(), batches)


def test_ar_field_gather_rejects_lossy_dtype_promotion() -> None:
    batches = [
        SimpleNamespace(token_ids=torch.tensor([[2**53 + 1]], dtype=torch.int64)),
        SimpleNamespace(token_ids=torch.tensor([[1.0]], dtype=torch.float64)),
    ]
    with pytest.raises(ValueError, match=r"token_ids.*dtypes must match"):
        ARRequestLayout().cat_batch_fields(batches, ("token_ids",))
