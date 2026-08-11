"""CPU contracts for request-local token-step batching policy."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest
import torch

from vrl.generation.bindings.token_autoregressive.executor import (
    ARChunkInputs,
    ARDiscreteChunkExecutorBase,
)
from vrl.generation.bindings.token_autoregressive.layout import (
    ARRequestLayout,
    ARSamplingParams,
)
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.families.emu3.runtime import Emu3ChunkExecutor
from vrl.models.families.glm_image.runtime import GlmImageChunkExecutor
from vrl.models.families.janus_pro.runtime import JanusProChunkExecutor


class _Model:
    @staticmethod
    def decode_image_tokens(
        token_ids: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        return torch.zeros(token_ids.shape[0], 3, 2, 2)


class _Executor(ARDiscreteChunkExecutorBase):
    family = "test_ar"
    task = "ar_t2i"

    def __init__(self) -> None:
        self.model = _Model()
        self.prepare_calls = 0

    def _ar_runner(self, request: GenerationRequest) -> object:
        del request
        return object()

    def prepare_chunk_inputs(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> ARChunkInputs:
        del request
        self.prepare_calls += 1
        rows = chunk.sample_count
        ids = torch.zeros(rows, 1, dtype=torch.long)
        mask = torch.ones_like(ids)
        return ARChunkInputs(
            init_args=(torch.zeros(rows, 1),),
            init_kwargs={},
            image_decode_kwargs={},
            prompt_input_ids=ids,
            prompt_attention_mask=mask,
            uncond_input_ids=ids,
            uncond_attention_mask=mask,
            context={},
        )


_MISSING = object()


def _request(batch_size: object = _MISSING) -> GenerationRequest:
    sampling = {}
    if batch_size is not _MISSING:
        sampling["ar_scheduler_batch_size"] = batch_size
    return GenerationRequest(
        request_id="req",
        family="test_ar",
        task="ar_t2i",
        inputs=["prompt"],
        samples_per_prompt=3,
        sampling=sampling,
    )


def _chunk() -> SampleChunk:
    return SampleChunk(
        prompt_index=0,
        sample_start=0,
        sample_count=3,
    )


def test_scheduler_policy_is_not_duplicated_in_sampling_params() -> None:
    names = {field.name for field in fields(ARSamplingParams)}

    assert "use_ar_scheduler" not in names
    assert "ar_scheduler_batch_size" not in names


@pytest.mark.parametrize("batch_size", [None, 1, 3, 5])
def test_shared_discrete_executor_passes_scheduler_policy_to_loop(
    batch_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int | None] = []

    class _FakeLoop:
        def __init__(
            self,
            *,
            runner: object,
            scheduler_batch_size: int | None,
            init_args: tuple[torch.Tensor, ...],
            init_kwargs: dict[str, Any],
        ) -> None:
            del runner, init_kwargs
            seen.append(scheduler_batch_size)
            self.row_count = init_args[0].shape[0]

        def run(self) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.zeros(self.row_count, 2, dtype=torch.long),
                torch.zeros(self.row_count, 2),
            )

    monkeypatch.setattr(
        "vrl.generation.composition.token_autoregressive.token_loop.TokenAutoregressiveLoop",
        _FakeLoop,
    )
    executor = _Executor()

    result = executor.forward_chunk_plan(_request(batch_size), _chunk())

    assert seen == [batch_size]
    assert result.chunk.sample_count == 3
    assert executor.prepare_calls == 1


@pytest.mark.parametrize("batch_size", [True, False, 0, -1, 1.0, 1.5, "2"])
def test_request_boundary_rejects_coercible_or_non_positive_batch_size(
    batch_size: object,
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer or null"):
        ARRequestLayout().resolve_scheduler_batch_size(_request(batch_size))


def test_invalid_scheduler_policy_fails_before_family_preparation() -> None:
    executor = _Executor()

    with pytest.raises(ValueError, match="must be a positive integer or null"):
        executor.forward_chunk_plan(_request(0), _chunk())

    assert executor.prepare_calls == 0


@pytest.mark.parametrize(
    "executor_cls",
    [JanusProChunkExecutor, Emu3ChunkExecutor, GlmImageChunkExecutor],
)
def test_discrete_families_use_scheduler_consuming_shared_executor(
    executor_cls: type[ARDiscreteChunkExecutorBase],
) -> None:
    assert executor_cls.forward_chunk_plan is ARDiscreteChunkExecutorBase.forward_chunk_plan
