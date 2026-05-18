"""Tests for the shared AR decode loop driver."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.ar.decode_loop import (
    ARDecodeLoop,
    ARStepBatch,
    ARStepOutput,
    ARStepResult,
    ARTokenLoopInit,
)
from vrl.generation.types import GenerationRequest, GenerationSampleRow


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        prompts=["p0"],
        samples_per_prompt=3,
    )


def _sample_row(index: int) -> GenerationSampleRow:
    return GenerationSampleRow(
        prompt_index=0,
        sample_index=index,
        prompt="p0",
        prompt_id="prompt-0",
        group_id="group-0",
        sample_id=f"sample-{index}",
        trajectory_id=f"traj-{index}",
        seed=None,
        metadata={"existing": index},
    )


class _DeterministicARContractRunner:
    def __init__(self) -> None:
        self.step_inputs: list[tuple[list[int], int, list[float], list[float]]] = []

    def init_ar(self) -> ARTokenLoopInit:
        return ARTokenLoopInit(
            state={},
            cache_lanes={"kv": torch.tensor([[0.0], [1.0], [2.0]])},
            row_lanes={"hidden": torch.tensor([[10.0], [20.0], [30.0]])},
        )

    def step_ar(
        self,
        state: dict[str, Any],
        batch: ARStepBatch,
    ) -> ARStepOutput:
        del state
        kv = batch.cache_lanes["kv"]
        hidden = batch.row_lanes["hidden"]
        self.step_inputs.append(
            (
                batch.row_indices,
                batch.position,
                [float(value) for value in kv.flatten()],
                [float(value) for value in hidden.flatten()],
            )
        )
        return ARStepOutput(
            result=ARStepResult(
                sequence_ids=batch.sequence_ids,
                positions=batch.positions,
                token=[0 for _ in batch.row_indices],
                log_prob=[0.0 for _ in batch.row_indices],
            ),
            updated_cache_lanes={"kv": kv + 10.0},
            updated_row_lanes={"hidden": hidden + 1.0},
        )

    def finalize_ar(self, state: dict[str, Any]) -> dict[str, Any]:
        return state


def test_ar_decode_loop_schedules_contract_cache_lanes() -> None:
    runner = _DeterministicARContractRunner()

    result = ARDecodeLoop(
        request=_request(),
        sample_rows=[_sample_row(index) for index in range(3)],
        runner=runner,
        max_new_tokens=2,
        tokenizer_key="janus_pro",
        dtype="float32",
        scheduler_batch_size=2,
    ).run()

    assert runner.step_inputs == [
        ([0, 1], 0, [0.0, 1.0], [10.0, 20.0]),
        ([2], 0, [2.0], [30.0]),
        ([0, 1], 1, [10.0, 11.0], [11.0, 21.0]),
        ([2], 1, [12.0], [31.0]),
    ]
    assert result.scheduler_batches == 4
    assert result.engine_counters["ar_scheduled_decode_loop_enabled"] is True
    assert result.engine_counters["ar_scheduler_batches"] == 4


def test_ar_decode_loop_requires_family_hooks() -> None:
    try:
        ARDecodeLoop(
            request=_request(),
            sample_rows=[_sample_row(0)],
            runner=object(),
            max_new_tokens=1,
            tokenizer_key="janus_pro",
            dtype="float32",
            scheduler_batch_size=None,
        ).run()
    except TypeError as exc:
        assert "init_ar" in str(exc)
        assert "step_ar" in str(exc)
        assert "finalize_ar" in str(exc)
    else:
        raise AssertionError("ARDecodeLoop should reject models without AR hooks")
