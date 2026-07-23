"""Tests for the shared AR decode loop driver."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.composition.token_autoregressive.token_loop import TokenAutoregressiveLoop
from vrl.generation.steps.token import TokenLoopInit, TokenStepBatch, TokenStepOutput
from vrl.generation.types import GenerationRequest, GenerationSampleRow


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["p0"],
        samples_per_prompt=3,
    )


def _sample_row(index: int) -> GenerationSampleRow:
    return GenerationSampleRow(
        prompt_index=0,
        sample_index=index,
        prompt="p0",
        group_id="group-0",
        sample_id=f"sample-{index}",
        trajectory_id=f"traj-{index}",
        seed=None,
        metadata={"existing": index},
    )


class _DeterministicARContractRunner:
    def __init__(self) -> None:
        self.step_inputs: list[tuple[list[int], int, list[float], list[float]]] = []

    def init_token(self) -> TokenLoopInit:
        return TokenLoopInit(
            state={},
            cache_lanes={"kv": torch.tensor([[0.0], [1.0], [2.0]])},
            row_lanes={"hidden": torch.tensor([[10.0], [20.0], [30.0]])},
        )

    def step_token(
        self,
        state: dict[str, Any],
        batch: TokenStepBatch,
    ) -> TokenStepOutput:
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
        return TokenStepOutput(
            updated_cache_lanes={"kv": kv + 10.0},
            updated_row_lanes={"hidden": hidden + 1.0},
        )

    def finalize_token(self, state: dict[str, Any]) -> dict[str, Any]:
        return state


def test_ar_decode_loop_schedules_contract_cache_lanes() -> None:
    """Checks AR decode loop schedules contract cache lanes."""
    runner = _DeterministicARContractRunner()

    result = TokenAutoregressiveLoop(
        request=_request(),
        sample_rows=[_sample_row(index) for index in range(3)],
        runner=runner,
        max_new_tokens=2,
        tokenizer_key="janus_pro",
        dtype="float32",
        scheduler_batch_size=2,
    ).run()

    assert result.finalized == {}
    assert runner.step_inputs == [
        ([0, 1], 0, [0.0, 1.0], [10.0, 20.0]),
        ([2], 0, [2.0], [30.0]),
        ([0, 1], 1, [10.0, 11.0], [11.0, 21.0]),
        ([2], 1, [12.0], [31.0]),
    ]


def test_ar_decode_loop_requires_family_hooks() -> None:
    """Checks AR decode loop requires family hooks."""
    try:
        TokenAutoregressiveLoop(
            request=_request(),
            sample_rows=[_sample_row(0)],
            runner=object(),
            max_new_tokens=1,
            tokenizer_key="janus_pro",
            dtype="float32",
            scheduler_batch_size=None,
        ).run()
    except TypeError as exc:
        assert "init_token" in str(exc)
        assert "step_token" in str(exc)
        assert "finalize_token" in str(exc)
    else:
        raise AssertionError("TokenAutoregressiveLoop should reject models without AR hooks")
