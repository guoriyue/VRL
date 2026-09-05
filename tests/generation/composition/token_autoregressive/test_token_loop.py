"""Tests for the shared token-autoregressive loop."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.generation.composition.token_autoregressive.token_loop import (
    TokenAutoregressiveEnvelope,
    TokenAutoregressiveLoop,
)
from vrl.generation.steps.token import TokenLoopInit, TokenStepBatch, TokenStepOutput


class _DeterministicRunner:
    def __init__(self, *, row_count: int = 3, step_count: int = 2) -> None:
        self.row_count = row_count
        self.step_count = step_count
        self.state = {"step_calls": 0}
        self.step_inputs: list[tuple[list[int], int, list[float]]] = []

    def init_token(self) -> TokenLoopInit:
        hidden = torch.arange(1, self.row_count + 1, dtype=torch.float32).unsqueeze(-1) * 10
        return TokenLoopInit(
            state=self.state,
            row_count=self.row_count,
            step_count=self.step_count,
            row_lanes={"hidden": hidden},
        )

    def step_token(
        self,
        state: dict[str, Any],
        batch: TokenStepBatch,
    ) -> TokenStepOutput:
        assert state is self.state
        hidden = batch.row_lanes["hidden"]
        state["step_calls"] += 1
        self.step_inputs.append(
            (
                batch.row_indices,
                batch.position,
                [float(value) for value in hidden.flatten()],
            )
        )
        return TokenStepOutput(updated_row_lanes={"hidden": hidden + 1.0})

    def finalize_token(self, state: dict[str, Any]) -> dict[str, Any]:
        assert state is self.state
        return state


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (
            2,
            [
                ([0, 1], 0, [10.0, 20.0]),
                ([2], 0, [30.0]),
                ([0, 1], 1, [11.0, 21.0]),
                ([2], 1, [31.0]),
            ],
        ),
        (
            3,
            [
                ([0, 1, 2], 0, [10.0, 20.0, 30.0]),
                ([0, 1, 2], 1, [11.0, 21.0, 31.0]),
            ],
        ),
        (
            4,
            [
                ([0, 1, 2], 0, [10.0, 20.0, 30.0]),
                ([0, 1, 2], 1, [11.0, 21.0, 31.0]),
            ],
        ),
    ],
)
def test_loop_runs_position_major_bounded_row_batches(
    batch_size: int,
    expected: list[tuple[list[int], int, list[float]]],
) -> None:
    runner = _DeterministicRunner()

    result = TokenAutoregressiveLoop(
        runner=runner,
        scheduler_batch_size=batch_size,
        init_kwargs={"unsupported_init_kwarg": True},
    ).run()

    assert result is runner.state
    assert result["step_calls"] == len(expected)
    assert runner.step_inputs == expected


def test_loop_defaults_to_one_full_row_batch() -> None:
    runner = _DeterministicRunner(row_count=2, step_count=1)

    TokenAutoregressiveLoop(runner=runner).run()

    assert runner.step_inputs == [([0, 1], 0, [10.0, 20.0])]


def test_loop_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="scheduler_batch_size must be a positive integer"):
        TokenAutoregressiveLoop(
            runner=_DeterministicRunner(),
            scheduler_batch_size=0,
        ).run()


def test_envelope_rejects_init_without_row_lanes() -> None:
    init = TokenLoopInit(
        state=object(),
        row_count=1,
        step_count=1,
    )

    with pytest.raises(ValueError, match="row_lanes must be non-empty"):
        TokenAutoregressiveEnvelope.from_init(init)


def test_envelope_rejects_row_lane_count_mismatch() -> None:
    init = TokenLoopInit(
        state=object(),
        row_count=2,
        step_count=1,
        row_lanes={"hidden": torch.zeros(1, 1)},
    )

    with pytest.raises(ValueError, match="cannot split tensor with batch=1 into 2 rows"):
        TokenAutoregressiveEnvelope.from_init(init)


def test_loop_rejects_unknown_row_update() -> None:
    class _Runner(_DeterministicRunner):
        def step_token(
            self,
            state: dict[str, Any],
            batch: TokenStepBatch,
        ) -> TokenStepOutput:
            del state
            return TokenStepOutput(
                updated_row_lanes={
                    "unknown": torch.zeros(len(batch.row_indices), 1),
                },
            )

    with pytest.raises(KeyError, match="unknown AR row lane"):
        TokenAutoregressiveLoop(runner=_Runner()).run()


@pytest.mark.parametrize(
    ("row_indices", "error", "message"),
    [
        ([], ValueError, "at least one row index"),
        ([3], IndexError, "out of range"),
        ([0, 0], ValueError, "must be unique"),
    ],
)
def test_envelope_rejects_invalid_or_duplicate_rows(
    row_indices: list[int],
    error: type[Exception],
    message: str,
) -> None:
    envelope = TokenAutoregressiveEnvelope.from_init(_DeterministicRunner().init_token())

    with pytest.raises(error, match=message):
        envelope.build_step_batch(row_indices, position=0)


def test_envelope_rejects_unknown_row_update() -> None:
    envelope = TokenAutoregressiveEnvelope.from_init(_DeterministicRunner().init_token())
    batch = envelope.build_step_batch([0], position=0)

    with pytest.raises(KeyError, match="unknown AR row lane"):
        envelope.apply_step_output(
            batch,
            TokenStepOutput(updated_row_lanes={"unknown": torch.zeros(1, 1)}),
        )
