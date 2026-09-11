"""Tests for the token-step protocol."""

from __future__ import annotations

import pytest

from vrl.generation.steps.token import TokenLoopInit, TokenStepBatch


@pytest.mark.parametrize(
    ("row_count", "step_count", "message"),
    [
        (0, 1, r"TokenLoopInit\.row_count must be >= 1"),
        (1, 0, r"TokenLoopInit\.step_count must be >= 1"),
    ],
)
def test_token_loop_init_rejects_invalid_shape(
    row_count: int | bool,
    step_count: int | bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TokenLoopInit(
            state=object(),
            row_count=row_count,
            step_count=step_count,
        )


@pytest.mark.parametrize(
    ("row_indices", "message"),
    [
        ([], "must be non-empty"),
        ([-1], "must be non-negative"),
        ([0.9], "must be integers"),
        ([True], "must be integers"),
        (["0"], "must be integers"),
        ([0, 0], "must be unique"),
    ],
)
def test_token_step_batch_rejects_invalid_row_indices(
    row_indices: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TokenStepBatch(
            row_indices=row_indices,
            position=0,
            row_lanes={},
        )


def test_token_step_batch_rejects_negative_position() -> None:
    with pytest.raises(ValueError, match=r"TokenStepBatch\.position must be >= 0"):
        TokenStepBatch(
            row_indices=[0],
            position=-1,
            row_lanes={},
        )


@pytest.mark.parametrize("position", [0.9, True, "0"])
def test_token_step_batch_rejects_non_integer_position(position) -> None:
    with pytest.raises(ValueError, match="position must be an integer"):
        TokenStepBatch(row_indices=[0], position=position, row_lanes={})
