"""Tests for the token-step protocol."""

from __future__ import annotations

import pytest

from vrl.generation.steps.token import TokenLoopInit, TokenStepBatch


@pytest.mark.parametrize(
    ("row_count", "step_count", "message"),
    [
        (0, 1, "row_count must be a positive integer"),
        (1, 0, "step_count must be a positive integer"),
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
    with pytest.raises(ValueError, match="position must be non-negative"):
        TokenStepBatch(
            row_indices=[0],
            position=-1,
            row_lanes={},
        )
