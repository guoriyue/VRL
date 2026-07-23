"""Tests for the token-step protocol."""

from __future__ import annotations

import pytest

from vrl.generation.steps.token import TokenStepBatch


@pytest.mark.parametrize("position", [0, 3])
def test_token_step_batch_accepts_valid_position(position: int) -> None:
    batch = TokenStepBatch(
        row_indices=[0, 2],
        position=position,
        row_lanes={},
    )

    assert batch.row_indices == [0, 2]
    assert batch.position == position


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
