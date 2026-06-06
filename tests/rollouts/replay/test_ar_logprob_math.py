"""Tests for memory-bounded AR categorical log-prob helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vrl.math.ar.logprob import gather_categorical_log_probs


def test_gather_categorical_log_probs_matches_full_log_softmax() -> None:
    """Checks gather categorical log-probs matches full log softmax."""
    logits = torch.randn(2, 5, 11, dtype=torch.bfloat16)
    token_ids = torch.tensor(
        [
            [0, 3, 6, 9, 10],
            [10, 8, 6, 4, 2],
        ],
        dtype=torch.long,
    )

    actual = gather_categorical_log_probs(logits, token_ids, chunk_size=3)
    expected = F.log_softmax(logits.float(), dim=-1).gather(
        -1,
        token_ids.unsqueeze(-1),
    ).squeeze(-1)

    assert torch.allclose(actual, expected)


def test_gather_categorical_log_probs_rejects_shape_mismatch() -> None:
    """Checks gather categorical log-probs rejects shape mismatch."""
    with pytest.raises(ValueError, match="leading shape"):
        gather_categorical_log_probs(
            torch.zeros(2, 3, 5),
            torch.zeros(2, 4, dtype=torch.long),
        )
