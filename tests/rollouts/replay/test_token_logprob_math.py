"""Tests for memory-bounded AR categorical log-prob helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vrl.math.token.logprob import gather_categorical_log_probs


def test_gather_categorical_log_probs_matches_full_log_softmax() -> None:
    """Chunked gathering over token positions (``chunk_size=3``) matches a full bf16-to-fp32
    ``log_softmax`` plus gather.
    """
    logits = torch.randn(2, 5, 11, dtype=torch.bfloat16)
    token_ids = torch.tensor(
        [
            [0, 3, 6, 9, 10],
            [10, 8, 6, 4, 2],
        ],
        dtype=torch.long,
    )

    actual = gather_categorical_log_probs(logits, token_ids, chunk_size=3)
    expected = (
        F.log_softmax(logits.float(), dim=-1)
        .gather(
            -1,
            token_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected)


def test_gather_categorical_log_probs_applies_temperature() -> None:
    """Checks temperature scales logits before normalization (policy contract)."""
    logits = torch.randn(2, 5, 11)
    token_ids = torch.randint(0, 11, (2, 5))

    actual = gather_categorical_log_probs(logits, token_ids, temperature=0.7)
    expected = (
        F.log_softmax(logits.float() / 0.7, dim=-1)
        .gather(
            -1,
            token_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("temperature", [0.0, float("nan")])
def test_gather_categorical_log_probs_rejects_nonpositive_temperature(
    temperature: float,
) -> None:
    """Greedy decoding is not represented as a near-zero categorical policy."""
    logits = torch.randn(2, 4, 7)
    token_ids = torch.randint(0, 7, (2, 4))

    with pytest.raises(ValueError, match="temperature must be finite and > 0"):
        gather_categorical_log_probs(logits, token_ids, temperature=temperature)


def test_gather_categorical_log_probs_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="leading shape"):
        gather_categorical_log_probs(
            torch.zeros(2, 3, 5),
            torch.zeros(2, 4, dtype=torch.long),
        )


def test_categorical_log_probs_accept_int32_token_ids():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    result = gather_categorical_log_probs(logits, torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(result, torch.log_softmax(logits, dim=-1)[:, 1])


@pytest.mark.parametrize("chunk_size", [-1, 0, True, 1.5, "2"])
def test_categorical_log_probs_reject_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        gather_categorical_log_probs(torch.ones(1, 3), torch.tensor([1]), chunk_size=chunk_size)
