"""Tests for the AR paged-attention backend contract."""

from __future__ import annotations

import pytest
import torch

from vrl.nn.layers.attention.paged import (
    ARAttentionConfig,
    ARAttentionPrefillInput,
    ARAttentionPrefillOutput,
    ARAttentionStepInput,
    ARAttentionStepOutput,
)


def test_paged_attention_config_requires_identity() -> None:
    """Checks paged attention config requires identity."""
    with pytest.raises(ValueError, match="family"):
        ARAttentionConfig(family="")
    with pytest.raises(ValueError, match="block_size"):
        ARAttentionConfig(family="janus_pro", block_size=0)


def test_paged_attention_prefill_validates_batch_shape() -> None:
    """Checks paged attention prefill validates batch shape."""
    with pytest.raises(ValueError, match="batch sizes"):
        ARAttentionPrefillInput(
            inputs_embeds=torch.zeros(2, 3, 4),
            attention_mask=torch.ones(1, 3),
            branch="cond",
        )


def test_paged_attention_step_validates_state_shape() -> None:
    """Checks paged attention step validates state shape."""
    with pytest.raises(ValueError, match="sequence_states"):
        ARAttentionStepInput(
            input_embeds=torch.zeros(2, 1, 4),
            attention_mask=torch.ones(2, 4),
            sequence_states=("row-0",),
        )


def test_paged_attention_outputs_accept_last_hidden_rank_two_or_three() -> None:
    """Checks paged attention outputs accept last hidden rank two or three."""
    ARAttentionPrefillOutput(
        last_hidden=torch.zeros(2, 4),
        sequence_states=("a", "b"),
    )
    ARAttentionStepOutput(
        last_hidden=torch.zeros(2, 1, 4),
        sequence_states=("a", "b"),
    )
