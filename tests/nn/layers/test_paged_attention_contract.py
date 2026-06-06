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
    ARPrefixCacheKey,
    ARPrefixCachePolicy,
)


def test_paged_attention_config_requires_identity() -> None:
    with pytest.raises(ValueError, match="family"):
        ARAttentionConfig(family="", model_key="janus")
    with pytest.raises(ValueError, match="block_size"):
        ARAttentionConfig(family="janus_pro", model_key="janus", block_size=0)


def test_paged_attention_prefill_validates_batch_shape() -> None:
    with pytest.raises(ValueError, match="batch sizes"):
        ARAttentionPrefillInput(
            inputs_embeds=torch.zeros(2, 3, 4),
            attention_mask=torch.ones(1, 3),
            branch="cond",
        )


def test_paged_attention_step_validates_state_shape() -> None:
    with pytest.raises(ValueError, match="sequence_states"):
        ARAttentionStepInput(
            input_embeds=torch.zeros(2, 1, 4),
            attention_mask=torch.ones(2, 4),
            sequence_states=("row-0",),
            branch_names=("cond", "cond"),
            position=0,
        )


def test_paged_attention_outputs_accept_last_hidden_rank_two_or_three() -> None:
    ARAttentionPrefillOutput(
        last_hidden=torch.zeros(2, 4),
        sequence_states=("a", "b"),
    )
    ARAttentionStepOutput(
        last_hidden=torch.zeros(2, 1, 4),
        sequence_states=("a", "b"),
    )


def test_prefix_cache_policy_requires_policy_version_match() -> None:
    key = _prefix_key(policy_version=1)
    assert ARPrefixCachePolicy().can_reuse(key, _prefix_key(policy_version=1))
    assert not ARPrefixCachePolicy().can_reuse(key, _prefix_key(policy_version=2))
    assert not ARPrefixCachePolicy(enabled=False).can_reuse(
        key,
        _prefix_key(policy_version=1),
    )


def _prefix_key(*, policy_version: int) -> ARPrefixCacheKey:
    return ARPrefixCacheKey.from_prompt_tokens(
        family="janus_pro",
        task="ar_t2i",
        policy_version=policy_version,
        tokenizer_key="janus",
        prompt_token_ids=[1, 2, 3],
        model_dtype="float16",
        cache_dtype="float16",
        cfg_branch_kind="cond",
        cache_layout_version="vllm.v1",
        paged_attention_config_hash="config",
    )
