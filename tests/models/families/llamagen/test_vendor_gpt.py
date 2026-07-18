"""Vendored LlamaGen GPT semantic locks.

These pin the two upstream facts the RL integration depends on:

1. The rope table zeroes the caption-prefix positions (the reason the GPT
   cannot be mapped onto ``LlamaForCausalLM``), and is a 2D grid over the
   image positions.
2. The static-KV-cache decode path (rollout) and the naive full-sequence
   forward with an explicit mask (trainer replay) produce identical logits —
   the train/infer logprob-parity invariant at the vendor level.
"""

from __future__ import annotations

import torch

from tests.models.families.llamagen.fixtures import (
    TINY_CAPTION_DIM,
    TINY_CLS_TOKEN_NUM,
    TINY_VOCAB_SIZE,
    build_tiny_gpt,
)


def test_rope_table_zeroes_caption_prefix() -> None:
    """Caption positions carry cos=sin=0 (their rotated q/k vanish)."""
    gpt = build_tiny_gpt()
    prefix = gpt.freqs_cis[:TINY_CLS_TOKEN_NUM]
    image = gpt.freqs_cis[TINY_CLS_TOKEN_NUM:]
    assert torch.equal(prefix, torch.zeros_like(prefix))
    assert bool((image.abs().sum(dim=(1, 2)) > 0).all())
    # 2D grid rope: block_size image positions follow the prefix.
    assert image.shape[0] == gpt.block_size


def test_kv_decode_matches_teacher_forced_forward() -> None:
    """Static-cache rollout logits == explicit-mask replay logits."""
    torch.manual_seed(7)
    gpt = build_tiny_gpt()
    batch, caption_len, image_len = 2, TINY_CLS_TOKEN_NUM, 6
    caption_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    # Left-padded caption features with zeroed pad rows (upstream layout).
    features = torch.randn(batch, caption_len, TINY_CAPTION_DIM)
    features = features * caption_mask[:, :, None]
    tokens = torch.randint(0, TINY_VOCAB_SIZE, (batch, image_len))

    # --- rollout path: setup_caches + causal-mask patch (upstream generate())
    with torch.device("cpu"):
        gpt.setup_caches(
            max_batch_size=batch,
            max_seq_length=caption_len + image_len,
            dtype=torch.float32,
        )
    gpt.causal_mask[:, :, :caption_len] = gpt.causal_mask[
        :, :, :caption_len
    ] * caption_mask.unsqueeze(1)
    eye = torch.eye(gpt.causal_mask.size(1), gpt.causal_mask.size(2))
    gpt.causal_mask[:] = gpt.causal_mask * (1 - eye) + eye

    step_logits = []
    logits, _ = gpt(idx=None, cond_idx=features, input_pos=torch.arange(caption_len))
    step_logits.append(logits[:, -1])
    for i in range(image_len - 1):
        logits, _ = gpt(
            idx=tokens[:, i : i + 1],
            cond_idx=None,
            input_pos=torch.tensor([caption_len + i]),
        )
        step_logits.append(logits[:, -1])
    rollout = torch.stack(step_logits, dim=1)  # [B, L, V]

    # --- replay path: no cache, explicit rollout-equivalent mask
    for block in gpt.layers:
        block.attention.kv_cache = None
    total_len = caption_len + image_len - 1
    replay_mask = (
        torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))
        .unsqueeze(0)
        .expand(batch, -1, -1)
        .clone()
    )
    replay_mask[:, :, :caption_len] &= caption_mask[:, None, :].bool()
    replay_mask |= torch.eye(total_len, dtype=torch.bool)
    logits, _ = gpt(
        idx=tokens[:, :-1],
        cond_idx=features,
        input_pos=torch.arange(total_len),
        mask=replay_mask.unsqueeze(1),
    )
    replay = logits[:, caption_len - 1 : caption_len - 1 + image_len]

    torch.testing.assert_close(replay, rollout, atol=1e-4, rtol=1e-4)
