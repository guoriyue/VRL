"""LlamaGen wrapper contract tests (tiny config, no checkpoints)."""

from __future__ import annotations

import pytest
import torch

from tests.models.families.llamagen.fixtures import (
    TINY_BLOCK_SIZE,
    TINY_CAPTION_DIM,
    TINY_CLS_TOKEN_NUM,
    TINY_VOCAB_SIZE,
    build_tiny_llamagen_model,
)
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.interfaces import ReplayResult
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory import build_ar_discrete_trajectory, build_training_view


def _request(samples: int = 2) -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="llamagen",
        task="ar_t2i",
        inputs=["a photo of a cat"],
        samples_per_prompt=samples,
    )


def _sample_rows(count: int = 2) -> list[GenerationSampleRow]:
    request = _request(count)
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            prompt_id="p0",
            group_id="g0",
            sample_id=f"s{index}",
            trajectory_id=f"t{index}",
            seed=None,
        )
        for index in range(count)
    ]


def _rollout_batch(model) -> RolloutBatch:
    enc = model.t5_tokenizer(
        ["a photo of a cat", "dog"],
        max_length=TINY_CLS_TOKEN_NUM,
    )
    token_ids = torch.randint(0, TINY_VOCAB_SIZE, (2, TINY_BLOCK_SIZE))
    trajectory = build_ar_discrete_trajectory(
        request=_request(),
        sample_rows=_sample_rows(),
        token_ids=token_ids,
        token_log_probs=torch.zeros_like(token_ids, dtype=torch.float32),
        token_mask=torch.ones_like(token_ids, dtype=torch.float32),
        prompt_input_ids=enc["input_ids"],
        prompt_attention_mask=enc["attention_mask"],
        uncond_input_ids=torch.zeros_like(enc["input_ids"]),
        uncond_attention_mask=torch.zeros_like(enc["attention_mask"]),
        context={"model_family": "llamagen"},
    )
    return RolloutBatch(
        observations=torch.ones(2, 1, 3, dtype=torch.long),
        actions=token_ids,
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
        training_view=build_training_view(trajectory),
    )


def test_encode_caption_left_pads_like_upstream() -> None:
    """Valid features roll to the end; padded rows are zeroed; mask flips."""
    model = build_tiny_llamagen_model()
    enc = model.t5_tokenizer(["ab", "abc"], max_length=TINY_CLS_TOKEN_NUM)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    embs, flipped = model.encode_caption(ids, mask)

    assert embs.shape == (2, TINY_CLS_TOKEN_NUM, TINY_CAPTION_DIM)
    assert torch.equal(flipped, torch.flip(mask, dims=[-1]))
    raw = model.t5_encoder(input_ids=ids).last_hidden_state
    # Row 0 has 2 valid tokens: positions 0..1 are zero pad, 2..3 the valid features.
    assert torch.equal(embs[0, :2], torch.zeros(2, TINY_CAPTION_DIM))
    torch.testing.assert_close(embs[0, 2:], raw[0, :2])
    # Row 1 has 3 valid tokens.
    assert torch.equal(embs[1, :1], torch.zeros(1, TINY_CAPTION_DIM))
    torch.testing.assert_close(embs[1, 1:], raw[1, :3])


def test_uncond_caption_embeds_is_learned_null_caption() -> None:
    """CFG uncond branch == CaptionEmbedder's uncond_embedding prefix slice."""
    model = build_tiny_llamagen_model()
    uncond = model.uncond_caption_embeds(3)
    buffer = model._gpt_trunk().cls_embedding.uncond_embedding[:TINY_CLS_TOKEN_NUM]
    assert uncond.shape == (3, TINY_CLS_TOKEN_NUM, TINY_CAPTION_DIM)
    torch.testing.assert_close(uncond[0], buffer)
    torch.testing.assert_close(uncond[2], buffer)


def test_forward_image_logits_shape_and_grad() -> None:
    """Teacher-forced logits cover every image position over the image vocab."""
    model = build_tiny_llamagen_model()
    for p in model._gpt_trunk().output.parameters():
        p.requires_grad_(True)
    enc = model.t5_tokenizer(["ab", "abcd"], max_length=TINY_CLS_TOKEN_NUM)
    embs, mask = model.encode_caption(enc["input_ids"], enc["attention_mask"])
    token_ids = torch.randint(0, TINY_VOCAB_SIZE, (2, TINY_BLOCK_SIZE))

    logits = model.forward_image_logits(embs, mask, token_ids)

    assert logits.shape == (2, TINY_BLOCK_SIZE, TINY_VOCAB_SIZE)
    assert logits.requires_grad


def test_decode_image_tokens_shape_and_size_validation() -> None:
    """16 tokens -> 4x4 grid -> 64 px through the ds16 VQ decoder."""
    model = build_tiny_llamagen_model()
    token_ids = torch.randint(0, TINY_VOCAB_SIZE, (2, TINY_BLOCK_SIZE))

    images = model.decode_image_tokens(token_ids, image_size=64)

    assert images.shape == (2, 3, 64, 64)
    # decode_code got the live-quantizer codebook width, not a hardcode.
    assert model.vq.decode_shapes[-1] == [2, 8, 4, 4]
    with pytest.raises(ValueError, match="inconsistent"):
        model.decode_image_tokens(token_ids, image_size=256)


def test_replay_forward_returns_typed_replay_result() -> None:
    """Replay contract: {"logits", "image_token_ids"} for image_tokens."""
    model = build_tiny_llamagen_model()
    batch = _rollout_batch(model)

    result = model.replay_forward(batch)

    assert isinstance(result, ReplayResult)
    segment = result.segments["image_tokens"]
    assert segment.segment == "image_tokens"
    assert set(segment.values) == {"logits", "image_token_ids"}
    assert segment.values["logits"].shape == (2, TINY_BLOCK_SIZE, TINY_VOCAB_SIZE)
    assert torch.equal(segment.values["image_token_ids"], batch.actions)


def test_model_exposes_trainer_replay_methods() -> None:
    """RuntimeModel surface: replay_forward / disable_adapter / load_trainable_state."""
    model = build_tiny_llamagen_model()
    assert callable(model.replay_forward)
    assert callable(model.disable_adapter)
    assert callable(model.load_trainable_state)
    with model.disable_adapter():
        pass


def test_replay_model_has_no_rollout_decode_surface() -> None:
    """Replay wrapper loads no VQ decoder and stubs the decode surface."""
    model = build_tiny_llamagen_model(replay=True)
    batch = _rollout_batch(model)

    result = model.replay_forward(batch)
    assert set(result.segments["image_tokens"].values) == {"logits", "image_token_ids"}

    with pytest.raises(RuntimeError, match="cannot decode image tokens"):
        model.decode_image_tokens(torch.zeros(1, TINY_BLOCK_SIZE, dtype=torch.long))
    with pytest.raises(RuntimeError, match="VQ decoder"):
        _ = model.vq_model
