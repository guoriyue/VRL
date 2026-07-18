"""GLM-Image trainer-replay contract tests (teacher-forced, tiny real model)."""

from __future__ import annotations

import pytest
import torch

from tests.models.ar.glm_image.fixtures import (
    TINY_CODEBOOK,
    build_tiny_glm_image_model,
    build_tiny_glm_image_replay_model,
)
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.ar.glm_image.model import glm_image_token_num
from vrl.models.interfaces import ReplayResult
from vrl.models.utils import count_trainable_params
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory import build_ar_discrete_trajectory, build_training_view

# 128x192 target -> large 4x6 (24 tokens) + preview 13x19 (247 tokens).
HEIGHT, WIDTH = 128, 192
TOTAL = glm_image_token_num(HEIGHT, WIDTH)


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="glm_image",
        task="ar_t2i",
        inputs=["a red apple"],
        samples_per_prompt=2,
    )


def _sample_rows() -> list[GenerationSampleRow]:
    request = _request()
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
        for index in range(2)
    ]


def _discrete_batch(context: dict | None = None) -> RolloutBatch:
    torch.manual_seed(1)
    token_ids = torch.randint(0, TINY_CODEBOOK, (2, TOTAL))
    trajectory = build_ar_discrete_trajectory(
        request=_request(),
        sample_rows=_sample_rows(),
        token_ids=token_ids,
        token_log_probs=torch.zeros_like(token_ids, dtype=torch.float32),
        token_mask=torch.ones_like(token_ids, dtype=torch.float32),
        prompt_input_ids=torch.ones(2, 4, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 4, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 4, dtype=torch.long),
        uncond_attention_mask=torch.zeros(2, 4, dtype=torch.long),
        context=(
            context
            if context is not None
            else {
                "model_family": "glm_image",
                "image_height": HEIGHT,
                "image_width": WIDTH,
            }
        ),
    )
    return RolloutBatch(
        observations=torch.ones(2, 1, 4, dtype=torch.long),
        actions=token_ids,
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
        training_view=build_training_view(trajectory),
    )


def test_glm_image_model_exposes_trainer_replay_methods() -> None:
    model = build_tiny_glm_image_model()
    assert callable(model.replay_forward)
    assert callable(model.disable_adapter)
    assert callable(model.load_trainable_state)


def test_replay_forward_returns_codebook_logits() -> None:
    model = build_tiny_glm_image_model()
    batch = _discrete_batch()

    result = model.replay_forward(batch)

    assert isinstance(result, ReplayResult)
    segment = result.segments["image_tokens"]
    assert segment.segment == "image_tokens"
    assert set(segment.values) == {"logits", "image_token_ids"}
    logits = segment.values["logits"]
    # Restricted to the codebook prefix — the 24-wide vision vocab's
    # image_start/image_end/reserved columns never appear in replay logits.
    assert logits.shape == (2, TOTAL, TINY_CODEBOOK)
    assert torch.isfinite(logits).all()
    assert torch.equal(segment.values["image_token_ids"], batch.actions)


def test_replay_forward_requires_pixel_dims_in_context() -> None:
    model = build_tiny_glm_image_model()
    batch = _discrete_batch(context={"model_family": "glm_image"})
    with pytest.raises(RuntimeError, match="image_height/image_width"):
        model.replay_forward(batch)


def test_replay_forward_rejects_token_count_mismatch() -> None:
    model = build_tiny_glm_image_model()
    batch = _discrete_batch(
        context={
            "model_family": "glm_image",
            "image_height": 1024,
            "image_width": 1024,
        },
    )
    with pytest.raises(RuntimeError, match="does not match"):
        model.replay_forward(batch)


def test_replay_positions_track_per_row_valid_prompt_lengths() -> None:
    """Left-padded rows must get shifted decode positions, not padded ones.

    Same tokens + same valid prompt suffix but different left padding must
    produce identical logits (padding is attention-masked and the position
    schedule is offset by the VALID length only).
    """
    model = build_tiny_glm_image_model()
    torch.manual_seed(2)
    token_ids = torch.randint(0, TINY_CODEBOOK, (1, TOTAL))
    prompt = torch.tensor([[30, 31, 16]])
    mask = torch.tensor([[1, 1, 1]])
    padded_prompt = torch.tensor([[0, 0, 30, 31, 16]])
    padded_mask = torch.tensor([[0, 0, 1, 1, 1]])

    embed = model.language_model.get_input_embeddings()
    grids = model._replay_grids(_discrete_batch(), TOTAL)
    plain = model.forward_image_logits(
        embed(prompt), mask, token_ids, grids=grids,
    )
    padded = model.forward_image_logits(
        embed(padded_prompt), padded_mask, token_ids, grids=grids,
    )
    assert torch.allclose(plain, padded, atol=1e-4)


def test_replay_model_replays_without_vision_tower_or_decode_stack() -> None:
    model = build_tiny_glm_image_replay_model()
    batch = _discrete_batch()

    result = model.replay_forward(batch)

    assert result.segments["image_tokens"].values["logits"].shape == (
        2, TOTAL, TINY_CODEBOOK,
    )
    with pytest.raises(RuntimeError, match="cannot decode image tokens"):
        model.decode_image_tokens(
            batch.actions, height=HEIGHT, width=WIDTH, prompts=["a", "b"],
        )
    with pytest.raises(RuntimeError, match="GlmImageProcessor"):
        _ = model.processor


def test_lora_wrap_keeps_replay_and_adapter_surfaces_working() -> None:
    model = build_tiny_glm_image_model(use_lora=True)

    assert count_trainable_params(model) > 0

    batch = _discrete_batch()
    logits = model.replay_forward(batch).segments["image_tokens"].values["logits"]
    assert logits.shape == (2, TOTAL, TINY_CODEBOOK)

    with model.disable_adapter():
        ref_logits = (
            model.replay_forward(batch).segments["image_tokens"].values["logits"]
        )
    assert ref_logits.shape == logits.shape


def test_disable_adapter_without_lora_is_noop() -> None:
    model = build_tiny_glm_image_model()
    with model.disable_adapter():
        pass
