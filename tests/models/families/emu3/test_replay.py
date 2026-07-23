"""Emu3 trainer-replay contract tests (teacher-forced, tiny real model)."""

from __future__ import annotations

import pytest
import torch

from tests.models.families.emu3.fixtures import (
    TINY_EOF_IDX,
    TINY_EOI_IDX,
    TINY_EOL_IDX,
    TINY_EOS_IDX,
    TINY_GEN_VOCAB,
    TINY_IMAGE_VOCAB,
    build_tiny_emu3_model,
    build_tiny_emu3_replay_model,
)
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.families.emu3.model import emu3_grid_token_num
from vrl.models.interfaces import ReplayResult
from vrl.models.utils import count_trainable_params
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory import build_ar_discrete_trajectory

HEIGHT, WIDTH = 2, 3
TOTAL = emu3_grid_token_num(HEIGHT, WIDTH)  # 11
FORCED = {3: TINY_EOL_IDX, 7: TINY_EOL_IDX, 8: TINY_EOF_IDX, 9: TINY_EOI_IDX, 10: TINY_EOS_IDX}


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="emu3",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
    )


def _sample_rows() -> list[GenerationSampleRow]:
    request = _request()
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            group_id="g0",
            sample_id=f"s{index}",
            trajectory_id=f"t{index}",
            seed=None,
        )
        for index in range(2)
    ]


def _schedule_conformant_token_ids() -> torch.Tensor:
    torch.manual_seed(1)
    token_ids = torch.randint(0, TINY_IMAGE_VOCAB, (2, TOTAL))
    for position, forced_idx in FORCED.items():
        token_ids[:, position] = forced_idx
    return token_ids


def _discrete_batch(context: dict | None = None) -> RolloutBatch:
    token_ids = _schedule_conformant_token_ids()
    trajectory = build_ar_discrete_trajectory(
        request=_request(),
        sample_rows=_sample_rows(),
        token_ids=token_ids,
        token_log_probs=torch.zeros_like(token_ids, dtype=torch.float32),
        token_mask=torch.ones_like(token_ids, dtype=torch.float32),
        prompt_input_ids=torch.ones(2, 4, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 4, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 4, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 4, dtype=torch.long),
        context=(
            context
            if context is not None
            else {"model_family": "emu3", "image_height": HEIGHT, "image_width": WIDTH}
        ),
    )
    return RolloutBatch(
        observations=torch.ones(2, 1, 4, dtype=torch.long),
        actions=token_ids,
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
    )


def test_emu3_model_exposes_trainer_replay_methods() -> None:
    model = build_tiny_emu3_model()
    assert callable(model.replay_forward)
    assert callable(model.disable_adapter)
    assert callable(model.load_trainable_state)


def test_replay_forward_returns_masked_gen_vocab_logits() -> None:
    model = build_tiny_emu3_model()
    batch = _discrete_batch()

    result = model.replay_forward(batch)

    assert isinstance(result, ReplayResult)
    segment = result.segments["image_tokens"]
    assert segment.segment == "image_tokens"
    assert set(segment.values) == {"logits", "image_token_ids"}
    logits = segment.values["logits"]
    assert logits.shape == (2, TOTAL, TINY_GEN_VOCAB)
    assert torch.equal(segment.values["image_token_ids"], batch.actions)

    # Free positions: image columns finite, structural columns -inf.
    free = logits[:, 0, :]
    assert torch.isfinite(free[:, :TINY_IMAGE_VOCAB]).all()
    assert torch.isinf(free[:, TINY_IMAGE_VOCAB:]).all()
    # Forced positions: exactly one finite column (the forced token) — the
    # renormalized log-prob replay computes there is 0, matching rollout.
    for position, forced_idx in FORCED.items():
        row = logits[:, position, :]
        finite = torch.isfinite(row)
        assert finite[:, forced_idx].all(), position
        assert finite.sum(dim=-1).eq(1).all(), position


def test_replay_forward_requires_grid_dims_in_context() -> None:
    model = build_tiny_emu3_model()
    batch = _discrete_batch(context={"model_family": "emu3"})
    with pytest.raises(RuntimeError, match="image_height/image_width"):
        model.replay_forward(batch)


def test_replay_forward_rejects_grid_token_count_mismatch() -> None:
    model = build_tiny_emu3_model()
    batch = _discrete_batch(
        context={"model_family": "emu3", "image_height": 3, "image_width": 3},
    )
    with pytest.raises(RuntimeError, match="does not match"):
        model.replay_forward(batch)


def test_replay_model_replays_without_vq_or_processor() -> None:
    model = build_tiny_emu3_replay_model()
    batch = _discrete_batch()

    result = model.replay_forward(batch)

    assert result.segments["image_tokens"].values["logits"].shape == (
        2,
        TOTAL,
        TINY_GEN_VOCAB,
    )
    with pytest.raises(RuntimeError, match="cannot decode image tokens"):
        model.decode_image_tokens(batch.actions, height=HEIGHT, width=WIDTH)
    with pytest.raises(RuntimeError, match="Emu3Processor"):
        _ = model.processor
    with pytest.raises(RuntimeError, match="VQ decoder"):
        _ = model.vq_model


def test_lora_wrap_keeps_replay_and_adapter_surfaces_working() -> None:
    model = build_tiny_emu3_model(use_lora=True)

    assert count_trainable_params(model) > 0

    batch = _discrete_batch()
    logits = model.replay_forward(batch).segments["image_tokens"].values["logits"]
    assert logits.shape == (2, TOTAL, TINY_GEN_VOCAB)

    with model.disable_adapter():
        ref_logits = model.replay_forward(batch).segments["image_tokens"].values["logits"]
    assert ref_logits.shape == logits.shape


def test_disable_adapter_without_lora_is_noop() -> None:
    model = build_tiny_emu3_model()
    with model.disable_adapter():
        pass
