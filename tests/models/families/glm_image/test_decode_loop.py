"""GLM-Image scheduled decode-loop tests on a tiny real model (CPU, no weights).

Pins the end-to-end native-cache decode path (single-branch prefill + mrope
position stepping) AND the rollout-vs-replay log-prob parity invariant: the
teacher-forced replay logits at temperature/top_p == (1.0, 1.0) must
reproduce the rollout's per-token log-probs exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.models.families.glm_image.fixtures import (
    TINY_CODEBOOK,
    TINY_IMAGE_START_ID,
    build_tiny_glm_image_model,
)
from vrl.generation.composition.token_autoregressive.token_loop import TokenAutoregressiveLoop
from vrl.models.families.glm_image.runner import GlmImageTokenRunner

# Tiny grids: large 4x6 + preview 2x3 -> 30 generated tokens.
TOKEN_H, TOKEN_W, PREV_H, PREV_W = 4, 6, 2, 3
TOTAL = TOKEN_H * TOKEN_W + PREV_H * PREV_W
GRIDS = ((TOKEN_H, TOKEN_W), (PREV_H, PREV_W))


def _run_tiny_decode_loop(model, batch_size: int = 2, *, top_p: float = 1.0):
    embed = model.language_model.get_input_embeddings()
    # t2i-shaped prompt: text ids then the trailing image_start token; row
    # padding is on the left like the checkpoint tokenizer produces.
    cond_ids = torch.tensor([[30, 31, 32, TINY_IMAGE_START_ID]] * batch_size)
    cond_mask = torch.ones(batch_size, 4, dtype=torch.long)
    result = TokenAutoregressiveLoop(
        runner=GlmImageTokenRunner(model),
        scheduler_batch_size=batch_size,
        init_args=(embed(cond_ids), cond_mask),
        init_kwargs={
            "token_h": TOKEN_H,
            "token_w": TOKEN_W,
            "prev_h": PREV_H,
            "prev_w": PREV_W,
            "temperature": 1.0,
            "top_p": top_p,
        },
    ).run()
    return result, cond_ids, cond_mask


def test_decode_loop_produces_codebook_raster_end_to_end() -> None:
    torch.manual_seed(0)
    model = build_tiny_glm_image_model()

    result, _ids, _mask = _run_tiny_decode_loop(model)
    token_ids, logprobs = result

    assert token_ids.shape == (2, TOTAL)
    assert logprobs.shape == (2, TOTAL)
    # Plain raster: every position is a free codebook draw.
    assert (token_ids >= 0).all()
    assert (token_ids < TINY_CODEBOOK).all()
    assert (logprobs < 0).all()


def test_rollout_logprobs_match_teacher_forced_replay() -> None:
    """The old/new log-prob parity invariant, end to end.

    With temperature=1.0 / top_p=1.0 the rollout's behavior distribution IS
    the conditional policy, so replaying the sampled tokens through
    ``forward_image_logits`` (full teacher-forced pass with explicit mrope
    positions, no KV cache) must reproduce the rollout log-probs. This pins
    prefill/step position ids against the replay-side position math.
    """
    torch.manual_seed(0)
    model = build_tiny_glm_image_model()

    result, cond_ids, cond_mask = _run_tiny_decode_loop(model)
    token_ids, rollout_logprobs = result

    embed = model.language_model.get_input_embeddings()
    logits = model.forward_image_logits(
        embed(cond_ids),
        cond_mask,
        token_ids,
        grids=GRIDS,
    )
    replay_logprobs = (
        F.log_softmax(logits.float(), dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    )
    assert torch.allclose(replay_logprobs, rollout_logprobs, atol=1e-4)


def test_image_decode_upsamples_tokens_without_recording_gradients(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import vrl.models.families.glm_image.model as model_module

    model = build_tiny_glm_image_model()
    monkeypatch.setattr(model_module, "glm_image_grid_dims", lambda h, w: (2, 2, 1, 1))
    expected = torch.tensor([[1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]])
    pixels = torch.full((1, 3, 4, 4), 0.5, requires_grad=True)

    def decode(**kwargs):
        assert not torch.is_grad_enabled()
        assert torch.equal(kwargs["prior_token_ids"], expected)
        return SimpleNamespace(images=pixels * 1.0)

    pipeline = Mock(side_effect=decode)
    monkeypatch.setattr(model, "_require_decode_pipeline", lambda: pipeline)
    with torch.enable_grad():
        output = model.decode_image_tokens(
            torch.tensor([[0, 1, 2, 3, 4]]), height=64, width=64, prompts=["test"]
        )
        assert torch.is_grad_enabled()
    assert not output.requires_grad
    assert torch.equal(output, torch.zeros_like(pixels))
    pipeline.assert_called_once()
