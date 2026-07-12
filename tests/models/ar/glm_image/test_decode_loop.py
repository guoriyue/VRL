"""GLM-Image scheduled decode-loop tests on a tiny real model (CPU, no weights).

Pins the end-to-end native-cache decode path (single-branch prefill + mrope
position stepping) AND the rollout-vs-replay log-prob parity invariant: the
teacher-forced replay logits at temperature/top_p == (1.0, 1.0) must
reproduce the rollout's per-token log-probs exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.models.ar.glm_image.fixtures import (
    TINY_CODEBOOK,
    TINY_IMAGE_START_ID,
    build_tiny_glm_image_model,
)
from vrl.generation.ar.decode_loop import ARDecodeLoop
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.ar.glm_image.runner import GlmImageTokenRunner

# Tiny grids: large 4x6 + preview 2x3 -> 30 generated tokens.
TOKEN_H, TOKEN_W, PREV_H, PREV_W = 4, 6, 2, 3
TOTAL = TOKEN_H * TOKEN_W + PREV_H * PREV_W
GRIDS = ((TOKEN_H, TOKEN_W), (PREV_H, PREV_W))


def _run_tiny_decode_loop(model, batch_size: int = 2, *, top_p: float = 1.0):
    request = GenerationRequest(
        request_id="test-glm-image-decode",
        family="glm_image",
        task="ar_t2i",
        inputs=[""],
        samples_per_prompt=batch_size,
    )
    rows = [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt="",
            prompt_id="prompt-0",
            group_id="group-0",
            sample_id=f"sample-{index}",
            trajectory_id=f"trajectory-{index}",
            seed=None,
            metadata={},
        )
        for index in range(batch_size)
    ]
    embed = model.language_model.get_input_embeddings()
    # t2i-shaped prompt: text ids then the trailing image_start token; row
    # padding is on the left like the checkpoint tokenizer produces.
    cond_ids = torch.tensor([[30, 31, 32, TINY_IMAGE_START_ID]] * batch_size)
    cond_mask = torch.ones(batch_size, 4, dtype=torch.long)
    result = ARDecodeLoop(
        request=request,
        sample_rows=rows,
        runner=GlmImageTokenRunner(model),
        max_new_tokens=TOTAL,
        tokenizer_key="glm_image",
        dtype="float32",
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
    token_ids, logprobs = result.finalized

    assert token_ids.shape == (2, TOTAL)
    assert logprobs.shape == (2, TOTAL)
    # Plain raster: every position is a free codebook draw.
    assert (token_ids >= 0).all()
    assert (token_ids < TINY_CODEBOOK).all()
    assert (logprobs < 0).all()
    assert result.engine_counters["ar_prefill_forwards"] == 1
    assert result.engine_counters["ar_decode_forwards"] == TOTAL - 1


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
    token_ids, rollout_logprobs = result.finalized

    embed = model.language_model.get_input_embeddings()
    logits = model.forward_image_logits(
        embed(cond_ids), cond_mask, token_ids, grids=GRIDS,
    )
    replay_logprobs = (
        F.log_softmax(logits.float(), dim=-1)
        .gather(-1, token_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.allclose(replay_logprobs, rollout_logprobs, atol=1e-4)
