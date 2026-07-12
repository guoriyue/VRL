"""LlamaGen CFG dual-sequence runner tests through the scheduled decode loop."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.models.ar.llamagen.fixtures import (
    TINY_BLOCK_SIZE,
    TINY_CLS_TOKEN_NUM,
    TINY_VOCAB_SIZE,
    build_tiny_llamagen_model,
)
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.generation.ar.decode_loop import ARDecodeLoop, ARStepBatch
from vrl.models.ar.llamagen.runner import LlamaGenARModelRunner

BATCH = 2


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="llamagen",
        task="ar_t2i",
        inputs=["a photo of a cat"],
        samples_per_prompt=BATCH,
        sampling={"seed": 0},
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
            seed=0,
        )
        for index in range(BATCH)
    ]


def _conditioning(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc = model.t5_tokenizer(
        ["a photo of a cat", "dog"],
        max_length=TINY_CLS_TOKEN_NUM,
    )
    cond, mask = model.encode_caption(enc["input_ids"], enc["attention_mask"])
    uncond = model.uncond_caption_embeds(BATCH)
    return cond, uncond, mask


def _run_loop(model, *, top_k: int = 0, guidance_scale: float = 3.0):
    cond, uncond, mask = _conditioning(model)
    return ARDecodeLoop(
        request=_request(),
        sample_rows=_sample_rows(),
        runner=LlamaGenARModelRunner(model),
        max_new_tokens=TINY_BLOCK_SIZE,
        tokenizer_key="llamagen",
        dtype=str(cond.dtype),
        scheduler_batch_size=BATCH,
        init_args=(cond, uncond, mask, mask),
        init_kwargs={
            "guidance_scale": guidance_scale,
            "temperature": 1.0,
            "top_k": top_k,
            "top_p": 1.0,
            "image_token_num": TINY_BLOCK_SIZE,
        },
    ).run()


def test_rollout_logprobs_match_teacher_forced_replay() -> None:
    """The old_log_prob invariant, end to end.

    Rollout samples through the vendored static KV cache from the CFG-guided
    distribution (guidance 3.0), but scores each token under the plain
    conditional policy. Recomputing that policy with the teacher-forced replay
    forward must reproduce the rollout log-probs exactly — this simultaneously
    pins (a) KV-cache/teacher-forced numeric parity and (b) that log-probs are
    scored on cond logits, never the guided combination (guided != cond here).
    """
    model = build_tiny_llamagen_model()
    torch.manual_seed(0)
    result = _run_loop(model, guidance_scale=3.0)
    token_ids, logprobs = result.finalized

    assert token_ids.shape == (BATCH, TINY_BLOCK_SIZE)
    assert logprobs.shape == (BATCH, TINY_BLOCK_SIZE)
    assert token_ids.dtype == torch.long
    assert int(token_ids.min()) >= 0 and int(token_ids.max()) < TINY_VOCAB_SIZE

    cond, _, mask = _conditioning(model)
    logits = model.forward_image_logits(cond, mask, token_ids)
    expected = (
        F.log_softmax(logits.float(), dim=-1)
        .gather(-1, token_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    torch.testing.assert_close(logprobs, expected, atol=1e-4, rtol=1e-4)


def test_top_k_one_is_deterministic_greedy() -> None:
    """top_k=1 collapses the guided sampling distribution to argmax."""
    model = build_tiny_llamagen_model()
    torch.manual_seed(0)
    first, _ = _run_loop(model, top_k=1).finalized
    torch.manual_seed(123)
    second, _ = _run_loop(model, top_k=1).finalized
    assert torch.equal(first, second)


def test_step_requires_full_batch_row_coverage() -> None:
    """Partial-row steps must fail loudly: the static cache advances in-place."""
    model = build_tiny_llamagen_model()
    runner = LlamaGenARModelRunner(model)
    cond, uncond, mask = _conditioning(model)
    init = runner.init_ar(
        cond,
        uncond,
        mask,
        mask,
        image_token_num=TINY_BLOCK_SIZE,
    )
    partial = ARStepBatch(
        row_indices=[0],
        positions=[0],
        position=0,
        cache_lanes={},
        row_lanes={
            "cond_logits": init.row_lanes["cond_logits"][:1],
            "uncond_logits": init.row_lanes["uncond_logits"][:1],
        },
    )
    with pytest.raises(ValueError, match="full-batch"):
        runner.step_ar(init.state, partial)


def test_step_counters_report_native_cache() -> None:
    """Debug counters advertise the native (non-paged) KV cache path."""
    model = build_tiny_llamagen_model()
    runner = LlamaGenARModelRunner(model)
    cond, uncond, mask = _conditioning(model)
    init = runner.init_ar(cond, uncond, mask, mask, image_token_num=TINY_BLOCK_SIZE)
    step = ARStepBatch(
        row_indices=list(range(BATCH)),
        positions=[0] * BATCH,
        position=0,
        cache_lanes={},
        row_lanes={
            "cond_logits": init.row_lanes["cond_logits"],
            "uncond_logits": init.row_lanes["uncond_logits"],
        },
    )
    output = runner.step_ar(init.state, step)
    counters = output.result.debug_counters
    assert counters["ar_kv_cache_enabled"] is True
    assert counters["ar_paged_attention_enabled"] is False
    assert counters["ar_prefill_forwards"] == 1
    assert set(output.updated_row_lanes) == {"cond_logits", "uncond_logits"}
