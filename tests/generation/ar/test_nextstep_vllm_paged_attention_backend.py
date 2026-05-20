"""CUDA smoke tests for NextStep's vLLM paged-attention backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from vrl.generation.types import GenerationRequest
from vrl.models.ar.nextstep_1 import runtime as nextstep_runtime
from vrl.models.ar.nextstep_1.runner import (
    NextStep1ARModelRunner,
    build_nextstep_vllm_paged_attention_backend,
)
from vrl.models.ar.nextstep_1.runtime import NextStep1PipelineExecutor
from vrl.nn.layers.attention.paged import (
    ARPagedAttentionPrefillInput,
    ARPagedAttentionStepInput,
    ARPagedAttentionUnavailable,
)


def test_nextstep_vllm_paged_attention_matches_hf_qwen_one_step() -> None:
    if not torch.cuda.is_available():
        pytest.skip("NextStep vLLM paged attention requires CUDA")

    torch.manual_seed(0)
    trunk = _tiny_qwen_trunk()
    try:
        backend = build_nextstep_vllm_paged_attention_backend(_TinyNextStepLike(trunk))
    except ARPagedAttentionUnavailable as exc:
        pytest.skip(f"vLLM paged-attention internals are unavailable: {exc}")

    embeds = torch.randn(1, 4, 512, device="cuda", dtype=torch.float16)
    mask = torch.tensor([[0, 0, 1, 1]], device="cuda", dtype=torch.long)
    step_embeds = torch.randn(1, 1, 512, device="cuda", dtype=torch.float16)
    step_mask = torch.tensor([[0, 0, 1, 1, 1]], device="cuda", dtype=torch.long)

    with torch.no_grad():
        hf_prefill = trunk(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
        paged_prefill = backend.prefill(
            ARPagedAttentionPrefillInput(
                inputs_embeds=embeds,
                attention_mask=mask,
                branch="cond",
                metadata={"image_token_num": 2},
            )
        )
        hf_step = trunk(
            inputs_embeds=step_embeds,
            attention_mask=step_mask,
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        )
        paged_step = backend.step(
            ARPagedAttentionStepInput(
                input_embeds=step_embeds,
                attention_mask=step_mask,
                sequence_states=paged_prefill.sequence_states,
                branch_names=("cond",),
                position=0,
            )
        )

    assert paged_prefill.metrics["backend"] == "nextstep_vllm_paged_attention"
    assert paged_prefill.last_hidden.shape == (1, 512)
    assert paged_step.last_hidden.shape == (1, 512)
    assert (
        hf_prefill.last_hidden_state[:, -1, :] - paged_prefill.last_hidden
    ).abs().max().item() <= 3e-3
    assert (
        hf_step.last_hidden_state[:, -1, :] - paged_step.last_hidden
    ).abs().max().item() <= 5e-3


def test_nextstep_runtime_uses_vllm_paged_attention_by_default(monkeypatch) -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(model_path="tiny-nextstep"),
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    backend = object()

    def build_backend(
        passed_model: object,
        *,
        block_size: int,
        cache_dtype: str,
    ) -> object:
        assert passed_model is model
        assert block_size == 32
        assert cache_dtype == "auto"
        return backend

    monkeypatch.setattr(
        nextstep_runtime,
        "build_nextstep_vllm_paged_attention_backend",
        build_backend,
    )
    request = GenerationRequest(
        request_id="nextstep-vllm-paged",
        family="nextstep_1",
        task="ar_t2i",
        prompts=[""],
        samples_per_prompt=1,
        sampling={
            "ar_paged_block_size": 32,
        },
    )

    runner = NextStep1PipelineExecutor(model)._ar_runner(request)

    assert isinstance(runner, NextStep1ARModelRunner)
    assert runner.paged_attention_backend is backend


def _tiny_qwen_trunk() -> Qwen2Model:
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        _attn_implementation="eager",
    )
    return Qwen2Model(config).eval().to(device="cuda", dtype=torch.float16)


class _TinyNextStepLike:
    config = SimpleNamespace(model_path="tiny-nextstep")
    device = torch.device("cuda")
    dtype = torch.float16

    def __init__(self, trunk: Qwen2Model) -> None:
        self._trunk = trunk

    def _lm_trunk(self) -> Qwen2Model:
        return self._trunk
