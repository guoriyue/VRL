"""Janus runner tests for the paged-attention AR hook."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from tests.models.steps.token.fixtures import RecordingHead, build_stub_janus_model
from vrl.generation.composition.causal.token_loop import CausalTokenLoop
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.families.janus_pro.model import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProModel,
)
from vrl.models.families.janus_pro.runner import JanusProARModelRunner
from vrl.models.families.janus_pro.runtime import JanusProChunkExecutor
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionConfig,
    ARAttentionPrefillInput,
    ARAttentionPrefillOutput,
    ARAttentionStepInput,
    ARAttentionStepOutput,
)
from vrl.nn.modules.ar_attention_backends import build_vllm_attention_backend

# Default cache dtype resolved from the source signature, so a default change
# (e.g. "auto" -> a concrete dtype) auto-flows into this assertion.
_DEFAULT_CACHE_DTYPE = (
    inspect.signature(build_vllm_attention_backend).parameters["cache_dtype"].default
)

HIDDEN = 8
TEXT_VOCAB = 32


@dataclass(frozen=True, slots=True)
class _PagedState:
    branch: str
    row: int
    tokens: int


class _RecordingPagedBackend(ARAttentionBackend):
    def __init__(self) -> None:
        super().__init__(ARAttentionConfig(family="janus_pro", model_key="test-janus"))
        self.prefill_requests: list[ARAttentionPrefillInput] = []
        self.step_requests: list[ARAttentionStepInput] = []

    def prefill(
        self,
        request: ARAttentionPrefillInput,
    ) -> ARAttentionPrefillOutput:
        self.prefill_requests.append(request)
        batch = request.inputs_embeds.shape[0]
        hidden = torch.zeros(batch, HIDDEN, device=request.inputs_embeds.device)
        hidden[:, 0] = 10.0 if request.branch == "cond" else 11.0
        states = tuple(
            _PagedState(
                branch=request.branch,
                row=row,
                tokens=request.attention_mask.shape[1],
            )
            for row in range(batch)
        )
        return ARAttentionPrefillOutput(
            last_hidden=hidden,
            sequence_states=states,
        )

    def step(self, request: ARAttentionStepInput) -> ARAttentionStepOutput:
        self.step_requests.append(request)
        batch = request.input_embeds.shape[0]
        hidden = torch.zeros(batch, HIDDEN, device=request.input_embeds.device)
        hidden[:, 0] = 12.0
        states = tuple(
            _PagedState(
                branch=state.branch,
                row=state.row,
                tokens=state.tokens + 1,
            )
            for state in request.sequence_states
        )
        return ARAttentionStepOutput(
            last_hidden=hidden,
            sequence_states=states,
        )


class _RecordingLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> _RecordingLM:
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            last_hidden_state=torch.zeros(1, 1, HIDDEN),
            past_key_values=None,
        )


def test_janus_runner_can_drive_one_paged_attention_image_step() -> None:
    """Checks Janus runner can drive one paged attention image step."""
    torch.manual_seed(0)
    model = _model()
    backend = _RecordingPagedBackend()

    CausalTokenLoop(
        request=_request(),
        sample_rows=_rows(batch_size=2),
        runner=JanusProARModelRunner(model, attention_backend=backend),
        max_new_tokens=2,
        tokenizer_key="janus_pro",
        dtype="float32",
        scheduler_batch_size=2,
        init_args=_prompt_tensors(),
        init_kwargs={"image_token_num": 2},
    ).run()

    assert model.mmgpt.language_model.calls == []
    assert [request.branch for request in backend.prefill_requests] == ["cond", "uncond"]
    assert len(backend.step_requests) == 1
    step = backend.step_requests[0]
    assert step.input_embeds.shape == (4, 1, HIDDEN)
    assert step.branch_names == ("cond", "cond", "uncond", "uncond")
    assert step.position == 0

    first_logits_hidden = model.mmgpt.gen_head.inputs[0]
    assert torch.equal(
        first_logits_hidden[:, 0, 0],
        torch.tensor([10.0, 10.0, 11.0, 11.0]),
    )
    second_logits_hidden = model.mmgpt.gen_head.inputs[1]
    assert torch.equal(
        second_logits_hidden[:, 0, 0],
        torch.tensor([12.0, 12.0, 12.0, 12.0]),
    )


def test_janus_runtime_uses_vllm_paged_attention_by_default(monkeypatch) -> None:
    """Checks Janus runtime uses vLLM paged attention by default."""
    model = _model()
    backend = _RecordingPagedBackend()

    def build_backend(
        passed_model: JanusProModel,
        *,
        family: str,
        block_size: int,
        cache_dtype: str,
    ) -> _RecordingPagedBackend:
        assert passed_model is model
        assert family == "janus_pro"
        assert block_size == 32
        assert cache_dtype == _DEFAULT_CACHE_DTYPE
        return backend

    monkeypatch.setattr(
        "vrl.nn.modules.ar_attention_backends.build_vllm_attention_backend",
        build_backend,
    )
    request = _request()
    request.sampling.update(
        {
            "ar_paged_block_size": 32,
        }
    )

    runner = JanusProChunkExecutor(model)._ar_runner(request)

    assert isinstance(runner, JanusProARModelRunner)
    assert runner.attention_backend is backend


def _model() -> JanusProModel:
    return build_stub_janus_model(
        language_model=_RecordingLM(),
        hidden_size=HIDDEN,
        image_vocab_size=JANUS_IMAGE_VOCAB_SIZE,
        gen_head=RecordingHead(image_vocab_size=JANUS_IMAGE_VOCAB_SIZE),
        image_token_num=2,
    )


def _prompt_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cond = torch.zeros(2, 3, HIDDEN)
    uncond = torch.zeros(2, 3, HIDDEN)
    cond_mask = torch.ones(2, 3, dtype=torch.long)
    uncond_mask = torch.ones(2, 3, dtype=torch.long)
    return cond, uncond, cond_mask, uncond_mask


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="test-janus-paged",
        family="janus_pro",
        task="ar_t2i",
        inputs=["test prompt"],
        samples_per_prompt=2,
    )


def _rows(*, batch_size: int) -> list[GenerationSampleRow]:
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt="test prompt",
            prompt_id="prompt-0",
            group_id="group-0",
            sample_id=f"sample-{index}",
            trajectory_id=f"trajectory-{index}",
            seed=None,
            metadata={},
        )
        for index in range(batch_size)
    ]
