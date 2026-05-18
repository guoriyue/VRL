"""Janus runner tests for the paged-attention AR hook."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from vrl.generation.ar.decode_loop import ARDecodeLoop
from vrl.generation.ar.paged_attention import (
    ARPagedAttentionBackend,
    ARPagedAttentionConfig,
    ARPagedAttentionPrefillInput,
    ARPagedAttentionPrefillOutput,
    ARPagedAttentionStepInput,
    ARPagedAttentionStepOutput,
)
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.ar.janus_pro import runtime as janus_runtime
from vrl.models.ar.janus_pro.model import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProConfig,
    JanusProModel,
)
from vrl.models.ar.janus_pro.runner import JanusProARModelRunner
from vrl.models.ar.janus_pro.runtime import JanusProPipelineExecutor

HIDDEN = 8
TEXT_VOCAB = 32


@dataclass(frozen=True, slots=True)
class _PagedState:
    branch: str
    row: int
    tokens: int


class _RecordingPagedBackend(ARPagedAttentionBackend):
    def __init__(self) -> None:
        super().__init__(
            ARPagedAttentionConfig(family="janus_pro", model_key="test-janus")
        )
        self.prefill_requests: list[ARPagedAttentionPrefillInput] = []
        self.step_requests: list[ARPagedAttentionStepInput] = []

    def prefill(
        self,
        request: ARPagedAttentionPrefillInput,
    ) -> ARPagedAttentionPrefillOutput:
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
        return ARPagedAttentionPrefillOutput(
            last_hidden=hidden,
            sequence_states=states,
        )

    def step(self, request: ARPagedAttentionStepInput) -> ARPagedAttentionStepOutput:
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
        return ARPagedAttentionStepOutput(
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


class _RecordingHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.inputs.append(hidden.detach().clone())
        logits = torch.full(
            (*hidden.shape[:-1], JANUS_IMAGE_VOCAB_SIZE),
            -20.0,
            device=hidden.device,
        )
        logits[..., 0] = hidden[..., 0]
        return logits


class _StubVQ(nn.Module):
    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        batch_size, _, height, width = shape
        return torch.zeros(batch_size, 3, height * 16, width * 16)


class _StubMMGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _RecordingLM()
        self.gen_head = _RecordingHead()
        self.gen_vision_model = _StubVQ()
        self.gen_aligner = nn.Identity()
        self.gen_embed = nn.Embedding(JANUS_IMAGE_VOCAB_SIZE, HIDDEN)

    def prepare_gen_img_embeds(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gen_embed(ids)


def test_janus_runner_can_drive_one_paged_attention_image_step() -> None:
    torch.manual_seed(0)
    model = _model()
    backend = _RecordingPagedBackend()

    ARDecodeLoop(
        request=_request(),
        sample_rows=_rows(batch_size=2),
        runner=JanusProARModelRunner(model, paged_attention_backend=backend),
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
    model = _model()
    backend = _RecordingPagedBackend()

    def build_backend(
        passed_model: JanusProModel,
        *,
        block_size: int,
        cache_dtype: str,
    ) -> _RecordingPagedBackend:
        assert passed_model is model
        assert block_size == 32
        assert cache_dtype == "auto"
        return backend

    monkeypatch.setattr(
        janus_runtime,
        "build_janus_vllm_paged_attention_backend",
        build_backend,
    )
    request = _request()
    request.sampling.update(
        {
            "ar_paged_block_size": 32,
        }
    )

    runner = JanusProPipelineExecutor(model)._ar_runner(request)

    assert isinstance(runner, JanusProARModelRunner)
    assert runner.paged_attention_backend is backend


def _model() -> JanusProModel:
    config = JanusProConfig(
        use_lora=False,
        image_token_num=2,
    )
    return JanusProModel(config=config, mmgpt=_StubMMGPT(), processor=object())


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
        prompts=[""],
        samples_per_prompt=2,
    )


def _rows(*, batch_size: int) -> list[GenerationSampleRow]:
    return [
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
