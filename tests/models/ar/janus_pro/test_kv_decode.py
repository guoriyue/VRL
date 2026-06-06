"""Janus AR rollout KV-cache decode tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from vrl.generation.ar.decode_loop import ARDecodeLoop
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.ar.janus_pro.model import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProConfig,
    JanusProModel,
)
from vrl.models.ar.janus_pro.runner import JanusProARModelRunner
from vrl.nn.modules.ar_attention_backends import build_torch_native_backend

HIDDEN = 8
TEXT_VOCAB = 32


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

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: Any = None,
        **_: Any,
    ) -> SimpleNamespace:
        call_index = len(self.calls)
        self.calls.append(
            {
                "shape": tuple(inputs_embeds.shape),
                "attention_mask_shape": tuple(attention_mask.shape),
                "use_cache": use_cache,
                "past_key_values": past_key_values,
            }
        )
        hidden = torch.zeros_like(inputs_embeds)
        hidden[:, -1, 0] = float(10 + call_index)
        key = torch.full(
            (inputs_embeds.shape[0], 1, 1),
            float(call_index),
            device=inputs_embeds.device,
        )
        value = key + 100
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=((key, value),),
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


def _run_ar_decode(model: JanusProModel) -> None:
    batch_size = 2
    request = GenerationRequest(
        request_id="test-janus-kv",
        family="janus_pro",
        task="ar_t2i",
        prompts=[""],
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
    ARDecodeLoop(
        request=request,
        sample_rows=rows,
        runner=JanusProARModelRunner(
            model,
            attention_backend=build_torch_native_backend(model, family="janus_pro"),
        ),
        max_new_tokens=2,
        tokenizer_key="janus_pro",
        dtype="float32",
        scheduler_batch_size=batch_size,
        init_args=_prompt_tensors(),
        init_kwargs={"image_token_num": 2},
    ).run()


def test_janus_kv_decode_uses_prompt_prefill_for_first_token_logits() -> None:
    """Checks Janus KV decode uses prompt prefill for first token logits."""
    torch.manual_seed(0)
    model = _model()

    _run_ar_decode(model)

    lm = model.mmgpt.language_model
    assert [call["shape"] for call in lm.calls] == [
        (2, 3, HIDDEN),
        (2, 3, HIDDEN),
        (4, 1, HIDDEN),
    ]
    assert [call["use_cache"] for call in lm.calls] == [True, True, True]
    assert lm.calls[2]["past_key_values"] is not None

    first_logits_hidden = model.mmgpt.gen_head.inputs[0]
    assert first_logits_hidden.shape == (4, 1, HIDDEN)
    assert torch.equal(
        first_logits_hidden[:, 0, 0],
        torch.tensor([10.0, 10.0, 11.0, 11.0]),
    )

    second_logits_hidden = model.mmgpt.gen_head.inputs[1]
    assert torch.equal(
        second_logits_hidden[:, 0, 0],
        torch.tensor([12.0, 12.0, 12.0, 12.0]),
    )
