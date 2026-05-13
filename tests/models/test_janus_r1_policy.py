"""Janus-Pro-R1 policy/executor contract tests with fake weights."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from vrl.engine import GenerationIdFactory, GenerationRequest
from vrl.models.families.janus_pro.policy import (
    JanusProConfig,
    JanusProPolicy,
)
from vrl.models.families.janus_pro.r1_types import (
    JanusR1GenerationResult,
    JanusR1Segment,
)
from vrl.models.families.janus_pro.runtime import JanusProR1PipelineExecutor

HIDDEN = 16
TEXT_VOCAB = 128
IMAGE_VOCAB = 64


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str) -> list[int]:
        if text == "Yes":
            return [self.bos_token_id, 3]
        if text == "No":
            return [self.bos_token_id, 4]
        return [self.bos_token_id, *[(ord(ch) % (TEXT_VOCAB - 5)) + 5 for ch in text]]

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding, truncation
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        for text in texts:
            ids = self.encode(text)[:max_length]
            pad_len = max_length - len(ids)
            rows.append([*ids, *([self.pad_token_id] * pad_len)])
            masks.append([*(1 for _ in ids), *(0 for _ in range(pad_len))])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class _Processor:
    tokenizer = _Tokenizer()


class _LM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, TEXT_VOCAB)

    @property
    def model(self) -> _LM:
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: object = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, output_hidden_states
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            logits=self.lm_head(inputs_embeds),
            past_key_values=past_key_values,
        )


class _VQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantize = nn.Module()
        self.quantize.embedding = nn.Embedding(IMAGE_VOCAB, 4)

    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        batch, _, height, width = shape
        return torch.zeros(batch, 3, height * 16, width * 16)


class _MMGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LM()
        self.gen_vision_model = _VQ()
        self.gen_head = nn.Linear(HIDDEN, IMAGE_VOCAB)
        self.gen_embed = nn.Embedding(IMAGE_VOCAB, HIDDEN)

    def prepare_gen_img_embeds(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gen_embed(ids.clamp_min(0) % IMAGE_VOCAB)


def _policy() -> JanusProPolicy:
    return JanusProPolicy(
        JanusProConfig(
            use_lora=False,
            device="cpu",
            image_token_num=4,
            r1_refine_mode="selfcheck",
        ),
        mmgpt=_MMGPT(),
        processor=_Processor(),
    )


def test_generate_with_refine_returns_three_segments_and_selects_final_image() -> None:
    policy = _policy()
    sample_calls: list[int] = []

    def sample_image_tokens(
        cond_inputs_embeds: torch.Tensor,
        uncond_inputs_embeds: torch.Tensor,
        cond_attention_mask: torch.Tensor,
        uncond_attention_mask: torch.Tensor,
        *,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        image_token_num: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del (
            cond_inputs_embeds,
            uncond_inputs_embeds,
            cond_attention_mask,
            uncond_attention_mask,
            cfg_weight,
            temperature,
        )
        value = 10 if not sample_calls else 20
        sample_calls.append(value)
        tokens = torch.full((2, int(image_token_num or 4)), value, dtype=torch.long)
        log_probs = torch.full(tokens.shape, -0.1 * len(sample_calls))
        return tokens, log_probs

    def decode_image_tokens(
        image_token_ids: torch.Tensor,
        *,
        image_size: int = 384,
    ) -> torch.Tensor:
        del image_size
        return image_token_ids[:, :1].float().view(-1, 1, 1, 1).expand(-1, 3, 2, 2)

    def sample_selfcheck_text(
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        yes_token_id: int,
        no_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del prompt_embeds, prompt_attention_mask, temperature, eos_token_id
        text = torch.tensor(
            [
                [yes_token_id, pad_token_id, pad_token_id],
                [no_token_id, pad_token_id, pad_token_id],
            ],
            dtype=torch.long,
        )[:, :max_new_tokens]
        log_probs = torch.full(text.shape, -0.25)
        mask = torch.zeros_like(log_probs)
        mask[:, 0] = 1.0
        return text, log_probs, mask, torch.tensor([True, False])

    policy.sample_image_tokens = sample_image_tokens  # type: ignore[method-assign]
    policy.decode_image_tokens = decode_image_tokens  # type: ignore[method-assign]
    policy._sample_selfcheck_text = sample_selfcheck_text  # type: ignore[method-assign]

    prompt_ids = torch.tensor([[5, 6, 0], [7, 8, 0]], dtype=torch.long)
    prompt_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.long)
    out = policy.generate_with_refine(
        prompt_ids,
        prompt_mask,
        cfg_weight=5.0,
        temperature=0.9,
        image_token_num=4,
        max_reflect_len=3,
        refine_mode="selfcheck",
    )

    assert sample_calls == [10, 20]
    assert set(out.segments) == {"initial_image", "selfcheck_text", "final_image"}
    assert out.segments["initial_image"].token_ids.shape == (2, 4)
    assert out.segments["selfcheck_text"].token_ids.shape == (2, 3)
    assert out.segments["final_image"].token_ids.shape == (2, 4)
    assert torch.equal(out.segments["final_image"].token_ids[0], torch.full((4,), 10))
    assert torch.equal(out.segments["final_image"].token_ids[1], torch.full((4,), 20))
    assert torch.equal(out.final_image[0], torch.full((3, 2, 2), 10.0))
    assert torch.equal(out.final_image[1], torch.full((3, 2, 2), 20.0))


class _ExecutorPolicy:
    processor = _Processor()
    device = torch.device("cpu")
    config = SimpleNamespace(r1_refine_mode="selfcheck")
    language_model = _LM()
    model_family = "janus_pro"

    def generate_with_refine(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        cfg_weight: float,
        temperature: float,
        image_token_num: int,
        max_reflect_len: int,
        task_stages: tuple[str, ...],
        uncond_input_ids: torch.Tensor,
        uncond_attention_mask: torch.Tensor,
        image_size: int,
        refine_mode: str,
    ) -> JanusR1GenerationResult:
        del (
            cfg_weight,
            temperature,
            max_reflect_len,
            task_stages,
            uncond_input_ids,
            uncond_attention_mask,
            image_size,
            refine_mode,
        )
        batch = prompt_input_ids.shape[0]
        token_mask = torch.ones(batch, image_token_num)
        prompt_embeds = torch.zeros(batch, prompt_input_ids.shape[1], HIDDEN)
        image = torch.ones(batch, 3, 2, 2)
        segments = {
            name: JanusR1Segment(
                name=name,
                token_ids=torch.full((batch, image_token_num), idx, dtype=torch.long),
                token_log_probs=torch.zeros(batch, image_token_num),
                token_mask=token_mask,
                prompt_embeds=prompt_embeds,
                attention_mask=prompt_attention_mask,
                visual=name != "selfcheck_text",
                cfg=name != "selfcheck_text",
            )
            for idx, name in enumerate(
                ("initial_image", "selfcheck_text", "final_image"),
                start=1,
            )
        }
        return JanusR1GenerationResult(
            initial_image=image * 1,
            final_image=image * 2,
            selfcheck=torch.zeros(batch, dtype=torch.bool),
            segments=segments,
            context={"source": "fake"},
        )


def test_r1_executor_forward_emits_canonical_family_and_segment_schema() -> None:
    executor = JanusProR1PipelineExecutor(_ExecutorPolicy())
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        prompts=["draw text"],
        samples_per_prompt=2,
        sampling={
            "image_token_num": 4,
            "image_size": 32,
            "max_text_length": 8,
            "max_reflect_len": 3,
            "final_image_policy": "always_generate",
        },
        return_artifacts={"output", "r1_segments"},
    )
    specs = GenerationIdFactory().build_sample_specs(request)

    out = executor.forward_plan(request, specs, executor.plan(request, specs))

    assert out.family == "janus_pro_r1"
    assert out.task == "ar_t2i_r1"
    assert out.output.shape == (2, 3, 2, 2)
    assert "segments" not in out.extra
    assert "selfcheck_text" not in out.extra
    assert out.trajectory is not None
    assert set(out.trajectory.segments) >= {
        "initial_image",
        "selfcheck_text",
        "final_image",
    }
    assert (
        out.trajectory.segments["final_image"].tensors["token_ids"].value.shape
        == (2, 4)
    )
    assert (
        out.trajectory.segments["selfcheck_text"].tensors["token_ids"].value.shape
        == (2, 4)
    )
