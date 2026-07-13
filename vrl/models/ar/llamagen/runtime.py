"""LlamaGen family runtime for Ray rollout workers."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.ar import (
    ARChunkInputs,
    ARDiscreteChunkExecutorBase,
    ARSamplingParams,
)
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.ar.build import ar_model_config_base
from vrl.models.ar.llamagen.runner import LlamaGenARModelRunner
from vrl.models.interfaces.runtime import ModelBuild

# LlamaGen LoRA defaults: the vendored GPT uses fused llama-style projection
# names (wqkv / wo), not per-head q_proj/k_proj/v_proj. Applied at read time so
# the carried ``model.lora`` block only needs the values it overrides.
_LLAMAGEN_LORA_DEFAULTS: dict[str, Any] = {
    "rank": 32,
    "alpha": 64,
    "target_modules": ("wqkv", "wo"),
    "dropout": 0.0,
    "init": "gaussian",
}


def llamagen_config_from_build(build: ModelBuild) -> dict[str, Any]:
    model_config = build.model_config or {}
    sampling_config = build.sampling_config or {}
    config = ar_model_config_base(build, _LLAMAGEN_LORA_DEFAULTS)

    for key in ("guidance_scale", "temperature", "top_k", "top_p", "image_token_num"):
        if key in sampling_config:
            config[key] = sampling_config[key]

    for key in ("gpt_ckpt", "vq_ckpt", "gpt_model", "t5_path"):
        # ``None`` means "unset" in YAML; defer to LlamaGenConfig's own default.
        value = model_config.get(key)
        if value is not None:
            config[key] = value

    return config


class LlamaGenChunkExecutor(ARDiscreteChunkExecutorBase):
    """AR executor for LlamaGen text-to-image rollouts.

    Same request/output contract as ``JanusProChunkExecutor`` with these
    family specifics:

    - Prompts are encoded by the frozen flan-t5-xl encoder into a fixed
      120-token caption prefix (``prompt_input_ids`` in the trajectory are T5
      token ids; replay re-encodes them with the same frozen T5).
    - CFG's unconditional branch is the checkpoint's learned null-caption
      embedding, not an empty prompt: ``uncond_input_ids`` in the trajectory
      are zeros and exist only to satisfy the shared trajectory schema.
    - ``sampling`` additionally understands ``top_k`` / ``top_p`` (upstream
      demo uses top_k=1000; default 0 = off).
    - ``max_text_length`` must equal the checkpoint's ``cls_token_num`` (120):
      the caption prefix length is baked into the GPT's rope table.
    """

    family: str = "llamagen"
    _runner_cls = LlamaGenARModelRunner
    _runner_attention_family = "llamagen"
    task: str = "ar_t2i"
    default_image_token_num: int | None = 256
    default_image_size: int | None = 256
    default_max_text_length: int | None = 120

    def __init__(self, model: Any) -> None:
        self.model = model

    # -- protocol ------------------------------------------------------

    def _ar_runner(self, request: GenerationRequest) -> LlamaGenARModelRunner:
        """Build the LlamaGen runner without a shared attention backend.

        DOCUMENTED DEVIATION: the vendored GPT's static in-place KV cache does
        not implement the HF ``past_key_values`` protocol the shared
        ``torch_native`` / ``vllm_paged`` backends require, so the runner
        drives the native cache itself. An explicit ``attention_backend``
        request is rejected instead of silently ignored.
        """
        backend = request.sampling.get("attention_backend")
        if backend is not None:
            raise ValueError(
                "llamagen does not support request.sampling.attention_backend="
                f"{backend!r}: the vendored GPT uses its own static KV cache "
                "inside the family runner."
            )
        return LlamaGenARModelRunner(self.model)

    def prepare_chunk_inputs(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> ARChunkInputs:
        """Encode the T5 caption prefix and wire the CFG decode loop."""

        sampling = request.sampling
        params: ARSamplingParams = self.layout.parse_sampling_params(request)

        cls_token_num = int(self.model.config.cls_token_num)
        if params.max_text_length != cls_token_num:
            raise ValueError(
                f"llamagen requires max_text_length == cls_token_num "
                f"({cls_token_num}); got {params.max_text_length}. The caption "
                "prefix length is baked into the GPT's rope table."
            )

        guidance_scale = float(sampling.get("guidance_scale", 7.5))
        temperature = float(sampling.get("temperature", 1.0))
        top_k = int(sampling.get("top_k", self.model.config.top_k))
        top_p = float(sampling.get("top_p", self.model.config.top_p))

        repeated_prompts = [chunk.prompt] * chunk.sample_count
        prompt_ids, prompt_mask = self._tokenize_prompts(
            repeated_prompts,
            max_text_length=params.max_text_length,
        )
        cond_embeds, cond_mask = self.model.encode_caption(prompt_ids, prompt_mask)
        uncond_embeds = self.model.uncond_caption_embeds(chunk.sample_count)

        return ARChunkInputs(
            max_new_tokens=params.image_token_num,
            decode_dtype=str(cond_embeds.dtype),
            # Upstream generate() drives the uncond branch with the COND
            # prompt's mask (cat([emb_masks, emb_masks])). Full-batch
            # scheduling is a hard requirement: the vendored static KV cache
            # advances in-place for the whole combined cond/uncond batch
            # (runner validates each step).
            init_args=(cond_embeds, uncond_embeds, cond_mask, cond_mask),
            init_kwargs={
                "guidance_scale": guidance_scale,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "image_token_num": params.image_token_num,
            },
            image_decode_kwargs={"image_size": params.image_size},
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            # No unconditional token ids exist — the null caption is the
            # checkpoint's learned embedding. Zeros keep the shared discrete
            # trajectory schema satisfied; replay never reads them.
            uncond_input_ids=torch.zeros_like(prompt_ids),
            uncond_attention_mask=torch.zeros_like(prompt_mask),
            context={
                "guidance_scale": guidance_scale,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "image_token_num": params.image_token_num,
                "uncond_source": "caption_embedder_uncond_embedding",
            },
        )

    # -- internals -----------------------------------------------------

    def _tokenize_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompts with the T5 tokenizer (upstream T5Embedder args).

        ``lower().strip()`` is upstream's ``use_text_preprocessing=False``
        branch (the default ftfy/bs4 caption cleaner is deliberately not
        replicated — see ``model._load_t5_encoder``). Right-padded ids/mask;
        ``encode_caption`` performs the upstream left-pad flip.
        """
        tokenizer = self.model.t5_tokenizer
        device = self.model.device

        texts = [prompt.lower().strip() for prompt in prompts]
        enc = tokenizer(
            texts,
            max_length=max_text_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        ids, mask = self._align_tokenizer_output(
            ids,
            mask,
            max_text_length=max_text_length,
            pad_id=getattr(tokenizer, "pad_token_id", None) or 0,
        )
        return ids.to(device), mask.to(device)


__all__ = [
    "LlamaGenChunkExecutor",
    "llamagen_config_from_build",
]
