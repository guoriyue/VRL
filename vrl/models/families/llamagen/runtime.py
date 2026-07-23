"""LlamaGen family runtime for Ray rollout workers."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.bindings.token_autoregressive import (
    ARChunkInputs,
    ARDiscreteChunkExecutorBase,
    ARSamplingParams,
)
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.families.llamagen.config import (
    LLAMAGEN_CAPTION_TOKEN_NUM,
    LLAMAGEN_DOWNSAMPLE_SIZE,
    LLAMAGEN_IMAGE_TOKEN_NUM,
    llamagen_image_grid_side,
    llamagen_image_size,
)
from vrl.models.families.llamagen.runner import LlamaGenARModelRunner
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.token.build import token_model_config_base


def llamagen_config_from_build(build: ModelBuild) -> dict[str, Any]:
    model_config = build.model_config or {}
    sampling_config = build.sampling_config or {}
    config = token_model_config_base(build)

    for key in ("guidance_scale", "temperature", "top_k", "top_p"):
        if key in sampling_config:
            config[key] = sampling_config[key]

    image_token_num = int(
        model_config.get("image_token_num", LLAMAGEN_IMAGE_TOKEN_NUM),
    )
    llamagen_image_grid_side(image_token_num)
    for name, expected, owner in (
        (
            "image_token_num",
            image_token_num,
            f"model.image_token_num={image_token_num}",
        ),
        (
            "image_size",
            llamagen_image_size(image_token_num, LLAMAGEN_DOWNSAMPLE_SIZE),
            "the decoded size derived from model.image_token_num and VQ stride",
        ),
        (
            "max_text_length",
            LLAMAGEN_CAPTION_TOKEN_NUM,
            "the checkpoint caption-prefix length",
        ),
    ):
        requested = sampling_config.get(name)
        if requested is not None and int(requested) != expected:
            raise ValueError(
                f"sampling.{name}={int(requested)} must equal {owner} ({expected})",
            )
    config["image_token_num"] = image_token_num

    for key in (
        "gpt_ckpt",
        "vq_ckpt",
        "gpt_model",
        "t5_path",
        "t5_revision",
    ):
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
    task: str = "ar_t2i"

    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def default_image_token_num(self) -> int:
        """Derive the request default from the GPT construction topology."""

        return int(self.model.config.image_token_num)

    @property
    def default_image_size(self) -> int:
        """Derive decoded pixels from the fixed token grid and VQ stride."""

        return llamagen_image_size(
            self.default_image_token_num,
            int(self.model.config.downsample_size),
        )

    @property
    def default_max_text_length(self) -> int:
        """Derive the request default from the checkpoint caption prefix."""

        return int(self.model.config.cls_token_num)

    # -- protocol ------------------------------------------------------

    def resolve_scheduler_batch_size(
        self,
        request: GenerationRequest,
        *,
        row_count: int,
    ) -> int | None:
        """Require every native static-KV step to cover the full chunk."""

        batch_size = super().resolve_scheduler_batch_size(
            request,
            row_count=row_count,
        )
        if batch_size is not None and batch_size < row_count:
            raise ValueError(
                "llamagen requires request.sampling.ar_scheduler_batch_size "
                f"to be null or >= chunk sample count ({row_count}); got {batch_size}",
            )
        return batch_size

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

        image_token_num = int(self.model.config.image_token_num)
        if params.image_token_num != image_token_num:
            raise ValueError(
                f"llamagen requires image_token_num == model.image_token_num "
                f"({image_token_num}); got {params.image_token_num}. The token "
                "grid is baked into the GPT's 2D RoPE table."
            )
        expected_image_size = self.default_image_size
        if params.image_size != expected_image_size:
            raise ValueError(
                f"llamagen requires image_size={expected_image_size} for "
                f"model.image_token_num={image_token_num}; got {params.image_size}."
            )
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
                "temperature": temperature,
                # Display/provenance-only: OnlineTrainer writes these behavior
                # sampler knobs into its first-step ``rollout_context`` record.
                "guidance_scale": guidance_scale,
                "top_k": top_k,
                "top_p": top_p,
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
