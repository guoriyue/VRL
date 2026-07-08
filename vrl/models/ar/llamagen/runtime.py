"""LlamaGen family runtime for Ray rollout workers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.ar import ARChunkExecutorBase, ARRequestLayout, ARSamplingParams
from vrl.generation.ar.decode_loop import ARDecodeLoop
from vrl.generation.capabilities import FamilyCapability
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import (
    GenerationMetrics,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.models.ar.build import build_ar_runtime_bundle, extract_ar_runtime_spec
from vrl.models.ar.capabilities import ar_discrete_family_capability
from vrl.models.ar.llamagen.model import (
    LLAMAGEN_HF_REPO,
    LlamaGenConfig,
    LlamaGenModel,
    LlamaGenReplayModel,
)
from vrl.models.ar.llamagen.runner import LlamaGenARModelRunner
from vrl.models.dtypes import dtype_to_config_string
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.trajectory import build_ar_discrete_trajectory
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

LLAMAGEN_FAMILY_CAPABILITY = ar_discrete_family_capability("llamagen", "ar_t2i")


def build_llamagen_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared AR bundle assembly."""

    config = _llamagen_config_from_runtime_spec(spec)
    return build_ar_runtime_bundle(
        spec,
        model=LlamaGenModel(LlamaGenConfig(**config)),
        capability=LLAMAGEN_FAMILY_CAPABILITY,
    )


def build_llamagen_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build a LlamaGen trainer replay bundle without the VQ decoder."""

    config = _llamagen_config_from_runtime_spec(spec)
    return build_ar_runtime_bundle(
        spec,
        model=LlamaGenReplayModel(LlamaGenConfig(**config)),
        capability=LLAMAGEN_FAMILY_CAPABILITY,
        replay=True,
    )


def extract_llamagen_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
) -> RuntimeBuildSpec:
    """Slice LlamaGen runtime construction fields out of a whole RL cfg."""

    return extract_ar_runtime_spec(
        cfg,
        device,
        weight_dtype,
        ar_task="ar_t2i",
        default_model_path=LLAMAGEN_HF_REPO,
    )


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


def _llamagen_config_from_runtime_spec(spec: RuntimeBuildSpec) -> dict[str, Any]:
    model_config = spec.model_config or {}
    sampling_config = spec.sampling_config or {}
    use_lora = spec.use_lora
    config: dict[str, Any] = {
        "model_path": spec.model_name_or_path,
        "dtype": dtype_to_config_string(spec.dtype),
        "device": str(spec.device),
        "use_lora": use_lora,
    }

    if use_lora:
        lora = dict(_LLAMAGEN_LORA_DEFAULTS)
        lora.update(model_config.get("lora") or {})
        config.update(
            {
                "lora_rank": int(lora["rank"]),
                "lora_alpha": int(lora["alpha"]),
                "lora_target_modules": tuple(lora["target_modules"]),
                "lora_dropout": float(lora["dropout"]),
                "lora_init": str(lora["init"]),
            },
        )

    for key in ("guidance_scale", "temperature", "top_k", "top_p", "image_token_num"):
        if key in sampling_config:
            config[key] = sampling_config[key]

    for key in ("gpt_ckpt", "vq_ckpt", "gpt_model", "t5_path"):
        # ``None`` means "unset" in YAML; defer to LlamaGenConfig's own default.
        value = model_config.get(key)
        if value is not None:
            config[key] = value

    return config


@dataclass(slots=True)
class LlamaGenARChunkResult:
    """Output of one prompt/sample LlamaGen AR chunk."""

    prompt_index: int
    sample_start: int
    sample_count: int
    output: torch.Tensor
    token_ids: torch.Tensor
    token_log_probs: torch.Tensor
    token_mask: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    uncond_input_ids: torch.Tensor
    uncond_attention_mask: torch.Tensor
    context: dict[str, Any]
    peak_memory_mb: float | None = None


class LlamaGenChunkExecutor(ARChunkExecutorBase):
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
    family_capability: FamilyCapability = LLAMAGEN_FAMILY_CAPABILITY
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

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: Any,
    ) -> LlamaGenARChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        from vrl.utils.profiling import record_function

        del execution_stage
        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, chunk)
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

        if params.seed is not None:
            torch.manual_seed(params.seed + self.layout.chunk_seed_offset(request, chunk))

        with record_function("engine.prefill"):
            repeated_prompts = [chunk.prompt] * chunk.sample_count
            prompt_ids, prompt_mask = self._tokenize_prompts(
                repeated_prompts,
                max_text_length=params.max_text_length,
            )
            cond_embeds, cond_mask = self.model.encode_caption(prompt_ids, prompt_mask)
            uncond_embeds = self.model.uncond_caption_embeds(chunk.sample_count)

        chunk_specs = self.layout.chunk_sample_rows(request, chunk)
        with (
            record_function("engine.decode_step"),
            record_function("engine.cache_read"),
            record_function("engine.cache_write"),
        ):
            decode_result = ARDecodeLoop(
                request=request,
                sample_rows=chunk_specs,
                runner=self._ar_runner(request),
                max_new_tokens=params.image_token_num,
                tokenizer_key="llamagen",
                dtype=str(cond_embeds.dtype),
                # Full-batch scheduling is a hard requirement: the vendored
                # static KV cache advances in-place for the whole combined
                # cond/uncond batch (runner validates each step).
                scheduler_batch_size=chunk.sample_count,
                # Upstream generate() drives the uncond branch with the COND
                # prompt's mask (cat([emb_masks, emb_masks])).
                init_args=(cond_embeds, uncond_embeds, cond_mask, cond_mask),
                init_kwargs={
                    "guidance_scale": guidance_scale,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "image_token_num": params.image_token_num,
                },
            ).run()
        token_ids, token_log_probs = decode_result.finalized
        with record_function("engine.vq_decode"):
            images = self.model.decode_image_tokens(
                token_ids,
                image_size=params.image_size,
            )
        token_mask = torch.ones_like(token_log_probs)
        peak_mem_mb = self.layout.peak_memory_mb()

        return LlamaGenARChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=images,
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            # No unconditional token ids exist — the null caption is the
            # checkpoint's learned embedding. Zeros keep the shared discrete
            # trajectory schema satisfied; replay never reads them.
            uncond_input_ids=torch.zeros_like(prompt_ids),
            uncond_attention_mask=torch.zeros_like(prompt_mask),
            context={
                "guidance_scale": guidance_scale,
                "top_k": top_k,
                "top_p": top_p,
                "image_token_num": params.image_token_num,
                "uncond_source": "caption_embedder_uncond_embedding",
                "ar_decode_loop_enabled": True,
            },
            peak_memory_mb=peak_mem_mb,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[LlamaGenARChunkResult],
    ) -> GenerationOutput:
        return LlamaGenChunkGatherer().gather_chunks(request, sample_rows, chunks)

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


class LlamaGenChunkGatherer:
    """Pure driver-side gatherer for LlamaGen AR chunk payloads."""

    layout = ARRequestLayout()

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[LlamaGenARChunkResult],
    ) -> GenerationOutput:
        """Pack prompt/sample AR chunks back into the canonical GenerationOutput."""

        fields = (
            "output",
            "token_ids",
            "token_log_probs",
            "token_mask",
            "prompt_input_ids",
            "prompt_attention_mask",
            "uncond_input_ids",
            "uncond_attention_mask",
        )
        ordered_ar_chunks = self.layout.ordered_chunks(
            request,
            sample_rows,
            chunks,
            row_fields=fields,
        )
        cat = self.layout.cat_chunk_fields(ordered_ar_chunks, fields)
        peak_mem_mb = self.layout.max_peak_memory_mb(ordered_ar_chunks)
        image_token_num = self.layout.parse_sampling_params(request).image_token_num
        chunk_context = dict(ordered_ar_chunks[0].context)
        metrics = GenerationMetrics(
            num_steps=image_token_num,
            chunks=len(ordered_ar_chunks),
            peak_memory_mb=peak_mem_mb,
            engine_counters={
                "ar_decode_loop_enabled": True,
                # One combined [cond | uncond] prefill forward (vs Janus' two
                # branch prefills).
                "ar_prefill_forwards": 1,
                "ar_decode_forwards": max(image_token_num - 1, 0),
                "ar_decode_tokens": len(sample_rows) * image_token_num,
                "ar_scheduler_enabled": False,
                "ar_scheduler_batch_size": request.sampling.get("ar_scheduler_batch_size"),
                "ar_scheduler_batches": None,
            },
        )
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_rows=list(sample_rows),
            token_ids=cat["token_ids"],
            token_log_probs=cat["token_log_probs"],
            token_mask=cat["token_mask"],
            prompt_input_ids=cat["prompt_input_ids"],
            prompt_attention_mask=cat["prompt_attention_mask"],
            uncond_input_ids=cat["uncond_input_ids"],
            uncond_attention_mask=cat["uncond_attention_mask"],
            context=chunk_context,
        )

        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=list(sample_rows),
            output=cat["output"],
            trajectory=trajectory,
            extra={},
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )


__all__ = [
    "LLAMAGEN_FAMILY_CAPABILITY",
    "LlamaGenARChunkResult",
    "LlamaGenChunkExecutor",
    "LlamaGenChunkGatherer",
    "build_llamagen_replay_runtime_bundle",
    "build_llamagen_runtime_bundle",
    "extract_llamagen_runtime_spec",
]
