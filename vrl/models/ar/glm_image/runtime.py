"""GLM-Image family runtime for Ray rollout workers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.ar import ARChunkExecutorBase, ARRequestLayout
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
from vrl.models.ar.glm_image.model import (
    GlmImageConfig,
    GlmImageModel,
    GlmImageReplayModel,
    glm_image_token_num,
)
from vrl.models.ar.glm_image.runner import GlmImageTokenRunner
from vrl.models.dtypes import dtype_to_config_string
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.trajectory import build_ar_discrete_trajectory
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

GLM_IMAGE_FAMILY_CAPABILITY = ar_discrete_family_capability("glm_image", "ar_t2i")


def build_glm_image_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared AR bundle assembly."""

    config = _glm_image_config_from_runtime_spec(spec)
    return build_ar_runtime_bundle(
        spec,
        model=GlmImageModel(GlmImageConfig(**config)),
        capability=GLM_IMAGE_FAMILY_CAPABILITY,
    )


def build_glm_image_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build a GLM-Image trainer replay bundle without the DiT decode stack."""

    config = _glm_image_config_from_runtime_spec(spec)
    return build_ar_runtime_bundle(
        spec,
        model=GlmImageReplayModel(GlmImageConfig(**config)),
        capability=GLM_IMAGE_FAMILY_CAPABILITY,
        replay=True,
    )


def extract_glm_image_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
) -> RuntimeBuildSpec:
    """Slice GLM-Image runtime construction fields out of a whole RL cfg."""

    return extract_ar_runtime_spec(
        cfg,
        device,
        weight_dtype,
        ar_task="ar_t2i",
        default_model_path="zai-org/GLM-Image",
    )


# GLM-Image LoRA defaults; applied at read time so the carried ``model.lora``
# block only needs the values it overrides (same shape as the emu3 stub).
_GLM_IMAGE_LORA_DEFAULTS: dict[str, Any] = {
    "rank": 32,
    "alpha": 64,
    "target_modules": ("q_proj", "v_proj"),
    "dropout": 0.0,
    "init": "gaussian",
}


def _glm_image_config_from_runtime_spec(spec: RuntimeBuildSpec) -> dict[str, Any]:
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
        lora = dict(_GLM_IMAGE_LORA_DEFAULTS)
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

    for key in (
        "temperature",
        "top_p",
        "image_height",
        "image_width",
        "decode_num_inference_steps",
        "decode_guidance_scale",
    ):
        if key in sampling_config:
            config[key] = sampling_config[key]

    return config


@dataclass(slots=True)
class GlmImageARChunkResult:
    """Output of one prompt/sample GLM-Image AR chunk."""

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


class GlmImageChunkExecutor(ARChunkExecutorBase):
    """AR executor for GLM-Image text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``temperature``: float — AR sampling temperature (checkpoint default 0.9).
    - ``top_p``: float — nucleus filtering (checkpoint default 0.75).
    - ``image_height`` / ``image_width``: int — target pixel size, multiples
      of 32; the processor derives the large + preview token grids from it.
    - ``max_text_length``: int — pad prompts to this length so ``L_text``
      is constant across multi-prompt requests (REQUIRED).
    - ``decode_num_inference_steps`` / ``decode_guidance_scale`` — frozen
      DiT decode segment knobs (postprocess only, never trained).
    - ``seed``: int | None — when set, ``torch.manual_seed`` is applied
      per chunk for parity tests.

    Family specifics vs Emu3:

    - NO AR-side CFG: the reference generation is plain sampling (see the
      runner docstring), so there is no ``guidance_scale`` sampling knob and
      no uncond branch — ``uncond_input_ids`` in the trajectory are zeros
      that satisfy the shared discrete schema (llamagen precedent); replay
      never reads them.
    - The token count is fully determined by the pixel target
      (``prev_h*prev_w + token_h*token_w``), so there is no
      ``image_token_num`` knob.
    - The trajectory context carries ``image_height``/``image_width``
      (replay rebuilds the mrope position schedule from them).
    """

    family: str = "glm_image"
    _runner_cls = GlmImageTokenRunner
    _runner_attention_family = "glm_image"
    task: str = "ar_t2i"
    family_capability: FamilyCapability = GLM_IMAGE_FAMILY_CAPABILITY

    def __init__(self, model: Any) -> None:
        """Construct the executor.

        Args:
          model: a ``GlmImageModel`` (or a stub exposing the same interface:
            ``processor``, ``device``, ``language_model``,
            ``encode_generation_prompts``, runner-step primitives, and
            ``decode_image_tokens``).
        """
        self.model = model

    # -- protocol ------------------------------------------------------

    def _ar_runner(self, request: GenerationRequest) -> GlmImageTokenRunner:
        """Build the GLM-Image runner without a shared attention backend.

        DOCUMENTED DEVIATION (llamagen precedent): GLM-Image decode positions
        are 3-axis mrope schedules and its decoder block carries
        post_self_attn/post_mlp layernorms — the shared ``vllm_paged``
        backend hand-walks 2-layernorm LLaMA-style blocks with 1D rope, and
        ``torch_native`` cannot inject position ids, so the runner drives the
        HF trunk + DynamicCache itself. An explicit ``attention_backend``
        request is rejected instead of silently ignored.
        """
        backend = request.sampling.get("attention_backend")
        if backend is not None:
            raise ValueError(
                "glm_image does not support request.sampling.attention_backend="
                f"{backend!r}: mrope position schedules are driven natively "
                "inside the family runner."
            )
        return GlmImageTokenRunner(self.model)

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: Any,
    ) -> GlmImageARChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        from vrl.utils.profiling import record_function

        del execution_stage
        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, chunk)
        sampling = request.sampling

        temperature = float(sampling.get("temperature", self.model.config.temperature))
        top_p = float(sampling.get("top_p", self.model.config.top_p))
        if "max_text_length" not in sampling:
            raise ValueError("request.sampling.max_text_length is required")
        max_text_length = int(sampling["max_text_length"])
        image_height = int(sampling.get("image_height", self.model.config.image_height))
        image_width = int(sampling.get("image_width", self.model.config.image_width))
        decode_steps = sampling.get("decode_num_inference_steps")
        decode_guidance = sampling.get("decode_guidance_scale")
        seed = None if sampling.get("seed") is None else int(sampling.get("seed"))

        if seed is not None:
            torch.manual_seed(seed + self.layout.chunk_seed_offset(request, chunk))

        with record_function("engine.prefill"):
            repeated_prompts = [chunk.prompt] * chunk.sample_count
            prompt_ids, prompt_mask, (token_h, token_w, prev_h, prev_w) = (
                self.model.encode_generation_prompts(
                    repeated_prompts,
                    max_text_length=max_text_length,
                    image_height=image_height,
                    image_width=image_width,
                )
            )
            cond_embeds = self._embed(prompt_ids)

        total_token_num = glm_image_token_num(image_height, image_width)
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
                max_new_tokens=total_token_num,
                tokenizer_key="glm_image",
                dtype=str(cond_embeds.dtype),
                scheduler_batch_size=chunk.sample_count,
                init_args=(cond_embeds, prompt_mask),
                init_kwargs={
                    "token_h": token_h,
                    "token_w": token_w,
                    "prev_h": prev_h,
                    "prev_w": prev_w,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            ).run()
        token_ids, token_log_probs = decode_result.finalized
        with record_function("engine.vq_decode"):
            images = self.model.decode_image_tokens(
                token_ids,
                height=image_height,
                width=image_width,
                prompts=repeated_prompts,
                num_inference_steps=(
                    None if decode_steps is None else int(decode_steps)
                ),
                guidance_scale=(
                    None if decode_guidance is None else float(decode_guidance)
                ),
            )
        # Every generated position is a free codebook draw (no forced
        # structural tokens), so the whole sequence is trainable.
        token_mask = torch.ones_like(token_log_probs)
        peak_mem_mb = self.layout.peak_memory_mb()

        return GlmImageARChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=images,
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            # No AR-side uncond branch exists (no CFG). Zeros keep the shared
            # discrete trajectory schema satisfied; replay never reads them.
            uncond_input_ids=torch.zeros_like(prompt_ids),
            uncond_attention_mask=torch.zeros_like(prompt_mask),
            context={
                "temperature": temperature,
                "top_p": top_p,
                "image_height": image_height,
                "image_width": image_width,
                "image_token_num": total_token_num,
                "ar_decode_loop_enabled": True,
            },
            peak_memory_mb=peak_mem_mb,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[GlmImageARChunkResult],
    ) -> GenerationOutput:
        return GlmImageChunkGatherer().gather_chunks(request, sample_rows, chunks)


class GlmImageChunkGatherer:
    """Pure driver-side gatherer for GLM-Image AR chunk payloads."""

    layout = ARRequestLayout()

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[GlmImageARChunkResult],
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
        # Grid-derived (not a sampling knob): read from the produced tokens.
        image_token_num = int(cat["token_ids"].shape[1])
        chunk_context = dict(ordered_ar_chunks[0].context)
        metrics = GenerationMetrics(
            num_steps=image_token_num,
            chunks=len(ordered_ar_chunks),
            peak_memory_mb=peak_mem_mb,
            engine_counters={
                "ar_decode_loop_enabled": True,
                # One single-branch prefill (no CFG uncond branch).
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
    "GLM_IMAGE_FAMILY_CAPABILITY",
    "GlmImageARChunkResult",
    "GlmImageChunkExecutor",
    "GlmImageChunkGatherer",
    "build_glm_image_replay_runtime_bundle",
    "build_glm_image_runtime_bundle",
    "extract_glm_image_runtime_spec",
]
