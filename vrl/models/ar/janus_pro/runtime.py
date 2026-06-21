"""Janus-Pro family runtime for Ray rollout workers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.ar import ARChunkExecutorBase, ARRequestLayout, ARSamplingParams
from vrl.generation.ar.decode_loop import ARDecodeLoop, call_with_supported_kwargs
from vrl.generation.capabilities import FamilyCapability
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.planner import attach_engine_plan, build_engine_plan
from vrl.generation.types import (
    GenerationMetrics,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.models.ar.capabilities import ar_discrete_family_capability
from vrl.models.ar.janus_pro import JANUS_R1_SEGMENTS
from vrl.models.ar.janus_pro.model import (
    JanusProConfig,
    JanusProModel,
    JanusProReplayModel,
)
from vrl.models.ar.janus_pro.runner import JanusProARModelRunner
from vrl.models.dtypes import dtype_to_config_string
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
from vrl.models.runtime_config import (
    extract_runtime_spec,
)
from vrl.trajectory import build_ar_discrete_trajectory, build_ar_multisegment_trajectory
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

JANUS_PRO_FAMILY_CAPABILITY = ar_discrete_family_capability("janus_pro", "ar_t2i")
JANUS_PRO_R1_FAMILY_CAPABILITY = ar_discrete_family_capability(
    "janus_pro_r1",
    "ar_t2i_r1",
    trajectory_kind="multisegment",
    trainable_segments=JANUS_R1_SEGMENTS,
)


def build_janus_pro_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the Janus-Pro model from a serializable runtime spec."""

    config = _janus_config_from_runtime_spec(spec)
    model = JanusProModel(JanusProConfig(**config))
    return RuntimeBundle(
        model=model,
        trainable_modules={"model": model},
        scheduler=None,
        raw_handle=model,
        runtime_caps={
            "family_capability": JANUS_PRO_FAMILY_CAPABILITY.to_dict(),
            "supports_chunked_execution": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": JANUS_PRO_FAMILY_CAPABILITY.family,
            "ar_task": spec.ar_task,
            "use_lora": spec.use_lora,
            **full_generation_bundle_metadata(),
        },
    )


def build_janus_pro_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build a Janus trainer replay bundle without VQ/vision/processor modules."""

    config = _janus_config_from_runtime_spec(spec)
    model = JanusProReplayModel(JanusProConfig(**config))
    family_capability = (
        JANUS_PRO_R1_FAMILY_CAPABILITY
        if spec.ar_task == "ar_t2i_r1"
        else JANUS_PRO_FAMILY_CAPABILITY
    )
    return RuntimeBundle(
        model=model,
        trainable_modules={"model": model},
        scheduler=None,
        raw_handle=None,
        runtime_caps={
            "family_capability": family_capability.to_dict(),
            "supports_chunked_execution": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": family_capability.family,
            "ar_task": spec.ar_task,
            "use_lora": spec.use_lora,
            **minimal_replay_bundle_metadata(),
        },
    )


def extract_janus_pro_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
) -> RuntimeBuildSpec:
    """Slice Janus-Pro runtime construction fields out of a whole RL cfg."""

    model_config = cfg.get("model") if hasattr(cfg, "get") else None
    model_path = (model_config or {}).get("path") if model_config is not None else None
    dtype = (model_config or {}).get("dtype") if model_config is not None else None
    return extract_runtime_spec(
        cfg,
        device,
        dtype_to_config_string(dtype if dtype is not None else (weight_dtype or "bfloat16")),
        ar_task="ar_t2i",
        model_name_or_path=model_path or "deepseek-ai/Janus-Pro-1B",
    )


# Janus LoRA defaults mirror the upstream Janus-Pro RL recipe; applied at read
# time so the carried ``model.lora`` block only needs the values it overrides.
_JANUS_LORA_DEFAULTS: dict[str, Any] = {
    "rank": 32,
    "alpha": 64,
    "target_modules": ("q_proj", "v_proj"),
    "dropout": 0.0,
    "init": "gaussian",
}


def _janus_config_from_runtime_spec(spec: RuntimeBuildSpec) -> dict[str, Any]:
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
        lora = _resolve_lora_block(spec, _JANUS_LORA_DEFAULTS)
        config.update(
            {
                "lora_rank": int(lora["rank"]),
                "lora_alpha": int(lora["alpha"]),
                "lora_target_modules": tuple(lora["target_modules"]),
                "lora_dropout": float(lora["dropout"]),
                "lora_init": str(lora["init"]),
            },
        )

    for key in ("cfg_weight", "temperature", "image_token_num"):
        if key in sampling_config:
            config[key] = sampling_config[key]

    for key in (
        "trust_remote_code",
        "vq_latent_channels",
    ):
        # ``None`` means "unset" in YAML; defer to JanusProConfig's own default
        # so a null cfg value does not override it (matches pre-refactor behavior
        # where unset keys never reached the config dict).
        value = model_config.get(key)
        if value is not None:
            config[key] = value

    return config


def _resolve_lora_block(spec: RuntimeBuildSpec, defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge the carried ``model.lora`` block over AR family LoRA defaults."""

    lora = dict(defaults)
    lora.update((spec.model_config or {}).get("lora") or {})
    return lora


def _ar_engine_counters(
    *,
    params: ARSamplingParams,
    batch_rows: int,
    scheduler_batches: int | None,
) -> dict[str, Any]:
    token_count = int(batch_rows) * int(params.image_token_num)
    return {
        "ar_decode_loop_enabled": True,
        "ar_prefill_forwards": 2,
        "ar_decode_forwards": max(int(params.image_token_num) - 1, 0),
        "ar_decode_tokens": token_count,
        "ar_scheduler_enabled": bool(params.use_ar_scheduler),
        "ar_scheduler_batch_size": params.ar_scheduler_batch_size,
        "ar_scheduler_batches": scheduler_batches,
    }


@dataclass(slots=True)
class JanusProARChunkResult:
    """Output of one prompt/sample Janus-Pro AR chunk."""

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


class JanusProChunkExecutor(ARChunkExecutorBase):
    """AR executor for Janus-Pro text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``cfg_weight``: float — classifier-free guidance scale.
    - ``temperature``: float — sampling temperature.
    - ``image_token_num``: int — number of AR image tokens to generate.
    - ``image_size``: int — VQ decoder output side length (pixels).
    - ``max_text_length``: int — pad/truncate prompts to this length so
      ``L_text`` is constant across multi-prompt requests.
    - ``seed``: int | None — when set, ``torch.manual_seed(seed)`` is
      applied before sampling for parity tests.

    The executor returns an ``GenerationOutput`` whose ``output`` is the
    decoded ``[B, 3, H, W]`` image tensor in ``[-1, 1]`` and whose
    ``extra`` dict carries:

    - ``token_ids``: ``[B, L_img]`` int64 — sampled image-token ids.
    - ``token_log_probs``: ``[B, L_img]`` float — per-token log-probs
      under the conditional (un-guided) policy. These are GRPO's
      ``old_log_prob``.
    - ``token_mask``: ``[B, L_img]`` float — ones tensor (Janus has no
      padding in the image-token sequence).
    - ``prompt_input_ids``: ``[B, L_text]`` int64.
    - ``prompt_attention_mask``: ``[B, L_text]`` int64.
    - ``uncond_input_ids``: ``[B, L_text]`` int64.
    - ``uncond_attention_mask``: ``[B, L_text]`` int64.

    These keys map directly into ``JanusProCollector``'s ``RolloutBatch``
    packing so the trainer's ``replay_forward`` contract stays explicit.
    """

    family: str = "janus_pro"
    _runner_cls = JanusProARModelRunner
    _runner_attention_family = "janus_pro"
    task: str = "ar_t2i"
    family_capability: FamilyCapability = JANUS_PRO_FAMILY_CAPABILITY

    def __init__(self, model: Any) -> None:
        """Construct the executor.

        Args:
          model: a ``JanusProModel`` (or a stub exposing the same
            interface: ``processor``, ``device``, ``language_model``,
            runner-step primitives, and ``decode_image_tokens``).
        """
        self.model = model

    # -- protocol ------------------------------------------------------

    def capability(self) -> FamilyCapability:
        return self.family_capability

    def plan(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
    ) -> Any:
        return build_engine_plan(
            request,
            sample_rows,
            capability=self.capability(),
        )

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        engine_plan: Any,
    ) -> GenerationOutput:
        from vrl.utils.profiling import record_function

        self.require_native_ar_engine(request)
        sampling = request.sampling
        params: ARSamplingParams = self.layout.parse_sampling_params(request)
        prompts = list(request.prompts)

        cfg_weight = float(sampling.get("cfg_weight", 5.0))
        temperature = float(sampling.get("temperature", 1.0))

        if params.seed is not None:
            # AR sampling uses torch.multinomial — we seed the global RNG
            # because that's the only entropy source in the AR runner.
            # This makes parity tests deterministic.
            torch.manual_seed(params.seed)

        # Repeat prompts samples_per_prompt times so the AR loop runs
        # samples_per_prompt independent sequences per prompt. Order is
        # prompt-major to match build_sample_rows.build_sample_rows.
        repeated_prompts = self.layout.expand_prompts(request)

        with record_function("engine.prefill"):
            # 1. Tokenise conditional + unconditional prompts.
            prompt_ids, prompt_mask = self._tokenize_prompts(
                repeated_prompts,
                max_text_length=params.max_text_length,
            )
            uncond_ids, uncond_mask = self._tokenize_prompts(
                [""] * len(repeated_prompts),
                max_text_length=params.max_text_length,
            )
            pad_id = getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
            prompt_ids, prompt_mask, uncond_ids, uncond_mask = self.layout.align_pair(
                prompt_ids,
                prompt_mask,
                uncond_ids,
                uncond_mask,
                pad_id=pad_id,
            )

            # 2. Embed both halves with the language model's input embedding.
            cond_embeds = self._embed(prompt_ids)
            uncond_embeds = self._embed(uncond_ids)

        # 3. Run the AR sampling loop.
        sample_kwargs = {
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "image_token_num": params.image_token_num,
            "ar_scheduler_batch_size": params.ar_scheduler_batch_size,
        }
        scheduler_batch_size = (
            params.ar_scheduler_batch_size if params.use_ar_scheduler else len(sample_rows)
        )
        with (
            record_function("engine.decode_step"),
            record_function("engine.cache_read"),
            record_function("engine.cache_write"),
        ):
            decode_result = ARDecodeLoop(
                request=request,
                sample_rows=sample_rows,
                runner=self._ar_runner(request),
                max_new_tokens=params.image_token_num,
                tokenizer_key="janus_pro",
                dtype=str(cond_embeds.dtype),
                scheduler_batch_size=scheduler_batch_size,
                init_args=(cond_embeds, uncond_embeds, prompt_mask, uncond_mask),
                init_kwargs=sample_kwargs,
            ).run()
        token_ids, token_log_probs = decode_result.finalized
        scheduler_batches = decode_result.scheduler_batches

        # 4. VQ decode tokens → pixels in [-1, 1].
        with record_function("engine.vq_decode"):
            images = self.model.decode_image_tokens(
                token_ids,
                image_size=params.image_size,
            )  # [B, 3, H, W]

        # 5. Token mask: every image-token position is meaningful (no
        # padding). Match the dtype of token_log_probs so trainer-side
        # multiplications don't trigger float upcasts.
        token_mask = torch.ones_like(token_log_probs)

        peak_mem_mb = self.layout.peak_memory_mb()
        engine_counters = _ar_engine_counters(
            params=params,
            batch_rows=len(sample_rows),
            scheduler_batches=scheduler_batches,
        )
        engine_counters.update(decode_result.engine_counters)
        engine_counters["ar_scheduler_enabled"] = bool(params.use_ar_scheduler)
        engine_counters["ar_scheduler_batch_size"] = params.ar_scheduler_batch_size
        metrics = GenerationMetrics(
            num_prompts=len(prompts),
            num_samples=len(sample_rows),
            num_steps=params.image_token_num,
            chunks=1,
            peak_memory_mb=peak_mem_mb,
            engine_counters=engine_counters,
        )

        trajectory_context: dict[str, Any] = {
            "cfg_weight": cfg_weight,
            "image_token_num": params.image_token_num,
            "ar_decode_loop_enabled": True,
        }
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_rows=sample_rows,
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context=trajectory_context,
        )

        return attach_engine_plan(
            GenerationOutput(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                prompts=prompts,
                sample_rows=sample_rows,
                output=images,
                trajectory=trajectory,
                extra={},
                metrics=metrics,
                peak_memory_mb=peak_mem_mb or 0.0,
            ),
            engine_plan,
        )

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: Any,
        plan_summary: Mapping[str, object],
    ) -> JanusProARChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        from vrl.utils.profiling import record_function

        del execution_stage, plan_summary
        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, chunk)
        sampling = request.sampling
        params: ARSamplingParams = self.layout.parse_sampling_params(request)

        cfg_weight = float(sampling.get("cfg_weight", 5.0))
        temperature = float(sampling.get("temperature", 1.0))

        if params.seed is not None:
            torch.manual_seed(params.seed + self.layout.chunk_seed_offset(request, chunk))

        with record_function("engine.prefill"):
            repeated_prompts = [chunk.prompt] * chunk.sample_count
            prompt_ids, prompt_mask = self._tokenize_prompts(
                repeated_prompts,
                max_text_length=params.max_text_length,
            )
            uncond_ids, uncond_mask = self._tokenize_prompts(
                [""] * chunk.sample_count,
                max_text_length=params.max_text_length,
            )
            pad_id = getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
            prompt_ids, prompt_mask, uncond_ids, uncond_mask = self.layout.align_pair(
                prompt_ids,
                prompt_mask,
                uncond_ids,
                uncond_mask,
                pad_id=pad_id,
            )

            cond_embeds = self._embed(prompt_ids)
            uncond_embeds = self._embed(uncond_ids)

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
                tokenizer_key="janus_pro",
                dtype=str(cond_embeds.dtype),
                scheduler_batch_size=chunk.sample_count,
                init_args=(cond_embeds, uncond_embeds, prompt_mask, uncond_mask),
                init_kwargs={
                    "cfg_weight": cfg_weight,
                    "temperature": temperature,
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

        return JanusProARChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=images,
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context={
                "cfg_weight": cfg_weight,
                "image_token_num": params.image_token_num,
                "ar_decode_loop_enabled": True,
            },
            peak_memory_mb=peak_mem_mb,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[JanusProARChunkResult],
    ) -> GenerationOutput:
        output = JanusProChunkGatherer().gather_chunks(request, sample_rows, chunks)
        return attach_engine_plan(output, self.plan(request, list(sample_rows)))

    # -- internals -----------------------------------------------------

    def _tokenize_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a list of prompts with the Janus chat template.

        Mirrors the pre-migration ``JanusProCollector._tokenize_prompts``
        contract: ``[B, max_text_length]`` ids + mask, right-padded with
        ``pad_token_id`` (or 0 if none), all on the model device.
        """
        tokenizer = self.model.processor.tokenizer
        device = self.model.device

        formatted = [self._format_t2i_prompt(p) for p in prompts]
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
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

    @staticmethod
    def _format_t2i_prompt(prompt: str) -> str:
        """Format a prompt with Janus' T2I chat template.

        Mirrors ``deepseek-ai/Janus/generation_inference.py``: a short
        chat-style header followed by the BOS image-generation tag.
        """
        return (
            f"<｜User｜>: {prompt}\n\n"  # noqa: RUF001
            f"<｜Assistant｜>:<begin_of_image>"  # noqa: RUF001
        )


class JanusProChunkGatherer:
    """Pure driver-side gatherer for Janus-Pro AR chunk payloads."""

    layout = ARRequestLayout()

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[JanusProARChunkResult],
    ) -> GenerationOutput:
        """Pack prompt/sample AR chunks back into the canonical GenerationOutput."""

        ordered_ar_chunks = self.layout.ordered_chunks(
            request,
            sample_rows,
            chunks,
            row_fields=(
                "output",
                "token_ids",
                "token_log_probs",
                "token_mask",
                "prompt_input_ids",
                "prompt_attention_mask",
                "uncond_input_ids",
                "uncond_attention_mask",
            ),
        )
        token_ids = torch.cat([chunk.token_ids for chunk in ordered_ar_chunks], dim=0)
        token_log_probs = torch.cat(
            [chunk.token_log_probs for chunk in ordered_ar_chunks],
            dim=0,
        )
        output = torch.cat([chunk.output for chunk in ordered_ar_chunks], dim=0)
        peak_mem_mb = self.layout.max_peak_memory_mb(ordered_ar_chunks)
        image_token_num = self.layout.parse_sampling_params(request).image_token_num
        chunk_context = dict(ordered_ar_chunks[0].context)
        metrics = GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_rows),
            num_steps=image_token_num,
            chunks=len(ordered_ar_chunks),
            peak_memory_mb=peak_mem_mb,
            engine_counters={
                "ar_decode_loop_enabled": True,
                "ar_prefill_forwards": 2,
                "ar_decode_forwards": max(image_token_num - 1, 0),
                "ar_decode_tokens": len(sample_rows) * image_token_num,
                "ar_scheduler_enabled": False,
                "ar_scheduler_batch_size": request.sampling.get("ar_scheduler_batch_size"),
                "ar_scheduler_batches": None,
            },
        )
        token_mask = torch.cat([chunk.token_mask for chunk in ordered_ar_chunks], dim=0)
        prompt_input_ids = torch.cat(
            [chunk.prompt_input_ids for chunk in ordered_ar_chunks],
            dim=0,
        )
        prompt_attention_mask = torch.cat(
            [chunk.prompt_attention_mask for chunk in ordered_ar_chunks],
            dim=0,
        )
        uncond_input_ids = torch.cat(
            [chunk.uncond_input_ids for chunk in ordered_ar_chunks],
            dim=0,
        )
        uncond_attention_mask = torch.cat(
            [chunk.uncond_attention_mask for chunk in ordered_ar_chunks],
            dim=0,
        )
        trajectory_context = chunk_context
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_rows=list(sample_rows),
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            uncond_input_ids=uncond_input_ids,
            uncond_attention_mask=uncond_attention_mask,
            context=trajectory_context,
        )

        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=list(sample_rows),
            output=output,
            trajectory=trajectory,
            extra={},
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )


@dataclass(slots=True)
class JanusProR1ChunkResult:
    """Output of one prompt/sample Janus-Pro-R1 chunk."""

    prompt_index: int
    sample_start: int
    sample_count: int
    output: torch.Tensor
    initial_image: torch.Tensor
    final_image: torch.Tensor
    selfcheck: torch.Tensor
    segments: dict[str, dict[str, Any]]
    context: dict[str, Any]
    peak_memory_mb: float | None = None


class JanusProR1ChunkExecutor(JanusProChunkExecutor):
    """R1-style Janus-Pro executor for three-stage AR T2I generation."""

    family: str = "janus_pro_r1"
    task: str = "ar_t2i_r1"
    family_capability: FamilyCapability = JANUS_PRO_R1_FAMILY_CAPABILITY

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        engine_plan: Any,
    ) -> GenerationOutput:
        from vrl.utils.profiling import record_function

        self.require_native_ar_engine(request)
        sampling = request.sampling
        params: ARSamplingParams = self.layout.parse_sampling_params(request)
        prompts = list(request.prompts)

        if params.seed is not None:
            torch.manual_seed(params.seed)

        with record_function("engine.prefill"):
            repeated_prompts = self.layout.expand_prompts(request)
            prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._tokenize_r1_prompts(
                repeated_prompts,
                max_text_length=params.max_text_length,
            )

        with (
            record_function("engine.decode_step"),
            record_function("engine.cache_read"),
            record_function("engine.cache_write"),
        ):
            scheduler_batches: list[int] = []
            result = call_with_supported_kwargs(
                self.model.generate_with_refine,
                prompt_ids,
                prompt_mask,
                cfg_weight=float(sampling.get("cfg_weight", 5.0)),
                temperature=float(sampling.get("temperature", 1.0)),
                image_token_num=params.image_token_num,
                max_reflect_len=int(sampling.get("max_reflect_len", 80)),
                task_stages=_parse_task_stages(sampling.get("task_stages")),
                uncond_input_ids=uncond_ids,
                uncond_attention_mask=uncond_mask,
                image_size=params.image_size,
                refine_mode=_resolve_refine_mode(sampling, self.model),
                image_sampler=self._r1_image_sampler(
                    request=request,
                    sample_rows=sample_rows,
                    params=params,
                    scheduler_batches=scheduler_batches,
                ),
            )

        peak_mem_mb = self.layout.peak_memory_mb()
        segment_extra = result["segments"]
        trajectory = build_ar_multisegment_trajectory(
            request=request,
            sample_rows=sample_rows,
            segments=segment_extra,
            decoded_outputs={
                "initial_image": result["initial_image"],
                "final_image": result["final_image"],
                "selfcheck": result["selfcheck"],
            },
            primary_segment="final_image",
            context=result["context"],
        )
        metrics = GenerationMetrics(
            num_prompts=len(prompts),
            num_samples=len(sample_rows),
            num_steps=_segment_token_steps(segment_extra),
            chunks=1,
            peak_memory_mb=peak_mem_mb,
            engine_counters=_ar_engine_counters(
                params=params,
                batch_rows=len(sample_rows),
                scheduler_batches=sum(scheduler_batches) if scheduler_batches else None,
            ),
        )

        return attach_engine_plan(
            GenerationOutput(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                prompts=prompts,
                sample_rows=sample_rows,
                output=result["final_image"],
                trajectory=trajectory,
                extra={},
                metrics=metrics,
                peak_memory_mb=peak_mem_mb or 0.0,
            ),
            engine_plan,
        )

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: Any,
        plan_summary: Mapping[str, object],
    ) -> JanusProR1ChunkResult:
        from vrl.utils.profiling import record_function

        del execution_stage, plan_summary
        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, chunk)
        sampling = request.sampling
        params: ARSamplingParams = self.layout.parse_sampling_params(request)

        if params.seed is not None:
            torch.manual_seed(params.seed + self.layout.chunk_seed_offset(request, chunk))

        with record_function("engine.prefill"):
            repeated_prompts = [chunk.prompt] * chunk.sample_count
            prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._tokenize_r1_prompts(
                repeated_prompts,
                max_text_length=params.max_text_length,
            )

        with (
            record_function("engine.decode_step"),
            record_function("engine.cache_read"),
            record_function("engine.cache_write"),
        ):
            chunk_specs = self.layout.chunk_sample_rows(request, chunk)
            scheduler_batches: list[int] = []
            result = call_with_supported_kwargs(
                self.model.generate_with_refine,
                prompt_ids,
                prompt_mask,
                cfg_weight=float(sampling.get("cfg_weight", 5.0)),
                temperature=float(sampling.get("temperature", 1.0)),
                image_token_num=params.image_token_num,
                max_reflect_len=int(sampling.get("max_reflect_len", 80)),
                task_stages=_parse_task_stages(sampling.get("task_stages")),
                uncond_input_ids=uncond_ids,
                uncond_attention_mask=uncond_mask,
                image_size=params.image_size,
                refine_mode=_resolve_refine_mode(sampling, self.model),
                image_sampler=self._r1_image_sampler(
                    request=request,
                    sample_rows=chunk_specs,
                    params=params,
                    scheduler_batches=scheduler_batches,
                ),
            )

        return JanusProR1ChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=result["final_image"],
            initial_image=result["initial_image"],
            final_image=result["final_image"],
            selfcheck=result["selfcheck"],
            segments=result["segments"],
            context={**result["context"], "ar_decode_loop_enabled": True},
            peak_memory_mb=self.layout.peak_memory_mb(),
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[JanusProR1ChunkResult],
    ) -> GenerationOutput:
        output = JanusProR1ChunkGatherer().gather_chunks(request, sample_rows, chunks)
        return attach_engine_plan(output, self.plan(request, list(sample_rows)))

    def _r1_image_sampler(
        self,
        *,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        params: ARSamplingParams,
        scheduler_batches: list[int],
    ) -> Any:
        """Build an R1 image sampler backed by the shared AR decode loop driver."""

        specs = list(sample_rows)
        scheduler_batch_size = (
            params.ar_scheduler_batch_size if params.use_ar_scheduler else len(specs)
        )

        def sample(
            cond_embeds: torch.Tensor,
            uncond_embeds: torch.Tensor,
            cond_mask: torch.Tensor,
            uncond_mask: torch.Tensor,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            image_token_num = int(kwargs.get("image_token_num", params.image_token_num))
            decode_result = ARDecodeLoop(
                request=request,
                sample_rows=specs,
                runner=self._ar_runner(request),
                max_new_tokens=image_token_num,
                tokenizer_key="janus_pro_r1",
                dtype=str(cond_embeds.dtype),
                scheduler_batch_size=scheduler_batch_size,
                init_args=(cond_embeds, uncond_embeds, cond_mask, uncond_mask),
                init_kwargs=kwargs,
            ).run()
            scheduler_batches.append(decode_result.scheduler_batches)
            return decode_result.finalized

        return sample

    def _tokenize_r1_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids, prompt_mask = self._tokenize_prompts(
            prompts,
            max_text_length=max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * len(prompts),
            max_text_length=max_text_length,
        )
        pad_id = getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
        return self.layout.align_pair(
            prompt_ids,
            prompt_mask,
            uncond_ids,
            uncond_mask,
            pad_id=pad_id,
        )


class JanusProR1ChunkGatherer:
    """Driver-side gatherer for Janus-Pro-R1 chunk payloads."""

    layout = ARRequestLayout()

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[JanusProR1ChunkResult],
    ) -> GenerationOutput:
        ordered = self.layout.ordered_chunks(
            request,
            sample_rows,
            chunks,
            row_fields=(
                "output",
                "initial_image",
                "final_image",
                "selfcheck",
            ),
        )
        output = torch.cat([chunk.output for chunk in ordered], dim=0)
        initial_image = torch.cat([chunk.initial_image for chunk in ordered], dim=0)
        final_image = torch.cat([chunk.final_image for chunk in ordered], dim=0)
        selfcheck = torch.cat([chunk.selfcheck for chunk in ordered], dim=0)
        segment_extra = _cat_segment_extra(ordered)
        context = dict(ordered[0].context)
        trajectory = build_ar_multisegment_trajectory(
            request=request,
            sample_rows=list(sample_rows),
            segments=segment_extra,
            decoded_outputs={
                "initial_image": initial_image,
                "final_image": final_image,
                "selfcheck": selfcheck,
            },
            primary_segment="final_image",
            context=context,
        )
        peak_mem_mb = self.layout.max_peak_memory_mb(ordered)
        num_steps = _segment_token_steps(segment_extra)
        metrics = GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_rows),
            num_steps=num_steps,
            chunks=len(ordered),
            peak_memory_mb=peak_mem_mb,
            engine_counters={
                "ar_decode_loop_enabled": True,
                "ar_prefill_forwards": 2,
                "ar_decode_forwards": max(num_steps - 1, 0),
                "ar_decode_tokens": len(sample_rows) * num_steps,
                "ar_scheduler_enabled": False,
                "ar_scheduler_batch_size": request.sampling.get("ar_scheduler_batch_size"),
                "ar_scheduler_batches": None,
            },
        )

        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=list(sample_rows),
            output=output,
            trajectory=trajectory,
            extra={},
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )


def _parse_task_stages(value: Any) -> tuple[str, ...]:
    if value is None:
        return JANUS_R1_SEGMENTS
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


def _resolve_refine_mode(sampling: dict[str, Any], model: Any) -> str:
    policy = sampling.get("final_image_policy")
    if policy == "always_generate":
        return "always"
    if policy == "use_selfcheck":
        return "selfcheck"
    return str(
        sampling.get(
            "refine_mode",
            getattr(getattr(model, "config", None), "r1_refine_mode", "selfcheck"),
        )
    )


def _cat_segment_extra(
    chunks: Sequence[JanusProR1ChunkResult],
) -> dict[str, dict[str, Any]]:
    names = tuple(chunks[0].segments)
    if set(names) != set(JANUS_R1_SEGMENTS):
        logger.warning("Unexpected Janus-Pro-R1 segment names: %s", names)

    out: dict[str, dict[str, Any]] = {}
    for name in names:
        first = chunks[0].segments[name]
        token_log_probs = None
        if first["token_log_probs"] is not None:
            token_log_probs = torch.cat(
                [chunk.segments[name]["token_log_probs"] for chunk in chunks],
                dim=0,
            )
        out[name] = {
            "name": name,
            "token_ids": torch.cat(
                [chunk.segments[name]["token_ids"] for chunk in chunks],
                dim=0,
            ),
            "token_log_probs": token_log_probs,
            "token_mask": torch.cat(
                [chunk.segments[name]["token_mask"] for chunk in chunks],
                dim=0,
            ),
            "prompt_embeds": torch.cat(
                [chunk.segments[name]["prompt_embeds"] for chunk in chunks],
                dim=0,
            ),
            "attention_mask": torch.cat(
                [chunk.segments[name]["attention_mask"] for chunk in chunks],
                dim=0,
            ),
            "prompt_attention_mask": torch.cat(
                [chunk.segments[name]["prompt_attention_mask"] for chunk in chunks],
                dim=0,
            ),
            "visual": first["visual"],
            "cfg": first["cfg"],
        }
    return out


def _segment_token_steps(segments: dict[str, dict[str, Any]]) -> int:
    return sum(int(segment["token_ids"].shape[1]) for segment in segments.values())


__all__ = [
    "JanusProARChunkResult",
    "JanusProChunkExecutor",
    "JanusProChunkGatherer",
    "JanusProR1ChunkExecutor",
    "JanusProR1ChunkGatherer",
    "JanusProR1ChunkResult",
    "build_janus_pro_replay_runtime_bundle",
    "build_janus_pro_runtime_bundle",
    "extract_janus_pro_runtime_spec",
]
