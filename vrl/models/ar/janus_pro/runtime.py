"""Janus-Pro family runtime for Ray rollout workers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.ar import (
    ARChunkInputs,
    ARDiscreteChunkExecutorBase,
    ARRequestLayout,
    ARSamplingParams,
)
from vrl.generation.ar.decode_loop import ARDecodeLoop, call_with_supported_kwargs
from vrl.generation.capabilities import FamilyCapability
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import (
    GenerationMetrics,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.models.ar.build import ar_model_config_base
from vrl.models.ar.capabilities import ar_discrete_family_capability
from vrl.models.ar.janus_pro import JANUS_R1_SEGMENTS
from vrl.models.ar.janus_pro.runner import JanusProARModelRunner
from vrl.models.interfaces.runtime import RuntimeBuildSpec
from vrl.trajectory import build_ar_multisegment_trajectory
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

JANUS_PRO_FAMILY_CAPABILITY = ar_discrete_family_capability("janus_pro", "ar_t2i")
JANUS_PRO_R1_FAMILY_CAPABILITY = ar_discrete_family_capability(
    "janus_pro_r1",
    "ar_t2i_r1",
    trajectory_kind="multisegment",
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


def janus_config_from_runtime_spec(spec: RuntimeBuildSpec) -> dict[str, Any]:
    model_config = spec.model_config or {}
    sampling_config = spec.sampling_config or {}
    config = ar_model_config_base(spec, _JANUS_LORA_DEFAULTS)

    for key in ("guidance_scale", "temperature", "image_token_num"):
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


class JanusProChunkExecutor(ARDiscreteChunkExecutorBase):
    """AR executor for Janus-Pro text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``guidance_scale``: float — classifier-free guidance scale.
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

    def prepare_chunk_inputs(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> ARChunkInputs:
        """Encode cond+uncond prompts and wire the CFG decode loop."""

        sampling = request.sampling
        params: ARSamplingParams = self.layout.parse_sampling_params(request)

        guidance_scale = float(sampling.get("guidance_scale", 5.0))
        temperature = float(sampling.get("temperature", 1.0))

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

        return ARChunkInputs(
            max_new_tokens=params.image_token_num,
            decode_dtype=str(cond_embeds.dtype),
            init_args=(cond_embeds, uncond_embeds, prompt_mask, uncond_mask),
            init_kwargs={
                "guidance_scale": guidance_scale,
                "temperature": temperature,
                "image_token_num": params.image_token_num,
            },
            image_decode_kwargs={"image_size": params.image_size},
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context={
                "guidance_scale": guidance_scale,
                "image_token_num": params.image_token_num,
            },
            # Two branch prefills: cond and uncond run as separate forwards.
            prefill_forwards=2,
        )

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

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> JanusProR1ChunkResult:
        from vrl.utils.profiling import record_function

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
                guidance_scale=float(sampling.get("guidance_scale", 5.0)),
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
        return JanusProR1ChunkGatherer().gather_chunks(request, sample_rows, chunks)

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
        fields = ("output", "initial_image", "final_image", "selfcheck")
        ordered = self.layout.ordered_chunks(
            request,
            sample_rows,
            chunks,
            row_fields=fields,
        )
        cat = self.layout.cat_chunk_fields(ordered, fields)
        segment_extra = _cat_segment_extra(ordered)
        trajectory = build_ar_multisegment_trajectory(
            request=request,
            sample_rows=list(sample_rows),
            segments=segment_extra,
            decoded_outputs={
                "initial_image": cat["initial_image"],
                "final_image": cat["final_image"],
                "selfcheck": cat["selfcheck"],
            },
            primary_segment="final_image",
            context=dict(ordered[0].context),
        )
        peak_mem_mb = self.layout.max_peak_memory_mb(ordered)
        num_steps = _segment_token_steps(segment_extra)
        metrics = GenerationMetrics(
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
            output=cat["output"],
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
    "JanusProChunkExecutor",
    "JanusProR1ChunkExecutor",
    "JanusProR1ChunkGatherer",
    "JanusProR1ChunkResult",
    "janus_config_from_runtime_spec",
]
