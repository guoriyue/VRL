"""Emu3 family runtime for Ray rollout workers."""

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
from vrl.models.ar.emu3.model import (
    Emu3Config,
    Emu3Model,
    Emu3ReplayModel,
    emu3_forced_token_schedule,
    emu3_grid_token_num,
)
from vrl.models.ar.emu3.runner import Emu3TokenRunner
from vrl.models.dtypes import dtype_to_config_string
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.trajectory import build_ar_discrete_trajectory
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

EMU3_FAMILY_CAPABILITY = ar_discrete_family_capability("emu3", "ar_t2i")


def build_emu3_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared AR bundle assembly."""

    config = _emu3_config_from_runtime_spec(spec)
    return build_ar_runtime_bundle(
        spec,
        model=Emu3Model(Emu3Config(**config)),
        capability=EMU3_FAMILY_CAPABILITY,
    )


def build_emu3_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build an Emu3 trainer replay bundle without the VQ decoder/processor."""

    config = _emu3_config_from_runtime_spec(spec)
    return build_ar_runtime_bundle(
        spec,
        model=Emu3ReplayModel(Emu3Config(**config)),
        capability=EMU3_FAMILY_CAPABILITY,
        replay=True,
    )


def extract_emu3_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
) -> RuntimeBuildSpec:
    """Slice Emu3 runtime construction fields out of a whole RL cfg."""

    return extract_ar_runtime_spec(
        cfg,
        device,
        weight_dtype,
        ar_task="ar_t2i",
        default_model_path="BAAI/Emu3-Gen-hf",
    )


# Emu3 LoRA defaults; applied at read time so the carried ``model.lora`` block
# only needs the values it overrides (same shape as the janus/nextstep stubs).
_EMU3_LORA_DEFAULTS: dict[str, Any] = {
    "rank": 32,
    "alpha": 64,
    "target_modules": ("q_proj", "v_proj"),
    "dropout": 0.0,
    "init": "gaussian",
}


def _emu3_config_from_runtime_spec(spec: RuntimeBuildSpec) -> dict[str, Any]:
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
        lora = dict(_EMU3_LORA_DEFAULTS)
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

    for key in ("guidance_scale", "temperature", "image_area", "ratio"):
        if key in sampling_config:
            config[key] = sampling_config[key]

    return config


@dataclass(slots=True)
class Emu3ARChunkResult:
    """Output of one prompt/sample Emu3 AR chunk."""

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


class Emu3ChunkExecutor(ARChunkExecutorBase):
    """AR executor for Emu3 text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``guidance_scale``: float — classifier-free guidance scale.
    - ``temperature``: float — sampling temperature.
    - ``image_area``: int — target pixel area; the Emu3 processor derives
      the latent grid ``(height, width)`` from it (262144 -> 64x64 at 1:1).
    - ``ratio``: str — aspect ratio, e.g. ``"1:1"``.
    - ``max_text_length``: int — pad prompts to this length so ``L_text``
      is constant across multi-prompt requests (REQUIRED).
    - ``seed``: int | None — when set, ``torch.manual_seed`` is applied
      per chunk for parity tests.

    Emu3 deliberately does NOT read ``image_token_num``/``image_size`` from
    sampling: the token count is fully determined by the latent grid
    (``h*(w+1) + 3`` including the forced EOL/EOF/EOI/EOS structural tokens),
    so a user-set knob would be dead or contradictory.

    The trajectory carries the sampled generation-vocab token ids, per-token
    conditional log-probs (GRPO's ``old_log_prob``), a token mask that zeroes
    the forced structural positions, prompt-side replay inputs, and a context
    with ``image_height``/``image_width`` (replay needs them to rebuild the
    structural mask).
    """

    family: str = "emu3"
    _runner_cls = Emu3TokenRunner
    _runner_attention_family = "emu3"
    task: str = "ar_t2i"
    family_capability: FamilyCapability = EMU3_FAMILY_CAPABILITY

    def __init__(self, model: Any) -> None:
        """Construct the executor.

        Args:
          model: an ``Emu3Model`` (or a stub exposing the same interface:
            ``processor``, ``device``, ``language_model``,
            ``encode_generation_prompts``, runner-step primitives, and
            ``decode_image_tokens``).
        """
        self.model = model

    # -- protocol ------------------------------------------------------

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: Any,
    ) -> Emu3ARChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        from vrl.utils.profiling import record_function

        del execution_stage
        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, chunk)
        sampling = request.sampling

        guidance_scale = float(sampling.get("guidance_scale", 3.0))
        temperature = float(sampling.get("temperature", 1.0))
        if "max_text_length" not in sampling:
            raise ValueError("request.sampling.max_text_length is required")
        max_text_length = int(sampling["max_text_length"])
        image_area = sampling.get("image_area")
        ratio = sampling.get("ratio")
        seed = None if sampling.get("seed") is None else int(sampling.get("seed"))

        if seed is not None:
            torch.manual_seed(seed + self.layout.chunk_seed_offset(request, chunk))

        with record_function("engine.prefill"):
            repeated_prompts = [chunk.prompt] * chunk.sample_count
            prompt_ids, prompt_mask, (height, width) = (
                self.model.encode_generation_prompts(
                    repeated_prompts,
                    max_text_length=max_text_length,
                    image_area=image_area,
                    ratio=ratio,
                )
            )
            uncond_ids, uncond_mask, uncond_grid = (
                self.model.encode_generation_prompts(
                    [""] * chunk.sample_count,
                    max_text_length=max_text_length,
                    image_area=image_area,
                    ratio=ratio,
                )
            )
            if uncond_grid != (height, width):
                raise RuntimeError(
                    f"Emu3 uncond grid {uncond_grid} != cond grid {(height, width)}",
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

        total_token_num = emu3_grid_token_num(height, width)
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
                tokenizer_key="emu3",
                dtype=str(cond_embeds.dtype),
                scheduler_batch_size=chunk.sample_count,
                init_args=(cond_embeds, uncond_embeds, prompt_mask, uncond_mask),
                init_kwargs={
                    "guidance_scale": guidance_scale,
                    "temperature": temperature,
                    "height": height,
                    "width": width,
                },
            ).run()
        token_ids, token_log_probs = decode_result.finalized
        with record_function("engine.vq_decode"):
            images = self.model.decode_image_tokens(
                token_ids,
                height=height,
                width=width,
            )
        # Forced structural positions (EOL/EOF/EOI/EOS) always renormalize to
        # a single legal token (lp == 0 for old AND new policy), so mask them
        # out of the trainable token set instead of shipping constant terms.
        forced = emu3_forced_token_schedule(
            height, width, self.model.image_vocab_size,
        ).to(token_log_probs.device)
        token_mask = (
            (forced < 0)
            .to(token_log_probs.dtype)
            .unsqueeze(0)
            .expand(token_ids.shape[0], -1)
            .contiguous()
        )
        peak_mem_mb = self.layout.peak_memory_mb()

        return Emu3ARChunkResult(
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
                "guidance_scale": guidance_scale,
                "temperature": temperature,
                "image_height": height,
                "image_width": width,
                "image_token_num": total_token_num,
                "ar_decode_loop_enabled": True,
            },
            peak_memory_mb=peak_mem_mb,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[Emu3ARChunkResult],
    ) -> GenerationOutput:
        return Emu3ChunkGatherer().gather_chunks(request, sample_rows, chunks)


class Emu3ChunkGatherer:
    """Pure driver-side gatherer for Emu3 AR chunk payloads."""

    layout = ARRequestLayout()

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[Emu3ARChunkResult],
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
                "ar_prefill_forwards": 2,
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
    "EMU3_FAMILY_CAPABILITY",
    "Emu3ARChunkResult",
    "Emu3ChunkExecutor",
    "Emu3ChunkGatherer",
    "build_emu3_replay_runtime_bundle",
    "build_emu3_runtime_bundle",
    "extract_emu3_runtime_spec",
]
