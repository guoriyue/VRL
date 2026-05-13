"""Janus-Pro family runtime for Ray rollout workers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vrl.engine.ar import (
    ActiveSequence,
    ARGenerationSpec,
    ARPipelineExecutorBase,
    ARTokenScheduler,
    max_peak_memory_mb,
    ordered_chunks,
)
from vrl.engine.core.capabilities import FamilyCapability, ar_discrete_family_capability
from vrl.engine.core.planner import attach_engine_plan, build_engine_plan
from vrl.engine.core.protocols import PipelineChunkResult
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)
from vrl.engine.microbatching import MicroBatchPlan
from vrl.engine.trajectory import build_ar_discrete_trajectory, build_ar_multisegment_trajectory
from vrl.models.families.janus_pro.policy import JanusProConfig, JanusProPolicy
from vrl.models.families.janus_pro.r1_types import JanusR1Segment
from vrl.models.runtime import RuntimeBuildSpec, RuntimeBundle

logger = logging.getLogger(__name__)

JANUS_PRO_FAMILY_CAPABILITY = ar_discrete_family_capability("janus_pro", "ar_t2i")
JANUS_PRO_R1_FAMILY_CAPABILITY = ar_discrete_family_capability(
    "janus_pro_r1",
    "ar_t2i_r1",
    multisegment=True,
)


def build_janus_pro_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the Janus-Pro policy from a serializable runtime spec."""

    config = _janus_config_from_runtime_spec(spec)
    policy = JanusProPolicy(JanusProConfig(**config))
    return RuntimeBundle(
        policy=policy,
        trainable_modules={"policy": policy},
        scheduler=None,
        backend_kind="janus_pro",
        backend_handle=policy,
        runtime_caps={
            "family_capability": JANUS_PRO_FAMILY_CAPABILITY.to_dict(),
            "supports_chunked_execution": True,
            "supports_token_logprobs": True,
            "supports_cfg": True,
            "supports_batched_decode": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "use_lora": spec.use_lora,
        },
    )


def extract_janus_pro_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
) -> RuntimeBuildSpec:
    """Slice Janus-Pro runtime construction fields out of a whole RL cfg."""

    dtype = _cfg_path(cfg, "model.dtype", weight_dtype or "bfloat16")
    return RuntimeBuildSpec(
        model_name_or_path=str(
            _cfg_path(cfg, "model.path", "deepseek-ai/Janus-Pro-1B"),
        ),
        device=device,
        dtype=_dtype_to_config_string(dtype),
        backend_preference=("native",),
        task_variant="ar_t2i",
        use_lora=bool(_cfg_path(cfg, "model.use_lora", True)),
        lora_config={
            "rank": int(_cfg_path(cfg, "model.lora.rank", 32)),
            "alpha": int(_cfg_path(cfg, "model.lora.alpha", 64)),
            "dropout": float(_cfg_path(cfg, "model.lora.dropout", 0.0)),
            "target_modules": list(
                _cfg_path(cfg, "model.lora.target_modules", ("q_proj", "v_proj")),
            ),
            "init": str(_cfg_path(cfg, "model.lora.init", "gaussian")),
        },
        scheduler_config={
            "cfg_weight": float(_cfg_path(cfg, "sampling.cfg_weight", 5.0)),
            "temperature": float(_cfg_path(cfg, "sampling.temperature", 1.0)),
            "image_token_num": int(_cfg_path(cfg, "sampling.image_token_num", 576)),
        },
        extra={
            "freeze_vq": bool(_cfg_path(cfg, "model.freeze_vq", True)),
            "freeze_vision_encoder": bool(
                _cfg_path(cfg, "model.freeze_vision_encoder", True),
            ),
            "freeze_aligner": bool(_cfg_path(cfg, "model.freeze_aligner", True)),
        },
    )


def _janus_config_from_runtime_spec(spec: RuntimeBuildSpec) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_path": spec.model_name_or_path,
        "dtype": _dtype_to_config_string(spec.dtype),
        "device": str(spec.device),
        "use_lora": bool(spec.use_lora),
    }

    if spec.lora_config:
        config.update(
            {
                "lora_rank": int(spec.lora_config["rank"]),
                "lora_alpha": int(spec.lora_config["alpha"]),
                "lora_target_modules": tuple(spec.lora_config["target_modules"]),
            },
        )
        if "dropout" in spec.lora_config:
            config["lora_dropout"] = float(spec.lora_config["dropout"])
        if "init" in spec.lora_config:
            config["lora_init"] = str(spec.lora_config["init"])

    if spec.scheduler_config:
        for key in ("cfg_weight", "temperature", "image_token_num"):
            if key in spec.scheduler_config:
                config[key] = spec.scheduler_config[key]

    for key in (
        "trust_remote_code",
        "freeze_vq",
        "freeze_vision_encoder",
        "freeze_aligner",
        "vq_latent_channels",
    ):
        if key in spec.extra:
            config[key] = spec.extra[key]

    return config


def _dtype_to_config_string(value: Any) -> str:
    text = str(value).removeprefix("torch.")
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "float": "float32",
    }
    return aliases.get(text.lower(), text)


_MISSING = object()


def _cfg_path(cfg: Any, path: str, default: Any) -> Any:
    node = cfg
    for key in path.split("."):
        node = _cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return default
    return node


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


"""Janus-Pro AR text-to-image pipeline executor.

Owns the autoregressive image-token sampling + VQ decode previously
inlined in the Janus-Pro rollout collector. The collector keeps
reward scoring and ``RolloutBatch`` packing.

Boundary:

- This module MUST NOT import ``vrl.rollouts.*`` or ``RolloutBatch``.
- This module MUST NOT compute reward.
- Inputs come from ``GenerationRequest.sampling`` + ``prompts`` (the
  collector packs them).
- Outputs are the canonical ``OutputBatch`` whose ``output`` is the
  decoded image tensor and whose ``extra`` carries per-token ids,
  per-token log-probs, the prompt token ids/masks (needed for replay
  forward), and the unconditional token ids/masks (needed for replay /
  audit).

Difference from diffusion executors: AR runs a token loop. The default path
uses the model's black-box ``sample_image_tokens`` method; setting
``sampling.use_ar_scheduler`` routes through the executor-internal
``ARTokenScheduler`` and the model's ``init_ar_state`` / ``step_ar`` /
``finalize_ar_state`` contract.

Parity contract: same prompts + same seed (when seeded) produce
bitwise-equal token ids, log-probs, and images, since
``sample_image_tokens`` is wrapped under ``torch.no_grad`` and the only
randomness is ``torch.multinomial``. The collector must apply
``torch.manual_seed(seed)`` before calling the runtime to make this
reproducible.
"""

@dataclass(slots=True)
class JanusProARChunkResult(PipelineChunkResult):
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


class JanusProPipelineExecutor(ARPipelineExecutorBase):
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

    The executor returns an ``OutputBatch`` whose ``output`` is the
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
    task: str = "ar_t2i"
    family_capability: FamilyCapability = JANUS_PRO_FAMILY_CAPABILITY

    def __init__(self, model: Any) -> None:
        """Construct the executor.

        Args:
          model: a ``JanusProPolicy`` (or a stub exposing the same
            interface: ``processor``, ``device``, ``language_model``,
            ``sample_image_tokens``, ``decode_image_tokens``).
        """
        self.model = model

    # -- protocol ------------------------------------------------------

    def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
        return WorkloadSignature.from_request_and_capability(request, self.capability())

    def capability(self) -> FamilyCapability:
        return self.family_capability

    def plan(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> Any:
        return build_engine_plan(
            request,
            sample_specs,
            capability=self.capability(),
        )

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        from vrl.trainers.profiling import record_function

        engine_plan = self.plan(request, sample_specs)
        sampling = request.sampling
        spec: ARGenerationSpec = self.parse_spec(request)
        prompts = list(request.prompts)

        cfg_weight = float(sampling.get("cfg_weight", 5.0))
        temperature = float(sampling.get("temperature", 1.0))

        if spec.seed is not None:
            # AR sampling uses torch.multinomial — we seed the global RNG
            # because that's the only entropy source in sample_image_tokens.
            # This makes parity tests deterministic.
            torch.manual_seed(spec.seed)

        # Repeat prompts samples_per_prompt times so the AR loop runs
        # samples_per_prompt independent sequences per prompt. Order is
        # prompt-major to match GenerationIdFactory.build_sample_specs.
        repeated_prompts = self.expand_prompts(request)

        with record_function("engine.prefill"):
            # 1. Tokenise conditional + unconditional prompts.
            prompt_ids, prompt_mask = self._tokenize_prompts(
                repeated_prompts,
                max_text_length=spec.max_text_length,
            )
            uncond_ids, uncond_mask = self._tokenize_prompts(
                [""] * len(repeated_prompts),
                max_text_length=spec.max_text_length,
            )
            pad_id = getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
            prompt_ids, prompt_mask, uncond_ids, uncond_mask = self.align_pair(
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
            "image_token_num": spec.image_token_num,
        }
        if spec.use_ar_scheduler:
            with record_function("engine.decode_step"):
                token_ids, token_log_probs = self._sample_with_ar_scheduler(
                    request=request,
                    sample_specs=sample_specs,
                    cond_embeds=cond_embeds,
                    uncond_embeds=uncond_embeds,
                    prompt_mask=prompt_mask,
                    uncond_mask=uncond_mask,
                    image_token_num=spec.image_token_num,
                    sample_kwargs=sample_kwargs,
                )
        else:
            with (
                record_function("engine.decode_step"),
                record_function("engine.cache_read"),
                record_function("engine.cache_write"),
            ):
                token_ids, token_log_probs = self.model.sample_image_tokens(
                    cond_embeds,
                    uncond_embeds,
                    prompt_mask,
                    uncond_mask,
                    **sample_kwargs,
                )  # both [B, L_img]

        # 4. VQ decode tokens → pixels in [-1, 1].
        with record_function("engine.vq_decode"):
            images = self.model.decode_image_tokens(
                token_ids,
                image_size=spec.image_size,
            )  # [B, 3, H, W]

        # 5. Token mask: every image-token position is meaningful (no
        # padding). Match the dtype of token_log_probs so trainer-side
        # multiplications don't trigger float upcasts.
        token_mask = torch.ones_like(token_log_probs)

        peak_mem_mb = self.peak_memory_mb()
        metrics = GenerationMetrics(
            num_prompts=len(prompts),
            num_samples=len(sample_specs),
            num_steps=spec.image_token_num,
            micro_batches=1,
            peak_memory_mb=peak_mem_mb,
        )

        extra: dict[str, Any] = {
            "token_ids": token_ids,
            "token_log_probs": token_log_probs,
            "token_mask": token_mask,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "uncond_input_ids": uncond_ids,
            "uncond_attention_mask": uncond_mask,
            "context": {
                "cfg_weight": cfg_weight,
                "image_token_num": spec.image_token_num,
                "model_family": getattr(self.model, "model_family", "janus_pro"),
            },
        }
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_specs=sample_specs,
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context=extra["context"],
        )

        return attach_engine_plan(OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=prompts,
            sample_specs=sample_specs,
            output=images,
            rollout_trajectory_data=None,  # AR has no DiT trajectory
            trajectory=trajectory,
            extra=extra,
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        ), engine_plan)

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> JanusProARChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        from vrl.trainers.profiling import record_function

        self.validate_chunk(request, chunk)
        sampling = request.sampling
        spec: ARGenerationSpec = self.parse_spec(request)

        cfg_weight = float(sampling.get("cfg_weight", 5.0))
        temperature = float(sampling.get("temperature", 1.0))

        if spec.seed is not None:
            torch.manual_seed(spec.seed + self.chunk_seed_offset(request, chunk))

        with record_function("engine.prefill"):
            repeated_prompts = [chunk.prompt] * chunk.sample_count
            prompt_ids, prompt_mask = self._tokenize_prompts(
                repeated_prompts,
                max_text_length=spec.max_text_length,
            )
            uncond_ids, uncond_mask = self._tokenize_prompts(
                [""] * chunk.sample_count,
                max_text_length=spec.max_text_length,
            )
            pad_id = getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
            prompt_ids, prompt_mask, uncond_ids, uncond_mask = self.align_pair(
                prompt_ids,
                prompt_mask,
                uncond_ids,
                uncond_mask,
                pad_id=pad_id,
            )

            cond_embeds = self._embed(prompt_ids)
            uncond_embeds = self._embed(uncond_ids)

        # Distributed AR chunks stay at prompt/sample granularity. The
        # token-level scheduler remains executor-internal for direct execution.
        with (
            record_function("engine.decode_step"),
            record_function("engine.cache_read"),
            record_function("engine.cache_write"),
        ):
            token_ids, token_log_probs = self.model.sample_image_tokens(
                cond_embeds,
                uncond_embeds,
                prompt_mask,
                uncond_mask,
                cfg_weight=cfg_weight,
                temperature=temperature,
                image_token_num=spec.image_token_num,
            )
        with record_function("engine.vq_decode"):
            images = self.model.decode_image_tokens(
                token_ids,
                image_size=spec.image_size,
            )
        token_mask = torch.ones_like(token_log_probs)
        peak_mem_mb = self.peak_memory_mb()

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
                "image_token_num": spec.image_token_num,
                "model_family": getattr(self.model, "model_family", "janus_pro"),
            },
            peak_memory_mb=peak_mem_mb,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[JanusProARChunkResult],
    ) -> OutputBatch:
        output = JanusProChunkGatherer().gather_chunks(request, sample_specs, chunks)
        return attach_engine_plan(output, self.plan(request, list(sample_specs)))

    # -- internals -----------------------------------------------------

    def _sample_with_ar_scheduler(
        self,
        *,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
        cond_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor,
        image_token_num: int,
        sample_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Janus-Pro sampling through the executor-internal AR scheduler."""

        required = ("init_ar_state", "step_ar", "finalize_ar_state")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(
                "use_ar_scheduler=True requires model step API methods: " + ", ".join(missing)
            )

        if cond_embeds.shape[0] != len(sample_specs):
            raise ValueError(
                "Scheduled AR expects one sample spec per embedded row: "
                f"{len(sample_specs)} specs for {cond_embeds.shape[0]} rows"
            )

        state = self.model.init_ar_state(
            cond_embeds,
            uncond_embeds,
            prompt_mask,
            uncond_mask,
            **sample_kwargs,
        )
        sequences = [
            ActiveSequence(
                request_id=request.request_id,
                sample_id=spec.sample_id,
                family=request.family,
                task=request.task,
                tokenizer_key="janus_pro",
                dtype=str(cond_embeds.dtype),
                max_new_tokens=image_token_num,
                metadata={
                    **dict(spec.metadata),
                    "row_index": row_index,
                    "prompt_index": spec.prompt_index,
                    "sample_index": spec.sample_index,
                },
            )
            for row_index, spec in enumerate(sample_specs)
        ]
        scheduler = ARTokenScheduler(
            max_batch_size=max(
                1,
                int(request.sampling.get("ar_scheduler_batch_size", len(sequences))),
            )
        )
        scheduler.add_many(sequences)

        while True:
            batch = scheduler.pop_batch()
            if batch is None:
                break
            self.model.step_ar(state, batch.sequences)
            for sequence in batch.sequences:
                sequence.advance()
            scheduler.push_back_unfinished(batch)

        return self.model.finalize_ar_state(state)

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

        # Belt-and-braces: enforce L_text == max_text_length even if the
        # tokenizer ignored padding="max_length" (stubs / tokenisers
        # without a pad_token).
        if ids.shape[1] < max_text_length:
            pad_id = getattr(tokenizer, "pad_token_id", None) or 0
            extra_len = max_text_length - ids.shape[1]
            ids = torch.cat(
                [
                    ids,
                    torch.full(
                        (ids.shape[0], extra_len),
                        pad_id,
                        dtype=ids.dtype,
                    ),
                ],
                dim=1,
            )
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(
                        (mask.shape[0], extra_len),
                        dtype=mask.dtype,
                    ),
                ],
                dim=1,
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

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed = self.model.language_model.get_input_embeddings()
        return embed(token_ids)


class JanusProChunkGatherer:
    """Pure driver-side gatherer for Janus-Pro AR chunk payloads."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[JanusProARChunkResult],
    ) -> OutputBatch:
        """Pack prompt/sample AR chunks back into the canonical OutputBatch."""

        ordered_ar_chunks = ordered_chunks(
            request,
            sample_specs,
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
        peak_mem_mb = max_peak_memory_mb(ordered_ar_chunks)
        metrics = GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_specs),
            num_steps=int(request.sampling.get("image_token_num", 576)),
            micro_batches=len(ordered_ar_chunks),
            peak_memory_mb=peak_mem_mb,
        )
        extra: dict[str, Any] = {
            "token_ids": token_ids,
            "token_log_probs": token_log_probs,
            "token_mask": torch.cat(
                [chunk.token_mask for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "prompt_input_ids": torch.cat(
                [chunk.prompt_input_ids for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "prompt_attention_mask": torch.cat(
                [chunk.prompt_attention_mask for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "uncond_input_ids": torch.cat(
                [chunk.uncond_input_ids for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "uncond_attention_mask": torch.cat(
                [chunk.uncond_attention_mask for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "context": dict(ordered_ar_chunks[0].context),
        }
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_specs=list(sample_specs),
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=extra["token_mask"],
            prompt_input_ids=extra["prompt_input_ids"],
            prompt_attention_mask=extra["prompt_attention_mask"],
            uncond_input_ids=extra["uncond_input_ids"],
            uncond_attention_mask=extra["uncond_attention_mask"],
            context=extra["context"],
        )

        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=output,
            rollout_trajectory_data=None,
            trajectory=trajectory,
            extra=extra,
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )


# F is imported for potential future uses (entropy etc.) — keep silent
# usage so linters don't strip the import; Janus' executor itself only
# uses model.sample_image_tokens which already does softmax internally.
_ = F


"""Janus-Pro-R1 AR text-to-image pipeline executor.

This executor owns generation only. Reward computation, advantage
normalization, and rollout packing stay outside the model family layer.
"""

R1_SEGMENT_NAMES = ("initial_image", "selfcheck_text", "final_image")


@dataclass(slots=True)
class JanusProR1ChunkResult(PipelineChunkResult):
    """Output of one prompt/sample Janus-Pro-R1 chunk."""

    prompt_index: int
    sample_start: int
    sample_count: int
    output: torch.Tensor
    initial_image: torch.Tensor
    final_image: torch.Tensor
    selfcheck: torch.Tensor
    segments: dict[str, JanusR1Segment]
    context: dict[str, Any]
    peak_memory_mb: float | None = None


class JanusProR1PipelineExecutor(JanusProPipelineExecutor):
    """R1-style Janus-Pro executor for three-stage AR T2I generation."""

    family: str = "janus_pro_r1"
    task: str = "ar_t2i_r1"
    family_capability: FamilyCapability = JANUS_PRO_R1_FAMILY_CAPABILITY

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        from vrl.trainers.profiling import record_function

        engine_plan = self.plan(request, sample_specs)
        sampling = request.sampling
        spec: ARGenerationSpec = self.parse_spec(request)
        prompts = list(request.prompts)

        if spec.seed is not None:
            torch.manual_seed(spec.seed)

        with record_function("engine.prefill"):
            repeated_prompts = self.expand_prompts(request)
            prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._tokenize_r1_prompts(
                repeated_prompts,
                max_text_length=spec.max_text_length,
            )

        with (
            record_function("engine.decode_step"),
            record_function("engine.cache_read"),
            record_function("engine.cache_write"),
        ):
            result = self.model.generate_with_refine(
                prompt_ids,
                prompt_mask,
                cfg_weight=float(sampling.get("cfg_weight", 5.0)),
                temperature=float(sampling.get("temperature", 1.0)),
                image_token_num=spec.image_token_num,
                max_reflect_len=int(sampling.get("max_reflect_len", 80)),
                task_stages=_parse_task_stages(sampling.get("task_stages")),
                uncond_input_ids=uncond_ids,
                uncond_attention_mask=uncond_mask,
                image_size=spec.image_size,
                refine_mode=_resolve_refine_mode(sampling, self.model),
            )

        peak_mem_mb = self.peak_memory_mb()
        segment_extra = _segments_to_extra(result.segments)
        trajectory = build_ar_multisegment_trajectory(
            request=request,
            sample_specs=sample_specs,
            segments=segment_extra,
            decoded_outputs={
                "initial_image": result.initial_image,
                "final_image": result.final_image,
                "selfcheck": result.selfcheck,
            },
            primary_segment="final_image",
            context=result.context,
        )
        metrics = GenerationMetrics(
            num_prompts=len(prompts),
            num_samples=len(sample_specs),
            num_steps=_segment_token_steps(segment_extra),
            micro_batches=1,
            peak_memory_mb=peak_mem_mb,
        )

        return attach_engine_plan(OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=prompts,
            sample_specs=sample_specs,
            output=result.final_image,
            rollout_trajectory_data=None,
            trajectory=trajectory,
            extra={
                "initial_image": result.initial_image,
                "final_image": result.final_image,
                "selfcheck": result.selfcheck,
                "selfcheck_text": segment_extra["selfcheck_text"]["token_ids"],
                "segments": segment_extra,
                "context": result.context,
            },
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        ), engine_plan)

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> JanusProR1ChunkResult:
        from vrl.trainers.profiling import record_function

        self.validate_chunk(request, chunk)
        sampling = request.sampling
        spec: ARGenerationSpec = self.parse_spec(request)

        if spec.seed is not None:
            torch.manual_seed(spec.seed + self.chunk_seed_offset(request, chunk))

        with record_function("engine.prefill"):
            repeated_prompts = [chunk.prompt] * chunk.sample_count
            prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._tokenize_r1_prompts(
                repeated_prompts,
                max_text_length=spec.max_text_length,
            )

        with (
            record_function("engine.decode_step"),
            record_function("engine.cache_read"),
            record_function("engine.cache_write"),
        ):
            result = self.model.generate_with_refine(
                prompt_ids,
                prompt_mask,
                cfg_weight=float(sampling.get("cfg_weight", 5.0)),
                temperature=float(sampling.get("temperature", 1.0)),
                image_token_num=spec.image_token_num,
                max_reflect_len=int(sampling.get("max_reflect_len", 80)),
                task_stages=_parse_task_stages(sampling.get("task_stages")),
                uncond_input_ids=uncond_ids,
                uncond_attention_mask=uncond_mask,
                image_size=spec.image_size,
                refine_mode=_resolve_refine_mode(sampling, self.model),
            )

        return JanusProR1ChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=result.final_image,
            initial_image=result.initial_image,
            final_image=result.final_image,
            selfcheck=result.selfcheck,
            segments=result.segments,
            context=result.context,
            peak_memory_mb=self.peak_memory_mb(),
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[JanusProR1ChunkResult],
    ) -> OutputBatch:
        output = JanusProR1ChunkGatherer().gather_chunks(request, sample_specs, chunks)
        return attach_engine_plan(output, self.plan(request, list(sample_specs)))

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
        return self.align_pair(
            prompt_ids,
            prompt_mask,
            uncond_ids,
            uncond_mask,
            pad_id=pad_id,
        )


class JanusProR1ChunkGatherer:
    """Driver-side gatherer for Janus-Pro-R1 chunk payloads."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[JanusProR1ChunkResult],
    ) -> OutputBatch:
        ordered = ordered_chunks(
            request,
            sample_specs,
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
        trajectory = build_ar_multisegment_trajectory(
            request=request,
            sample_specs=list(sample_specs),
            segments=segment_extra,
            decoded_outputs={
                "initial_image": initial_image,
                "final_image": final_image,
                "selfcheck": selfcheck,
            },
            primary_segment="final_image",
            context=dict(ordered[0].context),
        )
        peak_mem_mb = self._max_peak_memory_mb(ordered)
        metrics = GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_specs),
            num_steps=_segment_token_steps(segment_extra),
            micro_batches=len(ordered),
            peak_memory_mb=peak_mem_mb,
        )

        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=output,
            rollout_trajectory_data=None,
            trajectory=trajectory,
            extra={
                "initial_image": initial_image,
                "final_image": final_image,
                "selfcheck": selfcheck,
                "selfcheck_text": segment_extra["selfcheck_text"]["token_ids"],
                "segments": segment_extra,
                "context": dict(ordered[0].context),
            },
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )

    @staticmethod
    def _max_peak_memory_mb(
        chunks: Sequence[JanusProR1ChunkResult],
    ) -> float | None:
        peaks = [chunk.peak_memory_mb for chunk in chunks if chunk.peak_memory_mb is not None]
        return max(peaks) if peaks else None


def _parse_task_stages(value: Any) -> tuple[str, ...]:
    if value is None:
        return R1_SEGMENT_NAMES
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


def _segment_to_extra(segment: JanusR1Segment) -> dict[str, Any]:
    return {
        "name": segment.name,
        "token_ids": segment.token_ids,
        "token_log_probs": segment.token_log_probs,
        "token_mask": segment.token_mask,
        "prompt_embeds": segment.prompt_embeds,
        "attention_mask": segment.attention_mask,
        "prompt_attention_mask": segment.attention_mask,
        "visual": segment.visual,
        "cfg": segment.cfg,
    }


def _segments_to_extra(
    segments: dict[str, JanusR1Segment],
) -> dict[str, dict[str, Any]]:
    return {name: _segment_to_extra(segment) for name, segment in segments.items()}


def _cat_segment_extra(
    chunks: Sequence[JanusProR1ChunkResult],
) -> dict[str, dict[str, Any]]:
    names = tuple(chunks[0].segments)
    if set(names) != set(R1_SEGMENT_NAMES):
        logger.warning("Unexpected Janus-Pro-R1 segment names: %s", names)

    out: dict[str, dict[str, Any]] = {}
    for name in names:
        first = chunks[0].segments[name]
        token_log_probs = None
        if first.token_log_probs is not None:
            token_log_probs = torch.cat(
                [chunk.segments[name].token_log_probs for chunk in chunks],
                dim=0,
            )
        out[name] = {
            "name": name,
            "token_ids": torch.cat(
                [chunk.segments[name].token_ids for chunk in chunks],
                dim=0,
            ),
            "token_log_probs": token_log_probs,
            "token_mask": torch.cat(
                [chunk.segments[name].token_mask for chunk in chunks],
                dim=0,
            ),
            "prompt_embeds": torch.cat(
                [chunk.segments[name].prompt_embeds for chunk in chunks],
                dim=0,
            ),
            "attention_mask": torch.cat(
                [chunk.segments[name].attention_mask for chunk in chunks],
                dim=0,
            ),
            "prompt_attention_mask": torch.cat(
                [chunk.segments[name].attention_mask for chunk in chunks],
                dim=0,
            ),
            "visual": first.visual,
            "cfg": first.cfg,
        }
    return out


def _segment_token_steps(segments: dict[str, dict[str, Any]]) -> int:
    return sum(int(segment["token_ids"].shape[1]) for segment in segments.values())


__all__ = [
    "JanusProARChunkResult",
    "JanusProChunkGatherer",
    "JanusProPipelineExecutor",
    "JanusProR1ChunkGatherer",
    "JanusProR1ChunkResult",
    "JanusProR1PipelineExecutor",
    "build_janus_pro_runtime_bundle",
    "extract_janus_pro_runtime_spec",
]
