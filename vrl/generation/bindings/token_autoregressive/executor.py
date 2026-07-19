"""Shared scaffolding for token-autoregressive generation executors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.bindings.token_autoregressive.layout import ARRequestLayout
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.protocols import GenerationChunkExecutor
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)


class ARChunkExecutorBase(
    GenerationChunkExecutor,
):
    """Base helpers for AR family executors.

    Owns the request-level plumbing (``plan`` / ``forward_plan``), mirroring
    ``DiffusionChunkExecutorBase``: the full-request
    path IS the production chunk path plus the family gatherer, so there is a
    single trajectory/metrics assembly line. Subclasses still own tokenization
    details, sampling math, decoding, and family-specific output packing
    (``forward_chunk_plan`` + their chunk gatherer).
    """

    family: str
    task: str
    model: Any
    default_image_token_num: int | None = None
    default_image_size: int | None = None
    default_max_text_length: int | None = None
    # AR runner wiring shared by families: subclasses declare their runner
    # class and the attention-backend registry key. The key is the model
    # architecture family, which is why it is separate from ``family`` (e.g.
    # janus_pro_r1 rolls out with the janus_pro architecture/backend).
    _runner_cls: type | None = None
    _runner_attention_family: str | None = None

    @property
    def layout(self) -> ARRequestLayout:
        return ARRequestLayout(
            default_image_token_num=self.default_image_token_num,
            default_image_size=self.default_image_size,
            default_max_text_length=self.default_max_text_length,
        )

    # -- request-level plumbing (shared; families own the chunk step) ----

    def plan(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
    ) -> Any:
        from vrl.generation.execution.planner import build_engine_plan

        return build_engine_plan(request)

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        engine_plan: Any,
    ) -> GenerationOutput:
        """Full-request path: the production chunk path plus the gatherer.

        Runs every planned chunk through ``forward_chunk_plan`` (with the same
        OOM-split retry the diffusion base uses) and assembles the output with
        the family chunk gatherer — the exact objects the per-chunk Ray
        dispatch produces and gathers, so this path cannot drift from
        production the way the old hand-rolled full-batch implementations did.
        """
        from vrl.generation.execution.chunks import run_sample_chunks_with_oom_retry

        chunks = run_sample_chunks_with_oom_retry(
            engine_plan.chunks,
            lambda chunk: self.forward_chunk_plan(request, chunk),
        )
        return self.gather_chunks(request, list(sample_rows), chunks)

    def _ar_runner(self, request: GenerationRequest) -> Any:
        """Build the family AR runner with the attention backend wired."""
        from vrl.nn.modules.ar_attention_backends import (
            attention_backend_name,
            resolve_attention_backend,
        )

        if self._runner_cls is None or self._runner_attention_family is None:
            raise RuntimeError(
                f"{type(self).__name__} must declare _runner_cls and _runner_attention_family",
            )
        sampling = request.sampling
        return self._runner_cls(
            self.model,
            attention_backend=resolve_attention_backend(
                self._runner_attention_family,
                attention_backend_name(sampling),
                self.model,
                block_size=int(sampling.get("ar_paged_block_size", 16)),
                cache_dtype=str(sampling.get("ar_paged_cache_dtype", "auto")),
            ),
        )

    def _embed(self, token_ids: Any) -> Any:
        """Token ids -> language-model input embeddings (same across families)."""
        embed = self.model.language_model.get_input_embeddings()
        return embed(token_ids)

    @staticmethod
    def _align_tokenizer_output(
        ids: Any,
        mask: Any,
        *,
        max_text_length: int,
        pad_id: int,
    ) -> tuple[Any, Any]:
        """Right-pad ids/mask to ``max_text_length``.

        Belt-and-braces shared by family tokenizers: enforce the length even
        if the tokenizer ignored ``padding="max_length"`` (stubs / tokenizers
        without a pad_token).
        """
        import torch

        if ids.shape[1] < max_text_length:
            extra_len = max_text_length - ids.shape[1]
            ids = torch.cat(
                [ids, torch.full((ids.shape[0], extra_len), pad_id, dtype=ids.dtype)],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], extra_len), dtype=mask.dtype)],
                dim=1,
            )
        return ids, mask

    def require_native_ar_engine(self, request: GenerationRequest) -> str:
        """Reject unsupported full-engine AR selectors before native parity runs."""

        engine = str(request.sampling.get("ar_engine", "native"))
        if engine == "native":
            return engine
        if engine == "vllm":
            raise ValueError(
                "request.sampling.ar_engine='vllm' is not a supported full-engine "
                "backend. AR paged attention is wired inside family runners, not "
                "through a vLLM LLMEngine adapter.",
            )
        raise ValueError("request.sampling.ar_engine must be 'native' if set")


@dataclass(slots=True)
class ARChunkInputs:
    """Family-prepared inputs for one discrete AR sample chunk.

    ``prepare_chunk_inputs`` (the one required hook of
    ``ARDiscreteChunkExecutorBase``) returns this: everything the shared chunk
    skeleton needs that only the family knows — parsed sampling knobs baked
    into decode-loop wiring, encoded prompt tensors, and the trajectory/replay
    context.
    """

    max_new_tokens: int
    # str(cond_embeds.dtype) — the decode loop's activation dtype source.
    decode_dtype: str
    init_args: tuple[Any, ...]
    init_kwargs: dict[str, Any]
    # Passed verbatim to ``model.decode_image_tokens(token_ids, **kwargs)``.
    image_decode_kwargs: dict[str, Any]
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    # Families without a real uncond branch (glm_image, llamagen) pass zeros
    # here to satisfy the shared discrete trajectory schema; replay never
    # reads them.
    uncond_input_ids: torch.Tensor
    uncond_attention_mask: torch.Tensor
    context: dict[str, Any]


@dataclass(slots=True)
class ARDiscreteChunkResult:
    """Output of one prompt/sample discrete AR chunk (shared by all discrete
    families — the field set janus_pro/emu3/glm_image/llamagen previously
    declared verbatim per family)."""

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
    # Display/provenance-only: emitted through per-chunk runtime debug metrics.
    peak_memory_mb: float | None = None


class ARDiscreteChunkExecutorBase(ARChunkExecutorBase):
    """Chunk-step template for discrete-token AR families.

    Owns the skeleton every discrete family previously copied verbatim
    (validate -> seed -> prefill -> ``TokenAutoregressiveLoop`` -> VQ decode -> token
    mask -> chunk result). Families implement ``prepare_chunk_inputs`` — the
    readable straight-line part: knob parsing, prompt encoding, decode-loop
    wiring — and may override ``chunk_token_mask`` (emu3 masks its forced
    structural positions).

    Families whose chunk step has a different shape stay off this template on
    the plain ``ARChunkExecutorBase``: nextstep_1 (continuous tokens, 3-tuple
    finalized decode payload) and janus_pro_r1 (inverted control flow through
    ``model.generate_with_refine``).
    """

    def prepare_chunk_inputs(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> ARChunkInputs:
        raise NotImplementedError

    def chunk_token_mask(
        self,
        inputs: ARChunkInputs,
        token_ids: torch.Tensor,
        token_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Trainable-token mask; the default trains every generated position."""

        del inputs, token_ids
        return torch.ones_like(token_log_probs)

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> ARDiscreteChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        from vrl.generation.composition.token_autoregressive.token_loop import (
            TokenAutoregressiveLoop,
        )
        from vrl.utils.profiling import record_function

        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, chunk)

        seed = request.sampling.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed) + self.layout.chunk_seed_offset(request, chunk))

        with record_function("engine.prefill"):
            inputs = self.prepare_chunk_inputs(request, chunk)

        with (
            record_function("engine.decode_step"),
            record_function("engine.cache_read"),
            record_function("engine.cache_write"),
        ):
            decode_result = TokenAutoregressiveLoop(
                request=request,
                sample_rows=self.layout.chunk_sample_rows(request, chunk),
                runner=self._ar_runner(request),
                max_new_tokens=inputs.max_new_tokens,
                tokenizer_key=self.family,
                dtype=inputs.decode_dtype,
                scheduler_batch_size=chunk.sample_count,
                init_args=inputs.init_args,
                init_kwargs=inputs.init_kwargs,
            ).run()
        token_ids, token_log_probs = decode_result.finalized
        with record_function("engine.vq_decode"):
            images = self.model.decode_image_tokens(
                token_ids,
                **inputs.image_decode_kwargs,
            )
        token_mask = self.chunk_token_mask(inputs, token_ids, token_log_probs)

        return ARDiscreteChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=images,
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=inputs.prompt_input_ids,
            prompt_attention_mask=inputs.prompt_attention_mask,
            uncond_input_ids=inputs.uncond_input_ids,
            uncond_attention_mask=inputs.uncond_attention_mask,
            context=dict(inputs.context),
            peak_memory_mb=self.layout.peak_memory_mb(),
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ARDiscreteChunkResult],
    ) -> GenerationOutput:
        return ARDiscreteChunkGatherer().gather_chunks(request, sample_rows, chunks)


@dataclass(frozen=True, slots=True)
class ARDiscreteChunkGatherer:
    """Pure driver-side gatherer for discrete AR chunk payloads.

    One class for every discrete family (mirroring ``DiffusionChunkGatherer``
    on the diffusion side): the payload is the shared
    ``ARDiscreteChunkResult``, so nothing here is family-specific.
    """

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ARDiscreteChunkResult],
    ) -> GenerationOutput:
        """Pack prompt/sample AR chunks back into the canonical GenerationOutput."""

        from vrl.trajectory import build_ar_discrete_trajectory

        layout = ARRequestLayout()
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
        ordered_ar_chunks = layout.ordered_chunks(
            request,
            sample_rows,
            chunks,
            row_fields=fields,
        )
        cat = layout.cat_chunk_fields(ordered_ar_chunks, fields)
        chunk_context = dict(ordered_ar_chunks[0].context)
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
        )


__all__ = [
    "ARChunkExecutorBase",
    "ARChunkInputs",
    "ARDiscreteChunkExecutorBase",
    "ARDiscreteChunkGatherer",
    "ARDiscreteChunkResult",
]
