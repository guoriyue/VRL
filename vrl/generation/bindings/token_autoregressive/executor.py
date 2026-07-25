"""Shared scaffolding for token-autoregressive generation executors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.bindings.token_autoregressive.layout import ARRequestLayout, right_pad
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.executor_base import ChunkExecutorBase
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.utils.cuda_memory import cuda_peak_allocated_mb


class ARChunkExecutorBase(ChunkExecutorBase):
    """Base helpers for AR family executors.

    Owns the request-level plumbing (``plan`` plus the inherited
    ``forward_plan``), mirroring ``DiffusionChunkExecutorBase``: the
    full-request path IS the production chunk path plus the family gatherer, so
    there is a single trajectory/metrics assembly line. Subclasses still own
    tokenization details, sampling math, decoding, and family-specific output
    packing (``forward_chunk_plan`` + their registered chunk gatherer).
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
    # Set by families whose runner drives its own KV cache and therefore takes
    # no shared attention backend. The text completes "<family> does not
    # support request.sampling.attention_backend=<backend>: <reason>", so an
    # explicit backend request is rejected instead of silently ignored.
    _native_runner_reason: str | None = None

    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def layout(self) -> ARRequestLayout:
        return ARRequestLayout(
            default_image_token_num=self.default_image_token_num,
            default_image_size=self.default_image_size,
            default_max_text_length=self.default_max_text_length,
        )

    def resolve_scheduler_batch_size(
        self,
        request: GenerationRequest,
        *,
        row_count: int,
    ) -> int | None:
        """Resolve the row bound at the executor policy boundary.

        ``row_count`` is available for family compatibility checks; the shared
        request-local scheduler itself accepts any positive bound.
        """

        del row_count
        return self.layout.resolve_scheduler_batch_size(request)

    # -- request-level plumbing (shared; families own the chunk step) ----

    def plan(
        self,
        request: GenerationRequest,
    ) -> Any:
        from vrl.generation.execution.planner import build_engine_plan

        return build_engine_plan(request)

    def _ar_runner(self, request: GenerationRequest) -> Any:
        """Build the family AR runner with the attention backend wired."""

        if self._runner_cls is None:
            raise RuntimeError(f"{type(self).__name__} must declare _runner_cls")
        sampling = request.sampling
        if self._native_runner_reason is not None:
            backend = sampling.get("attention_backend")
            if backend is not None:
                raise ValueError(
                    f"{self.family} does not support request.sampling."
                    f"attention_backend={backend!r}: {self._native_runner_reason}",
                )
            return self._runner_cls(self.model)

        from vrl.nn.modules.ar_attention_backends import (
            attention_backend_name,
            resolve_attention_backend,
        )

        if self._runner_attention_family is None:
            raise RuntimeError(
                f"{type(self).__name__} must declare _runner_attention_family",
            )
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

        return right_pad(ids, mask, target_length=max_text_length, pad_id=pad_id)

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
        from vrl.utils.profiling import profile_range

        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, chunk)
        scheduler_batch_size = self.resolve_scheduler_batch_size(
            request,
            row_count=chunk.sample_count,
        )

        seed = request.sampling.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed) + self.layout.chunk_seed_offset(request, chunk))

        with profile_range("engine.prefill"):
            inputs = self.prepare_chunk_inputs(request, chunk)

        with (
            profile_range("engine.decode_step"),
            profile_range("engine.cache_read"),
            profile_range("engine.cache_write"),
        ):
            token_ids, token_log_probs = TokenAutoregressiveLoop(
                runner=self._ar_runner(request),
                scheduler_batch_size=scheduler_batch_size,
                init_args=inputs.init_args,
                init_kwargs=inputs.init_kwargs,
            ).run()
        with profile_range("engine.vq_decode"):
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
            peak_memory_mb=cuda_peak_allocated_mb(),
        )


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
