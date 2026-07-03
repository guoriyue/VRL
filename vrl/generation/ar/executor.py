"""Shared scaffolding for autoregressive generation executors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vrl.generation.ar.layout import ARRequestLayout
from vrl.generation.capabilities import FamilyCapability
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

    Owns the request-level plumbing (``plan`` / ``forward_plan`` /
    ``capability``), mirroring ``DiffusionChunkExecutorBase``: the full-request
    path IS the production chunk path plus the family gatherer, so there is a
    single trajectory/metrics assembly line. Subclasses still own tokenization
    details, sampling math, decoding, and family-specific output packing
    (``forward_chunk_plan`` + their chunk gatherer).
    """

    family: str
    task: str
    model: Any
    family_capability: FamilyCapability | None = None
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

    def capability(self) -> FamilyCapability:
        if self.family_capability is None:
            raise RuntimeError(
                f"{type(self).__name__} must declare family_capability explicitly",
            )
        return self.family_capability

    def plan(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
    ) -> Any:
        from vrl.generation.execution.planner import build_engine_plan

        return build_engine_plan(
            request,
            list(sample_rows),
            capability=self.capability(),
        )

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
        from vrl.generation.execution.planner import attach_engine_plan

        chunks = run_sample_chunks_with_oom_retry(
            engine_plan.chunks,
            lambda chunk: self.forward_chunk_plan(
                request,
                chunk,
                engine_plan.chunk_stage_for(chunk),
            ),
        )
        return attach_engine_plan(
            self.gather_chunks(request, list(sample_rows), chunks),
            engine_plan,
        )

    def _ar_runner(self, request: GenerationRequest) -> Any:
        """Build the family AR runner with the attention backend wired."""
        from vrl.nn.modules.ar_attention_backends import (
            attention_backend_name,
            resolve_attention_backend,
        )

        if self._runner_cls is None or self._runner_attention_family is None:
            raise RuntimeError(
                f"{type(self).__name__} must declare _runner_cls and "
                "_runner_attention_family",
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


__all__ = [
    "ARChunkExecutorBase",
]
