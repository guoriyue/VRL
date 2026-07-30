"""Sample chunk helpers for generation executors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.utils.cuda_memory import empty_cuda_cache, is_cuda_out_of_memory

if TYPE_CHECKING:
    from vrl.generation.types import GenerationRequest, GenerationSampleRow


def validate_chunk_range(
    request: GenerationRequest,
    *,
    prompt_index: int,
    sample_start: int,
    sample_count: int,
) -> None:
    """Validate a chunk's prompt/sample range against its source request."""

    if prompt_index < 0 or prompt_index >= len(request.prompts):
        raise ValueError(f"chunk.prompt_index={prompt_index} is out of range")
    sample_end = sample_start + sample_count
    if sample_start < 0 or sample_count < 1:
        raise ValueError(
            "chunk sample range must have non-negative start and positive count",
        )
    if sample_end > request.samples_per_prompt:
        raise ValueError(
            "chunk sample range exceeds request.samples_per_prompt: "
            f"{sample_start}:{sample_end} > {request.samples_per_prompt}",
        )


def _require_rows(name: str, value: Any, count: int) -> None:
    """Require a chunk payload to have ``count`` leading batch rows.

    Accepts a tensor-like ``.shape`` or a plain list/tuple length so a gatherer
    can validate both concatenated tensors and python-sequence payloads. For
    tensors (the diffusion/AR case) this is identical to a strict ``shape[0]``
    check; the list/tuple branch is the chunk-AR superset.
    """

    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        actual = int(shape[0])
    elif isinstance(value, (list, tuple)):
        actual = len(value)
    else:
        raise ValueError(f"chunk {name} must have a leading batch dimension")
    if actual != count:
        raise ValueError(f"chunk {name} has {actual} rows, expected {count}")


def ordered_covering_chunks[TChunk](
    request: GenerationRequest,
    sample_rows: Sequence[GenerationSampleRow],
    chunks: Sequence[TChunk],
    *,
    row_fields: Sequence[str] = (),
) -> list[TChunk]:
    """Sort prompt-major chunks and check they exactly cover ``sample_rows``.

    The sort + coverage skeleton every chunk gatherer shares: prompt-major sort,
    per-chunk range validation, optional row-count checks on ``row_fields``, and
    an exact prompt-major coverage match against the request's sample rows.
    Families layer their own homogeneity checks over the returned ordered list.
    """

    if not chunks:
        raise ValueError("chunks must be non-empty")
    ordered = sorted(
        chunks,
        key=lambda chunk: (int(chunk.prompt_index), int(chunk.sample_start)),
    )
    expected = [(row.prompt_index, row.sample_index) for row in sample_rows]
    actual: list[tuple[int, int]] = []
    for chunk in ordered:
        prompt_index = int(chunk.prompt_index)
        sample_start = int(chunk.sample_start)
        sample_count = int(chunk.sample_count)
        validate_chunk_range(
            request,
            prompt_index=prompt_index,
            sample_start=sample_start,
            sample_count=sample_count,
        )
        for field_name in row_fields:
            _require_rows(field_name, getattr(chunk, field_name), sample_count)
        actual.extend(
            (prompt_index, sample_index)
            for sample_index in range(sample_start, sample_start + sample_count)
        )
    if actual != expected:
        raise ValueError(
            "chunks do not cover sample_rows in prompt-major order",
        )
    return ordered


@dataclass(frozen=True, slots=True)
class SampleChunk:
    """One prompt-major sample chunk for a generation request."""

    prompt_index: int
    prompt: str
    sample_start: int
    sample_count: int

    def __post_init__(self) -> None:
        if self.prompt_index < 0:
            raise ValueError("prompt_index must be >= 0")
        if self.sample_start < 0:
            raise ValueError("sample_start must be >= 0")
        if self.sample_count < 1:
            raise ValueError("sample_count must be >= 1")

    @property
    def sample_end(self) -> int:
        return self.sample_start + self.sample_count

    @property
    def chunk_key(self) -> str:
        """Stable key used by retry, telemetry, and chunk-result joins."""

        return f"prompt:{self.prompt_index}:samples:{self.sample_start}:{self.sample_end}"

    def split(self) -> tuple[SampleChunk, SampleChunk]:
        """Split this chunk into two ordered smaller chunks."""

        if self.sample_count <= 1:
            raise ValueError("Cannot split a single-sample chunk")
        left_count = self.sample_count // 2
        right_count = self.sample_count - left_count
        left = SampleChunk(
            prompt_index=self.prompt_index,
            prompt=self.prompt,
            sample_start=self.sample_start,
            sample_count=left_count,
        )
        right = SampleChunk(
            prompt_index=self.prompt_index,
            prompt=self.prompt,
            sample_start=self.sample_start + left_count,
            sample_count=right_count,
        )
        return left, right


def build_prompt_chunks(
    prompts: Sequence[str],
    samples_per_prompt: int,
    max_samples_per_chunk: int,
) -> tuple[SampleChunk, ...]:
    """Plan prompt-major sample chunks without changing RL group semantics."""

    if not prompts:
        raise ValueError("prompts must be non-empty")
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    if max_samples_per_chunk < 1:
        raise ValueError("max_samples_per_chunk must be >= 1")

    chunks: list[SampleChunk] = []
    for prompt_index, prompt in enumerate(prompts):
        sample_start = 0
        remaining = samples_per_prompt
        while remaining > 0:
            sample_count = min(max_samples_per_chunk, remaining)
            chunks.append(
                SampleChunk(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    sample_start=sample_start,
                    sample_count=sample_count,
                )
            )
            sample_start += sample_count
            remaining -= sample_count

    return tuple(chunks)


def run_sample_chunks_with_oom_retry[T](
    chunks: Sequence[SampleChunk],
    run_one: Callable[[SampleChunk], T],
) -> list[T]:
    """Run chunks, splitting CUDA-OOM chunks until the floor is reached."""

    results: list[T] = []
    pending = list(chunks)
    while pending:
        chunk = pending.pop(0)
        try:
            results.append(run_one(chunk))
        except RuntimeError as exc:
            # A single-sample chunk cannot be split further; re-raise the
            # original OOM instead of letting split() mask it with ValueError.
            if not is_cuda_out_of_memory(exc) or chunk.sample_count <= 1:
                raise
            empty_cuda_cache()
            left, right = chunk.split()
            pending.insert(0, right)
            pending.insert(0, left)
    return results


__all__ = [
    "SampleChunk",
    "build_prompt_chunks",
    "ordered_covering_chunks",
    "run_sample_chunks_with_oom_retry",
    "validate_chunk_range",
]
