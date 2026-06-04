"""Sample chunk helpers for generation executors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from vrl.utils.cuda_memory import empty_cuda_cache, is_cuda_out_of_memory

if TYPE_CHECKING:
    from vrl.generation.types import GenerationRequest

T = TypeVar("T")


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
        """Stable key used to join chunk assignments with engine plan units."""

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


@dataclass(frozen=True, slots=True)
class SampleChunkSchedule:
    """Prompt-major chunk schedule for one GenerationRequest."""

    prompts: tuple[str, ...]
    samples_per_prompt: int
    max_samples_per_chunk: int
    chunks: tuple[SampleChunk, ...]
    trajectory_kind: str | None = None
    batchable_axes: tuple[str, ...] = ()

    @property
    def total_samples(self) -> int:
        return len(self.prompts) * self.samples_per_prompt


def build_prompt_chunk_schedule(
    prompts: Sequence[str],
    samples_per_prompt: int,
    max_samples_per_chunk: int,
    *,
    capability: Any | None = None,
) -> SampleChunkSchedule:
    """Plan prompt-major sample chunks without changing RL group semantics."""

    if not prompts:
        raise ValueError("prompts must be non-empty")
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    if max_samples_per_chunk < 1:
        raise ValueError("max_samples_per_chunk must be >= 1")

    if capability is not None and not bool(
        getattr(capability, "supports_chunked_execution", True),
    ):
        max_samples_per_chunk = samples_per_prompt

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

    return SampleChunkSchedule(
        prompts=tuple(prompts),
        samples_per_prompt=samples_per_prompt,
        max_samples_per_chunk=max_samples_per_chunk,
        chunks=tuple(chunks),
        trajectory_kind=getattr(capability, "trajectory_kind", None),
        batchable_axes=tuple(getattr(capability, "batchable_axes", ())),
    )


def run_sample_chunks_with_oom_retry(
    chunks: Sequence[SampleChunk],
    run_one: Callable[[SampleChunk], T],
    *,
    min_sample_count: int = 1,
) -> list[T]:
    """Run chunks, splitting CUDA-OOM chunks until the floor is reached."""

    if min_sample_count < 1:
        raise ValueError("min_sample_count must be >= 1")

    results: list[T] = []
    pending = list(chunks)
    while pending:
        chunk = pending.pop(0)
        try:
            results.append(run_one(chunk))
        except RuntimeError as exc:
            if (
                not is_cuda_out_of_memory(exc)
                or chunk.sample_count <= min_sample_count
            ):
                raise
            empty_cuda_cache()
            left, right = chunk.split()
            pending.insert(0, right)
            pending.insert(0, left)
    return results


__all__ = [
    "SampleChunk",
    "SampleChunkSchedule",
    "build_prompt_chunk_schedule",
    "run_sample_chunks_with_oom_retry",
    "validate_chunk_range",
]
