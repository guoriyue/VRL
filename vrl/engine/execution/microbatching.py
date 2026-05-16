"""Micro-batching helpers for generation executors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from vrl.utils.cuda_memory import empty_cuda_cache, is_cuda_out_of_memory

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MicroBatchSample:
    """One executor micro-batch for a prompt-major rollout request."""

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

    def split(self) -> tuple[MicroBatchSample, MicroBatchSample]:
        """Split this micro-batch into two ordered smaller chunks."""

        if self.sample_count <= 1:
            raise ValueError("Cannot split a single-sample micro-batch")
        left_count = self.sample_count // 2
        right_count = self.sample_count - left_count
        left = MicroBatchSample(
            prompt_index=self.prompt_index,
            prompt=self.prompt,
            sample_start=self.sample_start,
            sample_count=left_count,
        )
        right = MicroBatchSample(
            prompt_index=self.prompt_index,
            prompt=self.prompt,
            sample_start=self.sample_start + left_count,
            sample_count=right_count,
        )
        return left, right


@dataclass(frozen=True, slots=True)
class MicroBatchSchedule:
    """Prompt-major micro-batch schedule for one GenerationRequest."""

    prompts: tuple[str, ...]
    samples_per_prompt: int
    max_samples_per_microbatch: int
    micro_batches: tuple[MicroBatchSample, ...]
    trajectory_kind: str | None = None
    batchable_axes: tuple[str, ...] = ()

    @property
    def total_samples(self) -> int:
        return len(self.prompts) * self.samples_per_prompt


def build_prompt_microbatch_schedule(
    prompts: Sequence[str],
    samples_per_prompt: int,
    max_samples_per_microbatch: int,
    *,
    capability: Any | None = None,
) -> MicroBatchSchedule:
    """Plan prompt-major micro-batches without changing RL group semantics."""

    if not prompts:
        raise ValueError("prompts must be non-empty")
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    if max_samples_per_microbatch < 1:
        raise ValueError("max_samples_per_microbatch must be >= 1")

    if capability is not None and not bool(
        getattr(capability, "supports_chunked_execution", True),
    ):
        max_samples_per_microbatch = samples_per_prompt

    micro_batches: list[MicroBatchSample] = []
    for prompt_index, prompt in enumerate(prompts):
        sample_start = 0
        remaining = samples_per_prompt
        while remaining > 0:
            sample_count = min(max_samples_per_microbatch, remaining)
            micro_batches.append(
                MicroBatchSample(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    sample_start=sample_start,
                    sample_count=sample_count,
                )
            )
            sample_start += sample_count
            remaining -= sample_count

    return MicroBatchSchedule(
        prompts=tuple(prompts),
        samples_per_prompt=samples_per_prompt,
        max_samples_per_microbatch=max_samples_per_microbatch,
        micro_batches=tuple(micro_batches),
        trajectory_kind=getattr(capability, "trajectory_kind", None),
        batchable_axes=tuple(getattr(capability, "batchable_axes", ())),
    )


def run_microbatch_samples_with_oom_retry(
    sample_batches: Sequence[MicroBatchSample],
    run_one: Callable[[MicroBatchSample], T],
    *,
    min_sample_count: int = 1,
) -> list[T]:
    """Run micro-batches, splitting CUDA-OOM chunks until the floor is reached."""

    if min_sample_count < 1:
        raise ValueError("min_sample_count must be >= 1")

    results: list[T] = []
    pending = list(sample_batches)
    while pending:
        micro_batch = pending.pop(0)
        try:
            results.append(run_one(micro_batch))
        except RuntimeError as exc:
            if (
                not is_cuda_out_of_memory(exc)
                or micro_batch.sample_count <= min_sample_count
            ):
                raise
            empty_cuda_cache()
            left, right = micro_batch.split()
            pending.insert(0, right)
            pending.insert(0, left)
    return results


__all__ = [
    "MicroBatchSample",
    "MicroBatchSchedule",
    "build_prompt_microbatch_schedule",
    "run_microbatch_samples_with_oom_retry",
]
