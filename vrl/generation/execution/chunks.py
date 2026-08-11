"""Sample chunk helpers for generation executors.

The chunk vocabulary both sides of the driver <-> worker split must agree on:
the driver plans and OOM-splits chunks, every family binding's gatherer
validates and reassembles them, and neither side may import the other. So the
chunk shape, the coverage/ordering checks, and the strict replay-merge
helpers live here in neutral ground — a gatherer bug and a planner bug would
otherwise disagree silently about what one chunk means.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from vrl.utils.cuda_memory import empty_cuda_cache, is_cuda_out_of_memory

if TYPE_CHECKING:
    from vrl.generation.types import GenerationRequest, GenerationSampleRow

# The generation facade reaches this module while parsing config. Tensor-only
# helpers import torch at call time so config resolution remains torch-free.


@dataclass(frozen=True, slots=True)
class SampleAlignedValues:
    """Explicit sample-axis wrapper for ragged Python replay values."""

    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("SampleAlignedValues.values must be non-empty")


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


def concatenate_sample_values(values: Sequence[Any], *, name: str) -> Any:
    """Concatenate one sample-aligned value from each ordered chunk."""

    import torch

    if not values:
        raise ValueError(f"cannot concatenate empty field {name!r}")
    if all(isinstance(value, torch.Tensor) for value in values):
        return torch.cat(list(values), dim=0)
    if all(isinstance(value, list) for value in values):
        return [item for value in values for item in value]
    if all(isinstance(value, tuple) for value in values):
        return tuple(item for value in values for item in value)
    raise TypeError(f"chunk field {name!r} must use one consistent concatenable type")


def gather_replay_tensors(
    replay_mappings: Sequence[Mapping[str, Any]],
    *,
    sample_counts: Sequence[int],
) -> dict[str, Any]:
    """Strictly merge sample-aligned and static replay values across chunks."""

    import torch

    if not replay_mappings:
        raise ValueError("replay_mappings must be non-empty")
    if len(sample_counts) != len(replay_mappings) or any(count < 1 for count in sample_counts):
        raise ValueError("sample_counts must provide one positive count per replay mapping")
    keys = tuple(replay_mappings[0])
    expected_keys = set(keys)
    if any(set(mapping) != expected_keys for mapping in replay_mappings[1:]):
        raise ValueError("replay_tensors keys must match across chunk results")

    gathered: dict[str, Any] = {}
    for key in keys:
        values = [mapping[key] for mapping in replay_mappings]
        if all(value is None for value in values):
            gathered[key] = None
        elif any(value is None for value in values):
            raise ValueError(f"replay tensor {key!r} must be present on all results")
        elif all(isinstance(value, torch.Tensor) for value in values):
            gathered[key] = concatenate_sample_values(
                values,
                name=f"replay_tensors.{key}",
            )
        elif all(isinstance(value, SampleAlignedValues) for value in values):
            aligned_values = [value.values for value in values]
            if any(
                len(value) != sample_count
                for value, sample_count in zip(aligned_values, sample_counts, strict=True)
            ):
                raise ValueError(
                    f"sample-aligned replay value {key!r} must match each chunk sample_count",
                )
            gathered[key] = concatenate_sample_values(
                aligned_values,
                name=f"replay_tensors.{key}",
            )
        elif any(isinstance(value, SampleAlignedValues) for value in values):
            raise TypeError(
                f"replay value {key!r} must use SampleAlignedValues on all results",
            )
        elif all(_values_match(value, values[0]) for value in values[1:]):
            gathered[key] = values[0]
        else:
            raise ValueError(f"non-batched replay value {key!r} must match across results")
    return gathered


def require_matching_chunk_context(
    contexts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return shared chunk context after checking every value matches."""

    if not contexts:
        raise ValueError("chunk contexts must be non-empty")
    first = contexts[0]
    for index, context in enumerate(contexts[1:], start=1):
        if not _values_match(context, first):
            raise ValueError(f"chunk context at ordered index {index} does not match")
    return dict(first)


def _values_match(left: Any, right: Any) -> bool:
    """Compare nested static replay/context values without tensor truth coercion."""

    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.device == right.device
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_values_match(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_values_match(a, b) for a, b in zip(left, right, strict=True))
        )
    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):
        return False


class ChunkResultWithIdentity(Protocol):
    """Chunk result carrying the authoritative source chunk."""

    chunk: SampleChunk


def ordered_covering_chunks[TChunk: ChunkResultWithIdentity](
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
        key=lambda result: (
            int(result.chunk.prompt_index),
            int(result.chunk.sample_start),
        ),
    )
    expected = [(row.prompt_index, row.sample_index) for row in sample_rows]
    actual: list[tuple[int, int]] = []
    for result in ordered:
        chunk = result.chunk
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
            _require_rows(field_name, getattr(result, field_name), sample_count)
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
            sample_start=self.sample_start,
            sample_count=left_count,
        )
        right = SampleChunk(
            prompt_index=self.prompt_index,
            sample_start=self.sample_start + left_count,
            sample_count=right_count,
        )
        return left, right


def build_prompt_chunks(
    prompt_count: int,
    samples_per_prompt: int,
    max_samples_per_chunk: int,
) -> tuple[SampleChunk, ...]:
    """Plan prompt-major sample chunks without changing RL group semantics."""

    if prompt_count < 1:
        raise ValueError("prompt_count must be >= 1")
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    if max_samples_per_chunk < 1:
        raise ValueError("max_samples_per_chunk must be >= 1")

    chunks: list[SampleChunk] = []
    for prompt_index in range(prompt_count):
        sample_start = 0
        remaining = samples_per_prompt
        while remaining > 0:
            sample_count = min(max_samples_per_chunk, remaining)
            chunks.append(
                SampleChunk(
                    prompt_index=prompt_index,
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
    "SampleAlignedValues",
    "SampleChunk",
    "build_prompt_chunks",
    "concatenate_sample_values",
    "gather_replay_tensors",
    "ordered_covering_chunks",
    "require_matching_chunk_context",
    "run_sample_chunks_with_oom_retry",
    "validate_chunk_range",
]
