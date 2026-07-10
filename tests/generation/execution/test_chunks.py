"""Tests for generation executor sample chunk helpers."""

from __future__ import annotations


def test_build_prompt_chunks_prompt_major() -> None:
    """Checks build prompt chunks prompt major."""
    from vrl.generation.execution.chunks import build_prompt_chunks

    chunks = build_prompt_chunks(
        ["a", "b"],
        samples_per_prompt=5,
        max_samples_per_chunk=2,
    )

    got = [
        (chunk.prompt_index, chunk.prompt, chunk.sample_start, chunk.sample_count)
        for chunk in chunks
    ]
    assert got == [
        (0, "a", 0, 2),
        (0, "a", 2, 2),
        (0, "a", 4, 1),
        (1, "b", 0, 2),
        (1, "b", 2, 2),
        (1, "b", 4, 1),
    ]


def test_run_sample_chunks_with_oom_retry_splits_until_success() -> None:
    """Checks run sample chunks with oom retry splits until success."""
    from vrl.generation.execution.chunks import (
        SampleChunk,
        run_sample_chunks_with_oom_retry,
    )

    seen: list[tuple[int, int]] = []

    def run_one(chunk: SampleChunk) -> int:
        seen.append((chunk.sample_start, chunk.sample_count))
        if chunk.sample_count > 2:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return chunk.sample_count

    results = run_sample_chunks_with_oom_retry(
        [
            SampleChunk(
                prompt_index=0,
                prompt="a",
                sample_start=0,
                sample_count=5,
            )
        ],
        run_one,
    )

    assert results == [2, 1, 2]
    assert seen == [(0, 5), (0, 2), (2, 3), (2, 1), (3, 2)]
