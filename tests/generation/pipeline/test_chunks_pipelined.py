"""forward_chunks_pipelined is BIT-EXACT to serial produce/teardown — the
pipelined (overlapped) path returns the same per-chunk results in chunk order as
running each chunk serially. The side-stream copy only changes WHEN the teardown
runs, never WHAT it produces; chunks are independent."""

from __future__ import annotations

from vrl.generation.diffusion.pipeline import forward_chunks_pipelined


def _serial(produce, teardown, chunks):
    return [teardown(produce(c)) for c in chunks]


def test_pipelined_results_equal_serial_in_chunk_order() -> None:
    # produce: deterministic per-chunk payload; teardown: identity (stand-in for
    # the GPU->CPU copy, which is value-preserving).
    def produce(chunk):
        return ("denoised", chunk)

    def teardown(result):
        return ("on_cpu", result)

    pipelined = forward_chunks_pipelined(
        object(), "req", ["c0", "c1", "c2", "c3"],
        produce_fn=produce, teardown_fn=teardown,
    )
    serial = _serial(produce, teardown, ["c0", "c1", "c2", "c3"])

    assert pipelined == serial
    # Order preserved by index, not completion order.
    assert pipelined == [
        ("on_cpu", ("denoised", "c0")),
        ("on_cpu", ("denoised", "c1")),
        ("on_cpu", ("denoised", "c2")),
        ("on_cpu", ("denoised", "c3")),
    ]


def test_every_chunk_produced_exactly_once_and_torn_down_exactly_once() -> None:
    produced: list = []
    torn: list = []

    def produce(chunk):
        produced.append(chunk)
        return ("r", chunk)

    def teardown(result):
        torn.append(result[1])
        return result

    chunks = [f"c{i}" for i in range(5)]
    out = forward_chunks_pipelined(object(), "req", chunks, produce_fn=produce, teardown_fn=teardown)

    assert produced == chunks  # produced in order, once each
    assert sorted(torn) == sorted(chunks)  # torn down once each (order may interleave)
    assert len(out) == len(chunks)


def test_single_chunk_still_produces_and_tears_down() -> None:
    out = forward_chunks_pipelined(
        object(), "req", ["only"],
        produce_fn=lambda c: ("p", c), teardown_fn=lambda r: ("t", r),
    )
    assert out == [("t", ("p", "only"))]


def test_empty_chunks_returns_empty() -> None:
    out = forward_chunks_pipelined(
        object(), "req", [], produce_fn=lambda c: c, teardown_fn=lambda r: r,
    )
    assert out == []
