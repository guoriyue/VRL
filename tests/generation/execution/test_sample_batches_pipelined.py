"""forward_batches_pipelined is BIT-EXACT to serial produce/teardown — the
pipelined (overlapped) path returns the same per-batch results in batch order as
running each batch serially. The side-stream copy only changes WHEN the teardown
runs, never WHAT it produces; batches are independent."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from vrl.generation.bindings.full_sequence_denoise.executor import DiffusionBatchResult
from vrl.generation.execution.pipeline import (
    _move_tree_to_cpu_async,
    forward_batches_pipelined,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import BatchProduceFence


def _executor(produce):
    return SimpleNamespace(
        forward_batch=lambda _request, batch: produce(batch),
    )


def test_pipelined_results_equal_serial_in_chunk_order() -> None:
    def produce(batch):
        return ("denoised", batch)

    pipelined = forward_batches_pipelined(
        _executor(produce),
        "req",
        ["c0", "c1", "c2", "c3"],
    )
    serial = [produce(batch) for batch in ["c0", "c1", "c2", "c3"]]

    assert pipelined == serial
    # Order preserved by index, not completion order.
    assert pipelined == [
        ("denoised", "c0"),
        ("denoised", "c1"),
        ("denoised", "c2"),
        ("denoised", "c3"),
    ]


def test_every_chunk_produced_exactly_once() -> None:
    produced: list = []

    def produce(batch):
        produced.append(batch)
        return ("r", batch)

    batches = [f"c{i}" for i in range(5)]
    out = forward_batches_pipelined(_executor(produce), "req", batches)

    assert produced == batches  # produced in order, once each
    assert len(out) == len(batches)


def test_cpu_completion_fence_follows_each_successful_chunk(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    fences: list[BatchProduceFence] = []

    output = forward_batches_pipelined(
        _executor(lambda batch: ("result", batch)),
        "req",
        ["c0", "c1", "c2"],
        completion_callback=fences.append,
    )

    assert output == [("result", "c0"), ("result", "c1"), ("result", "c2")]
    assert [fence.completed_batches for fence in fences] == [1, 2, 3]
    assert all(fence.event is None and fence.query() for fence in fences)


def test_cuda_completion_fence_is_recorded_before_publication(monkeypatch) -> None:
    operations: list[tuple[str, int]] = []
    events: list[Any] = []

    class _Event:
        def __init__(self) -> None:
            self.index = len(events)
            self.recorded = False
            events.append(self)

        def record(self, _stream=None) -> None:
            self.recorded = True
            operations.append(("record", self.index))

        def query(self) -> bool:
            return False

        def synchronize(self) -> None:
            operations.append(("synchronize", self.index))

    class _CopyStream:
        @staticmethod
        def wait_event(_event: Any) -> None:
            return None

    copy_stream = _CopyStream()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            Stream=lambda: copy_stream,
            Event=_Event,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import vrl.generation.execution.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "_move_tree_to_cpu_async",
        lambda result, _stream: result,
    )
    published: list[BatchProduceFence] = []

    def publish(fence: BatchProduceFence) -> None:
        assert fence.event is not None
        assert fence.event.recorded is True
        operations.append(("publish", fence.event.index))
        published.append(fence)

    assert forward_batches_pipelined(
        _executor(lambda batch: ("result", batch)),
        "req",
        ["c0"],
        completion_callback=publish,
    ) == [("result", "c0")]

    assert published == [BatchProduceFence(completed_batches=1, event=events[0])]
    assert published[0].query() is False
    assert operations[:2] == [("record", 0), ("publish", 0)]


def test_single_chunk_still_produces_and_tears_down() -> None:
    out = forward_batches_pipelined(
        _executor(lambda batch: ("p", batch)),
        "req",
        ["only"],
    )
    assert out == [("p", "only")]


def test_empty_chunks_returns_empty() -> None:
    out = forward_batches_pipelined(
        _executor(lambda batch: batch),
        "req",
        [],
    )
    assert out == []


def test_produce_error_joins_already_submitted_copy(monkeypatch) -> None:
    class _Event:
        def __init__(self) -> None:
            self.recorded_stream = None
            self.synchronize_calls = 0
            events.append(self)

        def record(self, stream=None) -> None:
            self.recorded_stream = stream

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    events: list[_Event] = []

    class _CopyStream:
        def __init__(self) -> None:
            self.waited_events: list[_Event] = []
            self.synchronize_calls = 0

        def wait_event(self, event) -> None:
            self.waited_events.append(event)

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    copy_stream = _CopyStream()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            Stream=lambda: copy_stream,
            Event=_Event,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    produced: list[str] = []
    torn_down: list[tuple[str, str]] = []

    def produce(batch: str) -> tuple[str, str]:
        produced.append(batch)
        if batch == "c1":
            raise RuntimeError("CUDA out of memory in second batch")
        return ("result", batch)

    def teardown(result: tuple[str, str], _stream) -> tuple[str, str]:
        torn_down.append(result)
        return result

    import vrl.generation.execution.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_move_tree_to_cpu_async", teardown)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        forward_batches_pipelined(
            _executor(produce),
            "req",
            ["c0", "c1"],
        )

    assert produced == ["c0", "c1"]
    assert torn_down == [("result", "c0")]
    assert len(events) == 2
    assert events[0].recorded_stream is None
    assert events[0].synchronize_calls == 0
    assert events[1].recorded_stream is copy_stream
    assert events[1].synchronize_calls == 1
    assert copy_stream.synchronize_calls == 1


def test_async_tree_move_preserves_slots_dataclass_and_records_source_stream(
    monkeypatch,
) -> None:
    class _CudaTensor:
        is_cuda = True
        shape = (2,)
        dtype = "float32"

        def __init__(self, name: str) -> None:
            self.name = name
            self.recorded_streams: list[object] = []

        def detach(self):
            return self

        def record_stream(self, stream) -> None:
            self.recorded_streams.append(stream)

    class _HostTensor:
        def __init__(self) -> None:
            self.source = None
            self.non_blocking = None

        def copy_(self, source, *, non_blocking: bool) -> None:
            self.source = source
            self.non_blocking = non_blocking

    hosts: list[_HostTensor] = []

    def _empty(*_args, **_kwargs) -> _HostTensor:
        host = _HostTensor()
        hosts.append(host)
        return host

    copy_stream = object()
    fake_torch = SimpleNamespace(
        Tensor=_CudaTensor,
        empty=_empty,
        cuda=SimpleNamespace(stream=lambda _stream: nullcontext()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sources = [_CudaTensor(name) for name in ("obs", "actions", "steps", "video", "replay")]
    batch = DiffusionBatchResult(
        batch=GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1),
        observations=sources[0],
        actions=sources[1],
        log_probs=None,
        timesteps=sources[2],
        kl=None,
        video=sources[3],
        replay_tensors={"latents": sources[4]},
        context={"prompt": "p"},
    )

    moved = _move_tree_to_cpu_async(batch, copy_stream)

    assert not hasattr(batch, "__dict__")
    assert isinstance(moved, DiffusionBatchResult)
    assert moved is not batch
    assert moved.observations.source is sources[0]
    assert moved.actions.source is sources[1]
    assert moved.timesteps.source is sources[2]
    assert moved.video.source is sources[3]
    assert moved.replay_tensors["latents"].source is sources[4]
    assert moved.context == {"prompt": "p"}
    assert len(hosts) == len(sources)
    assert all(host.non_blocking is True for host in hosts)
    assert all(source.recorded_streams == [copy_stream] for source in sources)
