"""execute_request_batches runs a request's batches in order on one worker and
returns the same per-batch results as running each batch serially. Each result
is copied to pinned CPU memory before the next batch is produced, so at most
one batch's payload is on the GPU at a time; batches are independent."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from vrl.generation.bindings.full_sequence_denoise.executor import DenoiseBatchResult
from vrl.generation.execution.executor_base import BatchExecutorBase
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import BatchCompletion
from vrl.trajectory import device as device_module


def _executor(produce):
    class _Executor(BatchExecutorBase):
        family = "test"

        def forward_batch(self, request, batch):
            del request
            return produce(batch)

    return _Executor()


def test_pipelined_results_equal_serial_in_batch_order() -> None:
    def produce(batch):
        return ("denoised", batch)

    pipelined = _executor(produce).execute_request_batches(
        "req",
        ["c0", "c1", "c2", "c3"],
    )
    serial = [produce(batch) for batch in ["c0", "c1", "c2", "c3"]]

    assert pipelined == serial
    assert pipelined == [
        ("denoised", "c0"),
        ("denoised", "c1"),
        ("denoised", "c2"),
        ("denoised", "c3"),
    ]


def test_every_batch_produced_exactly_once() -> None:
    produced: list = []

    def produce(batch):
        produced.append(batch)
        return ("r", batch)

    batches = [f"c{i}" for i in range(5)]
    out = _executor(produce).execute_request_batches("req", batches)

    assert produced == batches  # produced in order, once each
    assert len(out) == len(batches)


def test_each_batch_is_copied_to_cpu_before_the_next_is_produced(monkeypatch) -> None:
    operations: list[tuple[str, str]] = []

    def produce(batch):
        operations.append(("produce", batch))
        return ("result", batch)

    def copy(result):
        operations.append(("copy", result[1]))
        return ("host", result[1])

    monkeypatch.setattr(device_module, "copy_tensor_tree_to_pinned_cpu", copy)

    out = _executor(produce).execute_request_batches("req", ["c0", "c1", "c2"])

    assert out == [("host", "c0"), ("host", "c1"), ("host", "c2")]
    assert operations == [
        ("produce", "c0"),
        ("copy", "c0"),
        ("produce", "c1"),
        ("copy", "c1"),
        ("produce", "c2"),
        ("copy", "c2"),
    ]


def test_completion_fence_follows_each_copied_batch(monkeypatch) -> None:
    order: list[str] = []

    def copy(result):
        order.append(f"copy:{result[1]}")
        return result

    monkeypatch.setattr(device_module, "copy_tensor_tree_to_pinned_cpu", copy)
    completions: list[BatchCompletion] = []

    def publish(completion: BatchCompletion) -> None:
        order.append(f"completion:{completion.completed_batches}")
        completions.append(completion)

    output = _executor(lambda batch: ("result", batch)).execute_request_batches(
        "req",
        ["c0", "c1", "c2"],
        completion_callback=publish,
    )

    assert output == [("result", "c0"), ("result", "c1"), ("result", "c2")]
    assert [completion.completed_batches for completion in completions] == [1, 2, 3]
    # A completion is published only after its batch's result is on the CPU.
    assert order == [
        "copy:c0",
        "completion:1",
        "copy:c1",
        "completion:2",
        "copy:c2",
        "completion:3",
    ]


def test_single_batch_still_produces_and_copies() -> None:
    out = _executor(lambda batch: ("p", batch)).execute_request_batches(
        "req",
        ["only"],
    )
    assert out == [("p", "only")]


def test_empty_batches_returns_empty() -> None:
    out = _executor(lambda batch: batch).execute_request_batches(
        "req",
        [],
    )
    assert out == []


def test_produce_error_propagates_after_earlier_batches_were_copied(monkeypatch) -> None:
    produced: list[str] = []
    copied: list[tuple[str, str]] = []

    def produce(batch: str) -> tuple[str, str]:
        produced.append(batch)
        if batch == "c1":
            raise RuntimeError("CUDA out of memory in second batch")
        return ("result", batch)

    def copy(result):
        copied.append(result)
        return result

    monkeypatch.setattr(device_module, "copy_tensor_tree_to_pinned_cpu", copy)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _executor(produce).execute_request_batches("req", ["c0", "c1"])

    assert produced == ["c0", "c1"]
    assert copied == [("result", "c0")]


def test_pinned_copy_preserves_slots_dataclass_and_waits_once(monkeypatch) -> None:
    synchronize_calls: list[None] = []

    class _CudaTensor:
        is_cuda = True
        shape = (2,)
        dtype = "float32"

        def __init__(self, name: str) -> None:
            self.name = name

        def detach(self):
            return self

    class _HostTensor:
        def __init__(self) -> None:
            self.source = None
            self.non_blocking = None

        def copy_(self, source, *, non_blocking: bool) -> None:
            self.source = source
            self.non_blocking = non_blocking

    hosts: list[_HostTensor] = []

    def _empty(*_args, **kwargs) -> _HostTensor:
        assert kwargs["pin_memory"] is True
        host = _HostTensor()
        hosts.append(host)
        return host

    fake_torch = SimpleNamespace(
        Tensor=_CudaTensor,
        empty=_empty,
        cuda=SimpleNamespace(synchronize=lambda: synchronize_calls.append(None)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sources = [_CudaTensor(name) for name in ("latents", "steps", "video", "replay")]
    batch = DenoiseBatchResult(
        batch=GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1),
        latents=sources[0],
        log_probs=None,
        timesteps=sources[1],
        kl=None,
        video=sources[2],
        replay_tensors={"latents": sources[3]},
        context={"prompt": "p"},
    )

    moved = device_module.copy_tensor_tree_to_pinned_cpu(batch)

    assert not hasattr(batch, "__dict__")
    assert isinstance(moved, DenoiseBatchResult)
    assert moved is not batch
    assert moved.latents.source is sources[0]
    assert moved.timesteps.source is sources[1]
    assert moved.video.source is sources[2]
    assert moved.replay_tensors["latents"].source is sources[3]
    assert moved.context == {"prompt": "p"}
    assert len(hosts) == len(sources)
    assert all(host.non_blocking is True for host in hosts)
    assert len(synchronize_calls) == 1


def test_pinned_copy_without_cuda_tensors_does_not_synchronize(monkeypatch) -> None:
    class _CpuTensor:
        is_cuda = False

        def detach(self):
            return self

        def cpu(self):
            return ("cpu", self)

    def _fail_synchronize() -> None:
        raise AssertionError("no CUDA copy was queued")

    fake_torch = SimpleNamespace(
        Tensor=_CpuTensor,
        cuda=SimpleNamespace(synchronize=_fail_synchronize),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    source = _CpuTensor()

    assert device_module.copy_tensor_tree_to_pinned_cpu({"x": source}) == {"x": ("cpu", source)}
