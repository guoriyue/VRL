"""REAL-CUDA equality of execute_request_batches against serial produce + copy:
produce() does an actual GPU matmul so there is in-flight work the synchronous
pinned copy must wait on, and we assert torch.equal versus serial produce and
``.cpu()``. A copy that returned before the kernel finished, or one that
mutated values, would diverge here."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vrl.generation.bindings.full_sequence_denoise.executor import (  # noqa: E402
    DenoiseBatchResult,
)
from vrl.generation.execution.executor_base import BatchExecutorBase  # noqa: E402
from vrl.generation.execution.sample_batches import GenerationSampleBatch  # noqa: E402

pytestmark = pytest.mark.gpu


def _produce(batch: int):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(batch)
    a = torch.randn(512, 512, generator=g, device=dev, dtype=torch.float32)
    # Real in-flight GPU work: the batch result depends on a kernel that must
    # FINISH before the host copy reads it.
    out = a @ a
    return {"obs": out, "scalar": out.sum().reshape(1)}


def _serial(batches):
    results = []
    for c in batches:
        r = _produce(c)
        results.append({key: value.cpu() for key, value in r.items()})
    return results


def _executor(produce):
    class _Executor(BatchExecutorBase):
        family = "test"

        def forward_batch(self, request, batch):
            del request
            return produce(batch)

    return _Executor()


def test_pipelined_equals_serial_real_cuda() -> None:
    batches = [0, 1, 2, 3, 4, 5]

    serial = _serial(batches)
    pipelined = _executor(_produce).execute_request_batches("req", batches)

    assert len(pipelined) == len(serial)
    for idx, (sp, pp) in enumerate(zip(serial, pipelined, strict=True)):
        assert pp["obs"].device.type == "cpu"
        assert pp["obs"].is_pinned()
        assert torch.equal(sp["obs"], pp["obs"]), f"batch {idx} obs diverged"
        assert torch.equal(sp["scalar"], pp["scalar"]), f"batch {idx} scalar diverged"


def test_pipelined_preserves_batch_order_real_cuda() -> None:
    # Distinct seeds => distinct values; the result list must stay in batch order.
    batches = [10, 20, 30, 40]
    pipelined = _executor(_produce).execute_request_batches("req", batches)
    expected = [_produce(c)["scalar"].cpu() for c in batches]
    for got, exp in zip(pipelined, expected, strict=True):
        assert torch.equal(got["scalar"], exp)


def test_pipelined_moves_real_slots_batch_result_to_cpu() -> None:
    def _produce_batch(batch: int) -> DenoiseBatchResult:
        tensors = _produce(batch)
        return DenoiseBatchResult(
            batch=GenerationSampleBatch(prompt_index=0, sample_start=batch, sample_count=1),
            latents=tensors["obs"],
            log_probs=None,
            timesteps=None,
            video=tensors["scalar"],
            replay_tensors={},
            context={"batch": batch},
        )

    results = _executor(_produce_batch).execute_request_batches("req", [0, 1])

    assert all(isinstance(result, DenoiseBatchResult) for result in results)
    assert all(result.latents.device.type == "cpu" for result in results)
    assert all(result.video.device.type == "cpu" for result in results)
    assert [result.context["batch"] for result in results] == [0, 1]


def test_real_cuda_completion_counts_follow_copied_batches() -> None:
    completions: list[int] = []

    _executor(_produce).execute_request_batches(
        "req", [0, 1], completion_callback=completions.append
    )

    assert completions == [1, 2]
