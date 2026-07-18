"""REAL-CUDA bit-exactness of forward_chunks_pipelined — the 1-GPU integration
check the unit tests defer. produce() does an actual GPU matmul (so there is an
in-flight kernel the side-stream copy must wait on via the Event), the default
teardown is the real _move_tree_to_cpu_async D2H on the copy stream, and we assert
torch.equal vs serial produce+copy. A missing/incorrect Event (torn read) or a
value-mutating copy would diverge here."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vrl.generation.bindings.joint_denoise.executor import DiffusionChunkResult  # noqa: E402
from vrl.generation.execution.pipeline import (  # noqa: E402
    _move_tree_to_cpu_async,
    forward_chunks_pipelined,
)

pytestmark = pytest.mark.gpu


def _produce(chunk: int):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(chunk)
    a = torch.randn(512, 512, generator=g, device=dev, dtype=torch.float32)
    # Real in-flight GPU work: the chunk result depends on a kernel that must
    # FINISH before the side-stream D2H reads it (the Event guards exactly this).
    out = a @ a
    return {"obs": out, "scalar": out.sum().reshape(1)}


def _serial(chunks):
    results = []
    for c in chunks:
        r = _produce(c)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        cpu = _move_tree_to_cpu_async(r, s)
        s.synchronize()
        results.append(cpu)
    return results


def _executor(produce):
    return SimpleNamespace(
        forward_chunk_plan=lambda _request, chunk: produce(chunk),
    )


def test_pipelined_bit_exact_vs_serial_real_cuda() -> None:
    chunks = [0, 1, 2, 3, 4, 5]

    serial = _serial(chunks)
    pipelined = forward_chunks_pipelined(_executor(_produce), "req", chunks)

    assert len(pipelined) == len(serial)
    for idx, (sp, pp) in enumerate(zip(serial, pipelined, strict=True)):
        assert pp["obs"].device.type == "cpu"  # teardown moved it off GPU
        assert torch.equal(sp["obs"], pp["obs"]), f"chunk {idx} obs diverged"
        assert torch.equal(sp["scalar"], pp["scalar"]), f"chunk {idx} scalar diverged"


def test_pipelined_preserves_chunk_order_real_cuda() -> None:
    # Distinct seeds => distinct values; the result list must stay in chunk order
    # regardless of copy-completion order.
    chunks = [10, 20, 30, 40]
    pipelined = forward_chunks_pipelined(_executor(_produce), "req", chunks)
    expected = [_produce(c)["scalar"].cpu() for c in chunks]
    for got, exp in zip(pipelined, expected, strict=True):
        assert torch.equal(got["scalar"], exp)


def test_pipelined_moves_real_slots_chunk_result_to_cpu() -> None:
    def _produce_chunk(chunk: int) -> DiffusionChunkResult:
        tensors = _produce(chunk)
        return DiffusionChunkResult(
            prompt_index=0,
            sample_start=chunk,
            sample_count=1,
            observations=tensors["obs"],
            actions=None,
            log_probs=None,
            timesteps=None,
            kl=None,
            video=tensors["scalar"],
            replay_tensors={},
            context={"chunk": chunk},
        )

    results = forward_chunks_pipelined(
        _executor(_produce_chunk),
        "req",
        [0, 1],
    )

    assert all(isinstance(result, DiffusionChunkResult) for result in results)
    assert all(result.observations.device.type == "cpu" for result in results)
    assert all(result.video.device.type == "cpu" for result in results)
    assert [result.context["chunk"] for result in results] == [0, 1]
