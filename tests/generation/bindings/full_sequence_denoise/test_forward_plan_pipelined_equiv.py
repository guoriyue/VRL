"""End-to-end equivalence: DiffusionBatchExecutorBase.forward_plan_pipelined produces the
SAME gathered output as the serial forward_plan, on a real EnginePlan with real
tensors through the real stage chain + gather. This is the executor-level
correctness run; the ``real_cover`` label names the gpu-lane twin that owns the
mechanism's bit-exactness."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vrl.generation.bindings.full_sequence_denoise.executor import (  # noqa: E402
    DiffusionBatchExecutorBase,
)
from vrl.generation.execution.planner import EnginePlan  # noqa: E402
from vrl.generation.types import GenerationRequest  # noqa: E402


class _RealChunkExecutor:
    """Produces deterministic real tensors through one canonical batch method."""

    model = None  # no versioned slots

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def forward_batch(self, request, batch):
        del request
        g = torch.Generator(device=self.device).manual_seed(int(batch.sample_start) + 1)
        x = torch.randn(batch.sample_count, 8, generator=g, device=self.device) + 1.0
        return x @ torch.ones(8, 8, device=self.device)

    def gather_batches(self, request, sample_rows, batches):
        del request, sample_rows
        return SimpleNamespace(output=list(batches))


def _request(num_samples: int) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-equiv",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=num_samples,
        sampling={"samples_per_generation_batch": 2},
    )


def _plan(request, sample_rows):
    del sample_rows
    return EnginePlan.from_request(request)


@pytest.mark.real_cover(
    "tests/generation/execution/test_sample_batches_pipelined_cuda.py",
    why=(
        "the device falls back to CPU when CUDA is absent, and on CPU the pipelining "
        "mechanism this test wraps — the cross-stream Event and the async D2H copy on the "
        "side stream — does not execute at all; the gpu-lane file drives both for real"
    ),
)
def test_forward_plan_pipelined_matches_serial_forward_plan() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = _RealChunkExecutor(device)
    request = _request(6)
    sample_rows = request.sample_rows()
    plan = _plan(request, sample_rows)
    assert len(plan.sample_batches) >= 2

    serial = DiffusionBatchExecutorBase.forward_plan(ex, request, sample_rows, plan)
    pipelined = DiffusionBatchExecutorBase.forward_plan_pipelined(ex, request, sample_rows, plan)

    assert len(pipelined.output) == len(serial.output)
    for idx, (s, p) in enumerate(zip(serial.output, pipelined.output, strict=True)):
        # serial keeps GPU tensors; pipelined's teardown moved them to CPU — same
        # VALUES, compared on CPU.
        assert torch.equal(s.detach().cpu(), p.detach().cpu()), f"batch {idx} diverged"
