"""End-to-end equivalence: DiffusionChunkExecutorBase.forward_plan_pipelined produces the
SAME gathered output as the serial forward_plan, on a real EnginePlan with real
tensors through the real stage chain + gather. This is the executor-level
correctness run; the ``real_cover`` label names the gpu-lane twin that owns the
mechanism's bit-exactness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vrl.generation.bindings.full_sequence_denoise.executor import (  # noqa: E402
    DiffusionChunkExecutorBase,
)
from vrl.generation.execution.ids import build_sample_rows  # noqa: E402
from vrl.generation.execution.planner import build_engine_plan  # noqa: E402
from vrl.generation.types import GenerationOutput, GenerationRequest  # noqa: E402


class _RealStageExecutor:
    """Exposes exactly the stage methods + forward_chunk_plan + gather that
    forward_plan / forward_plan_pipelined call, producing REAL tensors per chunk
    (deterministic by sample_start) through a real matmul denoise."""

    model = None  # no versioned slots

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def build_prompt_stage_input(self, request, chunk):
        return chunk

    def run_prompt_encode_stage(self, chunk, *, stage_durations, record_function):
        g = torch.Generator(device=self.device).manual_seed(int(chunk.sample_start) + 1)
        return torch.randn(chunk.sample_count, 8, generator=g, device=self.device)

    def run_prepare_stage(self, x, *, stage_durations):
        return x + 1.0

    def run_denoise_stage(self, x, *, stage_durations):
        return x @ torch.ones(8, 8, device=self.device)

    def run_decode_stage(self, x):
        return {"samples": x}

    def apply_wire_storage_policy(self, request, result):
        return result

    def forward_chunk_plan(self, request, chunk):
        x = self.run_prompt_encode_stage(
            self.build_prompt_stage_input(request, chunk),
            stage_durations={},
            record_function=lambda *_a, **_k: _Null(),
        )
        x = self.run_prepare_stage(x, stage_durations={})
        x = self.run_denoise_stage(x, stage_durations={})
        return self.apply_wire_storage_policy(request, self.run_decode_stage(x))

    def gather_chunks(self, request, sample_rows, chunks):
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            sample_rows=list(sample_rows),
            output=[c["samples"] for c in chunks],
        )


class _Null:
    """No-op record_function context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _request(num_samples: int) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-equiv",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=num_samples,
        sampling={"samples_per_chunk": 2},
    )


def _plan(request, sample_rows):
    del sample_rows
    return build_engine_plan(request)


@pytest.mark.real_cover(
    "tests/generation/execution/test_chunks_pipelined_cuda.py",
    why=(
        "the device falls back to CPU when CUDA is absent, and on CPU the pipelining "
        "mechanism this test wraps — the cross-stream Event and the async D2H copy on the "
        "side stream — does not execute at all; the gpu-lane file drives both for real"
    ),
)
def test_forward_plan_pipelined_matches_serial_forward_plan() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = _RealStageExecutor(device)
    request = _request(6)
    sample_rows = build_sample_rows(request)
    plan = _plan(request, sample_rows)
    assert len(plan.chunks) >= 2

    serial = DiffusionChunkExecutorBase.forward_plan(ex, request, sample_rows, plan)
    pipelined = DiffusionChunkExecutorBase.forward_plan_pipelined(ex, request, sample_rows, plan)

    assert len(pipelined.output) == len(serial.output)
    for idx, (s, p) in enumerate(zip(serial.output, pipelined.output, strict=True)):
        # serial keeps GPU tensors; pipelined's teardown moved them to CPU — same
        # VALUES, compared on CPU.
        assert torch.equal(s.detach().cpu(), p.detach().cpu()), f"chunk {idx} diverged"
