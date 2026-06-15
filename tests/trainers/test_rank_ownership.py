"""Rank0 ↔ Ray ownership spike (readiness P6).

Not a real distributed run: monkeypatched torchrun env + a fake Ray runtime that
counts calls. This locks the ownership contract a future FSDP driver must follow
so N ranks never all open a Ray client, double-sample, or write duplicate
outputs:

    primary rank      -> drives the Ray rollout runtime (generate / update_weights)
                         and writes metrics / checkpoints / eval
    non-primary rank  -> touches neither; waits at the strategy barrier

The gate is ``context.is_primary``, resolved from env by
``resolve_training_context``. These tests prove that flag is right per rank under
the exact env a torchrun launch sets, and that gating on it yields the intended
call pattern. The barrier seam already exists (no-op in single_process, real in
the future FSDP strategy).
"""

from __future__ import annotations

import torch

from vrl.trainers.distributed import resolve_training_context
from vrl.trainers.strategy import SingleProcessStrategy

_FSDP_CFG = {
    "distributed": {"training": {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2}},
}


class _FakeRayRuntime:
    """Records how often each Ray-facing rollout entry point is hit."""

    def __init__(self) -> None:
        self.generate_calls = 0
        self.update_weights_calls = 0

    def generate(self) -> None:
        self.generate_calls += 1

    def update_weights(self) -> None:
        self.update_weights_calls += 1


class _FakeOutputSink:
    """Records metric/checkpoint/eval writes."""

    def __init__(self) -> None:
        self.writes = 0

    def write(self) -> None:
        self.writes += 1


def _fsdp_context(rank: int):
    return resolve_training_context(
        _FSDP_CFG,
        device=torch.device("cpu"),
        env={"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2"},
    )


def _drive_one_step(context, runtime: _FakeRayRuntime, sink: _FakeOutputSink) -> None:
    """The ownership rule a multi-rank driver must follow.

    Only the primary rank owns Ray I/O and output writes; this is the shape the
    FSDP loop has to keep so non-primary ranks never re-drive Ray or double-write.
    """
    if context.is_primary:
        runtime.generate()
        runtime.update_weights()
        sink.write()


def test_primary_rank_drives_ray_and_writes() -> None:
    ctx = _fsdp_context(0)
    assert ctx.is_primary is True

    runtime, sink = _FakeRayRuntime(), _FakeOutputSink()
    _drive_one_step(ctx, runtime, sink)

    assert (runtime.generate_calls, runtime.update_weights_calls, sink.writes) == (1, 1, 1)


def test_non_primary_rank_never_touches_ray_or_writes() -> None:
    ctx = _fsdp_context(1)
    assert ctx.is_primary is False

    runtime, sink = _FakeRayRuntime(), _FakeOutputSink()
    _drive_one_step(ctx, runtime, sink)

    assert (runtime.generate_calls, runtime.update_weights_calls, sink.writes) == (0, 0, 0)


def test_only_one_rank_generates_across_the_world() -> None:
    """Summed across every rank of the world, Ray generate fires exactly once."""
    runtime, sink = _FakeRayRuntime(), _FakeOutputSink()
    for rank in range(2):
        _drive_one_step(_fsdp_context(rank), runtime, sink)

    assert runtime.generate_calls == 1  # no double-sampling
    assert runtime.update_weights_calls == 1  # no duplicate weight push
    assert sink.writes == 1  # no duplicate metrics/checkpoint/eval


def test_barrier_seam_is_callable_and_noop_in_single_process() -> None:
    """The barrier seam exists for every rank (no-op now, real under FSDP)."""
    strategy = SingleProcessStrategy()
    for rank in range(2):
        _fsdp_context(rank)  # env resolves cleanly for each rank
        assert strategy.barrier() is None
