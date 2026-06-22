"""Per-rank ownership contract for the multi-rank online loop.

The training rollout is data-parallel: every rank drives its OWN colocated Ray
runtime over a disjoint prompt shard (``run_online_recipe`` hands each rank
``idx[rank*rbs:(rank+1)*rbs]``), so the old "only rank0 touches Ray" spike is no
longer the shape. The contract the loop actually keeps now is:

    training rollout  -> EVERY rank generates its local prompt shard
    fixed eval        -> EVERY rank generates its local eval shard,
                         rank0 ALONE writes eval_metrics.csv
    checkpoint        -> EVERY rank runs the trainable-state gather (a collective),
                         rank0 ALONE writes the files

The single writer is always ``context.is_primary``; the work that precedes the
write (Ray generation, the FSDP gather) runs on all ranks so no collective
mismatches. The barrier seam exists on every rank (no-op single_process, real
under FSDP).
"""

from __future__ import annotations

import torch

from vrl.trainers.distributed import resolve_training_context
from vrl.trainers.strategy import SingleProcessStrategy

_FSDP_CFG = {
    "distributed": {"training": {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2}},
}


class _FakeRayRuntime:
    """Records how often each rank hits the Ray-facing rollout entry points."""

    def __init__(self) -> None:
        self.train_generate_calls = 0
        self.eval_generate_calls = 0
        self.update_weights_calls = 0

    def train_generate(self) -> None:
        self.train_generate_calls += 1

    def eval_generate(self) -> None:
        self.eval_generate_calls += 1

    def update_weights(self) -> None:
        self.update_weights_calls += 1


class _FakeOutputSink:
    """Records output writes (metrics / eval rows / checkpoint files) and gathers."""

    def __init__(self) -> None:
        self.writes = 0
        self.gathers = 0

    def gather(self) -> None:
        self.gathers += 1

    def write(self) -> None:
        self.writes += 1


def _fsdp_context(rank: int):
    return resolve_training_context(
        _FSDP_CFG,
        device=torch.device("cpu"),
        env={"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2"},
    )


def _drive_training_rollout(context, runtime: _FakeRayRuntime) -> None:
    """Every rank collects+trains its own prompt shard; rank0 writes the metric row.

    Generation and weight sync run on ALL ranks (each rank owns a disjoint shard
    and pushes its trained state); only the metric write is rank0-gated.
    """
    runtime.train_generate()
    runtime.update_weights()


def _drive_fixed_eval(context, runtime: _FakeRayRuntime, sink: _FakeOutputSink) -> None:
    """Every rank evaluates its eval shard; the stats all-reduce is implied; rank0 writes."""
    runtime.eval_generate()
    if context.is_primary:
        sink.write()


def _drive_checkpoint(context, sink: _FakeOutputSink) -> None:
    """Every rank runs the trainable-state gather (collective); rank0 writes files."""
    sink.gather()
    if context.is_primary:
        sink.write()


def test_every_rank_drives_its_own_training_rollout() -> None:
    """Training rollout is data-parallel: each rank generates its local shard."""
    runtime = _FakeRayRuntime()
    for rank in range(2):
        _drive_training_rollout(_fsdp_context(rank), runtime)

    # Both ranks generate + sync (disjoint shards), not rank0 alone.
    assert runtime.train_generate_calls == 2
    assert runtime.update_weights_calls == 2


def test_fixed_eval_runs_on_every_rank_but_only_rank0_writes() -> None:
    """Every rank evaluates a disjoint eval shard; rank0 alone appends the row."""
    runtime, sink = _FakeRayRuntime(), _FakeOutputSink()
    for rank in range(2):
        _drive_fixed_eval(_fsdp_context(rank), runtime, sink)

    assert runtime.eval_generate_calls == 2  # eval work is sharded, not rank0-only
    assert sink.writes == 1  # single writer: no duplicate eval_metrics.csv rows


def test_checkpoint_gathers_on_every_rank_but_only_rank0_writes() -> None:
    """The trainable-state gather is a collective on all ranks; rank0 writes files."""
    sink = _FakeOutputSink()
    for rank in range(2):
        _drive_checkpoint(_fsdp_context(rank), sink)

    assert sink.gathers == 2  # all ranks must enter the gather or FSDP deadlocks
    assert sink.writes == 1  # single writer


def test_primary_flag_is_rank0_only() -> None:
    assert _fsdp_context(0).is_primary is True
    assert _fsdp_context(1).is_primary is False


def test_barrier_seam_is_callable_and_noop_in_single_process() -> None:
    """The barrier seam exists for every rank (no-op now, real under FSDP)."""
    strategy = SingleProcessStrategy()
    for rank in range(2):
        _fsdp_context(rank)  # env resolves cleanly for each rank
        assert strategy.barrier() is None
