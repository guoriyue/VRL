"""Actual strict schedule/coordinator with only CP leaders owning collectors.

Every rank runs the real coordinator stack: the CP leaders own a real
collector over the tiny SANA in-process runtime and a real trainer side; the
other ranks own nothing and must observe no transition.
"""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.rollouts.collector._helpers import Trace, real_collector, trainer_side
from vrl.ray.resources import RayLifecyclePlan
from vrl.rollouts.orchestration.strict_on_policy import (
    ContextParallelStrictRolloutSchedule,
    StrictOnPolicyRolloutSchedule,
)
from vrl.trainers.distributed import create_context_parallel_groups


def _worker(rank, rendezvous, spool):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90)
    )
    monkeypatch = pytest.MonkeyPatch()
    try:
        groups = create_context_parallel_groups(2)
        phase = Trace(monkeypatch)
        bench = None
        owner = None
        if groups.cp_rank == 0:
            bench = real_collector(monkeypatch, Path(spool) / f"rank{rank}")
            trainer = trainer_side(bench, initialized=True)
            phase.watch(bench.collector, "activate_generation_runtime", "activate_rollout")
            phase.watch(bench.collector, "offload_generation_runtime_memory", "offload_rollout")
            phase.watch(bench.collector, "shutdown", "collector_shutdown")
            phase.watch(bench.runtime, "update_weights", "sync_weights_after_train")
            owner = StrictOnPolicyRolloutSchedule(lifecycle=trainer.coordinator(bench))
        schedule = ContextParallelStrictRolloutSchedule(
            owner=owner, groups=groups, spool_dir=spool
        )
        result = asyncio.run(schedule.next_iteration([], group_size=2))
        assert result.batches == []
        asyncio.run(schedule.after_train_step())
        schedule.reset()
        if owner is not None:
            assert phase.events == [
                "activate_rollout",
                "offload_rollout",
                "sync_weights_after_train",
            ]
        else:
            assert phase.events == []
        if bench is not None and groups.dp_rank == 0:
            bench.collector.lifecycle = RayLifecyclePlan(trainer=(0,), rollout=(0,), reward=())
        with pytest.raises(RuntimeError, match="disjoint"):
            asyncio.run(schedule.next_iteration([], group_size=2))

        if bench is not None and groups.dp_rank == 0:
            phase.fail("sync_weights_after_train", "owner weight publication failed")
        with pytest.raises(RuntimeError, match="owner weight publication failed"):
            asyncio.run(schedule.after_train_step())
        asyncio.run(schedule.shutdown())
        assert phase.events.count("collector_shutdown") == (1 if owner is not None else 0)
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


def test_cp_strict_owner_lifecycle(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), str(tmp_path)), nprocs=4, join=True)
