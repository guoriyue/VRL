"""Actual strict schedule/coordinator with only CP leaders owning collectors."""

import asyncio
from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.rollouts.orchestration.test_strict_failure_path import _Collector, _schedule, _Strategy
from vrl.rollouts.orchestration.strict_on_policy import ContextParallelStrictRolloutSchedule
from vrl.trainers.distributed import create_context_parallel_groups


def _worker(rank, rendezvous, spool):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90)
    )
    try:
        groups = create_context_parallel_groups(2)
        calls = []
        collector = _Collector(calls, collect_raises=False, trainer_shares_gpu=False)
        owner = (
            _schedule(collector, _Strategy(calls), calls=calls, with_syncer=True)
            if groups.cp_rank == 0
            else None
        )
        schedule = ContextParallelStrictRolloutSchedule(
            owner=owner, groups=groups, spool_dir=spool
        )
        result = asyncio.run(schedule.next_iteration([], group_size=2))
        assert result.batches == []
        asyncio.run(schedule.after_train_step())
        schedule.reset()
        if owner is not None:
            assert calls == ["activate_rollout", "offload_rollout", "sync_weights_after_train"]
        else:
            assert calls == []
        if owner is not None and groups.dp_rank == 0:
            collector.generation_runtime.requires_driver_model_offload = True
        with pytest.raises(RuntimeError, match="disjoint"):
            asyncio.run(schedule.next_iteration([], group_size=2))

        async def fail_push(_state):
            raise RuntimeError("owner weight publication failed")

        if owner is not None and groups.dp_rank == 0:
            owner.lifecycle.weight_syncer.push = fail_push
        with pytest.raises(RuntimeError, match="owner weight publication failed"):
            asyncio.run(schedule.after_train_step())
        asyncio.run(schedule.shutdown())
        assert calls.count("collector_shutdown") == (1 if owner is not None else 0)
    finally:
        dist.destroy_process_group()


def test_cp_strict_owner_lifecycle(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), str(tmp_path)), nprocs=4, join=True)
