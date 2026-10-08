"""The skip-backward decision must be unanimous across training ranks.

A backward pass fires cross-rank collectives — FSDP2 per-layer all-gather +
reduce-scatter, or DDP's gradient all-reduce. If one rank skips an all-filtered
(zero-advantage) microbatch while another rank runs it, those collectives
mismatch and the job DEADLOCKS (an unrecoverable NCCL hang). ``all_ranks_true``
all-reduces the local ``has_work`` flag with MIN so every rank takes the SAME
branch: the microbatch runs only when ALL ranks have work. This spawns a real
gloo 2-rank group and asserts the agreed result is the logical AND of the ranks'
local flags.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.rollouts.collector._helpers import Trace, real_collector
from tests.trainers._strategy_policies import free_port
from tests.trainers.online._helpers import _diffusion_rollout_batch
from vrl.algorithms.logprob_mismatch import LogprobMismatchStats
from vrl.algorithms.types import InitialReplayStats, PolicyUpdateStats, TrainStepMetrics
from vrl.rewards import RewardOutput, RewardSample
from vrl.rewards.base import RewardFunction
from vrl.rollouts.batch import RolloutBatch
from vrl.trainers.distributed import DistributedTrainingContext, TrainingCollectives
from vrl.trainers.online.trainer import (
    OnlineTrainer,
    _distributed_initial_replay_stats,
    _ReplayMetrics,
    _TrainingMicrobatch,
)
from vrl.trainers.strategy import DDPStrategy, SingleProcessStrategy

# (rank0_has_work, rank1_has_work) -> the agreed result both ranks must return.
_CASES = {
    "both_have_work": ([True, True], True),
    "one_rank_empty": ([True, False], False),
    "both_empty": ([False, False], False),
}


def _rank_strategy():
    if not dist.is_initialized():
        return SingleProcessStrategy()
    return DDPStrategy(
        DistributedTrainingContext(
            strategy="ddp",
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            device=torch.device("cpu"),
        ),
        find_unused_parameters=False,
    )


def _rollout_batch(sample_count: int) -> RolloutBatch:
    return _diffusion_rollout_batch(
        rewards=torch.arange(sample_count, dtype=torch.float32),
        group_ids=torch.zeros(sample_count, dtype=torch.long),
        num_steps=1,
    )


def _run_rank(rank: int, world_size: int, port: int, local_flags: list[bool], q: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        collectives = _rank_strategy().collectives
        agreed = collectives.all_true(local_flags[rank])
        assert collectives.succeeded(local_flags[rank]) is agreed
        # A local strategy must not join this already-live distributed group.
        assert SingleProcessStrategy().collectives.all_true(local_flags[rank]) is local_flags[rank]
        q.put((rank, agreed))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("local_flags", "expected"),
    list(_CASES.values()),
    ids=list(_CASES),
)
def test_skip_backward_decision_is_unanimous(local_flags: list[bool], expected: bool) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_run_rank, args=(r, 2, port, local_flags, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, agreed = q.get(timeout=50)
        results[rank] = agreed
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0
    # Both ranks must agree, and on the AND of the local flags.
    assert results[0] is expected
    assert results[1] is expected


def test_falls_back_to_local_without_process_group() -> None:
    assert _rank_strategy().collectives.all_true(True) is True
    assert _rank_strategy().collectives.all_true(False) is False


def test_zero_weight_initial_replay_is_fully_neutral() -> None:
    resolved, has_measurements = _distributed_initial_replay_stats(
        InitialReplayStats(
            clip_fraction=float("nan"),
            active_clip_fraction=float("inf"),
            logprob_abs_diff_max=float("inf"),
            finite=False,
        ),
        local_weight=0.0,
        strategy=_rank_strategy(),
    )

    assert has_measurements is False
    assert resolved == InitialReplayStats()


def _run_parity_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        initial_replay, initial_has_measurements = _distributed_initial_replay_stats(
            InitialReplayStats(
                clip_fraction=(0.2, 0.8)[rank],
                active_clip_fraction=(0.1, 0.4)[rank],
                logprob_abs_diff_max=(0.1, 0.9)[rank],
            ),
            local_weight=(1.0, 3.0)[rank],
            strategy=_rank_strategy(),
        )
        mixed_aggregate = _ReplayMetrics()
        if rank == 0:
            mixed_aggregate.add(
                TrainStepMetrics(
                    update=PolicyUpdateStats(
                        clip_fraction=0.2,
                        active_clip_fraction=0.1,
                    ),
                    logprob_mismatch=LogprobMismatchStats(
                        logprob_abs_diff_max=0.1,
                    ),
                ),
                weight=1.0,
                capture_initial_replay=True,
            )
        mixed_local, mixed_weight = mixed_aggregate.initial_replay_snapshot()
        mixed_rank_replay, mixed_has_measurements = _distributed_initial_replay_stats(
            mixed_local,
            local_weight=mixed_weight,
            strategy=_rank_strategy(),
        )

        empty_local, empty_weight = _ReplayMetrics().initial_replay_snapshot()
        empty_rank_replay, empty_has_measurements = _distributed_initial_replay_stats(
            empty_local,
            local_weight=empty_weight,
            strategy=_rank_strategy(),
        )
        q.put(
            (
                rank,
                initial_replay,
                initial_has_measurements,
                mixed_rank_replay,
                mixed_has_measurements,
                empty_rank_replay,
                empty_has_measurements,
            )
        )
    finally:
        dist.destroy_process_group()


def test_initial_replay_stats_are_rank_consistent() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_run_parity_rank, args=(r, 2, port, q)) for r in range(2)]
    for process in procs:
        process.start()
    results = [q.get(timeout=50) for _ in range(2)]
    for process in procs:
        process.join(timeout=10)
        assert process.exitcode == 0

    for (
        _rank,
        initial_replay,
        initial_has_measurements,
        mixed_rank_replay,
        mixed_has_measurements,
        empty_rank_replay,
        empty_has_measurements,
    ) in results:
        assert initial_replay.clip_fraction == pytest.approx(0.65)
        assert initial_replay.active_clip_fraction == pytest.approx(0.325)
        assert initial_replay.logprob_abs_diff_max == pytest.approx(0.9)
        assert initial_replay.finite is True
        assert initial_has_measurements is True
        assert mixed_has_measurements is True
        assert mixed_rank_replay.finite is True
        assert mixed_rank_replay.clip_fraction == pytest.approx(0.2)
        assert mixed_rank_replay.active_clip_fraction == pytest.approx(0.1)
        assert mixed_rank_replay.logprob_abs_diff_max == pytest.approx(0.1)
        assert empty_has_measurements is False
        assert empty_rank_replay == InitialReplayStats()


def _run_replay_planner_rank(
    rank: int,
    world_size: int,
    port: int,
    local_counts: list[int],
    q: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        sample_count = local_counts[rank]
        batches = _TrainingMicrobatch.plan_balanced(
            [_rollout_batch(sample_count)],
            [torch.ones(sample_count)],
            training_microbatch_size=1,
            strategy=_rank_strategy(),
        )
        dummies = [batch for batch in batches if batch.is_dummy]
        q.put(
            (
                rank,
                len(batches),
                len(dummies),
                sum(batch.loss_weight for batch in batches),
                # Padding slots carry no weight and no advantage: they exist only
                # so every rank issues the same number of collectives.
                all(batch.loss_weight == 0.0 for batch in dummies),
                all(torch.count_nonzero(batch.advantages) == 0 for batch in dummies),
                all(not batch.is_dummy for batch in batches[:sample_count]),
            )
        )
    finally:
        dist.destroy_process_group()


def test_replay_planner_slot_count_is_unanimous_under_gloo() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(target=_run_replay_planner_rank, args=(r, 2, port, [8, 3], q))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, *result = q.get(timeout=50)
        results[rank] = tuple(result)
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    # Both ranks pad to the global maximum slot count; real work comes first
    # and the weights still sum to one per rank.
    assert results[0] == pytest.approx((8, 0, 1.0, True, True, True))
    assert results[1] == pytest.approx((8, 5, 1.0, True, True, True))


class _DistinctReward(RewardFunction):
    """Scores sample ``i`` of a call as ``i * i``: no score equals the group mean,
    so no sample's advantage is zero and none is filtered out."""

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        return RewardOutput(scores=tuple(float(i * i) for i in range(len(samples))))


def _rank_trainer(monkeypatch: pytest.MonkeyPatch, root: Path, *, samples: int) -> Any:
    """The online recipe's trainer wiring on tiny SANA with real cross-rank collectives.

    ``samples`` is this rank's group size, so the two ranks collect unequal
    batches; one training sample per replay slot. The strategy is the
    single-process one carrying the rank's real ``TrainingCollectives``: the
    slot plan and skip decision are reduced over the live gloo group, while the
    policy itself is not wrapped (SANA replay reads ``transformer.config``,
    which a DDP-wrapped transformer does not expose).
    """

    from vrl.scripts.common.factory import AlgorithmEvaluatorPair
    from vrl.trainers.weight_sync import RayRuntimeWeightSyncer

    bench = real_collector(
        monkeypatch,
        root,
        reward=_DistinctReward(),
        overrides=(
            f"rollout.n_samples_per_prompt={samples}",
            "rollout.samples_per_generation_batch=1",
            "actor.training_microbatch_size=1",
        ),
    )
    stack = bench.stack
    built = stack.resolved.built
    bundle = stack.trainer_bundle()
    context = DistributedTrainingContext(
        strategy="ddp",
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        device=torch.device("cpu"),
    )
    strategy = SingleProcessStrategy(context, collectives=TrainingCollectives(context))
    pair = AlgorithmEvaluatorPair.from_configs(
        family_entry=stack.family,
        built=built,
        collector_config=stack.collector_config(),
        scheduler=getattr(bundle, "scheduler", None),
    )
    trainer = OnlineTrainer(
        algorithm=pair.algorithm,
        collector=bench.collector,
        evaluator=pair.evaluator,
        model=bundle.model,
        ref_model=bundle.model,
        weight_syncer=RayRuntimeWeightSyncer(bench.runtime),
        sync_state_getter=lambda: strategy.export_rollout_state(bundle),
        config=built.trainer,
        device=torch.device("cpu"),
        strategy=strategy,
    )
    return trainer


def _run_replay_loop_rank(
    rank: int,
    world_size: int,
    port: int,
    local_counts: list[int],
    root: str,
    q: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    monkeypatch = pytest.MonkeyPatch()
    try:
        # A spawned rank escapes the conftest CUDA pin; this test is CPU-only.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        trainer = _rank_trainer(
            monkeypatch, Path(root) / f"rank-{rank}", samples=local_counts[rank]
        )
        probe = Trace(monkeypatch)
        probe.watch(trainer.evaluator, "evaluate", "evaluate")
        probe.watch(trainer, "_backward", "backward")
        batch = asyncio.run(trainer.collect_training_batch(["a cat"]))
        sample_count = sum(int(b.rewards.shape[0]) for b in batch.batches)
        trainer.begin_optimizer_update()
        trainer.backward_on_training_batch(batch, total_groups=1)
        backward_losses = [
            float(args[0].detach()) for event, args in probe.calls if event == "backward"
        ]
        q.put(
            (
                rank,
                sample_count,
                probe.events.count("evaluate"),
                len(backward_losses),
                len(trainer._update_agg_metrics.losses),
                sum(1 for value in backward_losses if value == 0.0),
            )
        )
        asyncio.run(trainer.rollout_schedule.shutdown())
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


def test_replay_loop_balances_evaluate_and_backward_counts_under_gloo(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(target=_run_replay_loop_rank, args=(r, 2, port, [8, 3], str(tmp_path), q))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, *result = q.get(timeout=120)
        results[rank] = tuple(result)
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0

    samples_0, evaluate_0, backward_0, metrics_0, zero_0 = results[0]
    samples_1, evaluate_1, backward_1, metrics_1, zero_1 = results[1]
    assert (samples_0, samples_1) == (8, 3)
    # Both ranks replay the same number of slots (the larger rank's), each slot
    # over the same trained denoise steps.
    steps = evaluate_0 // 8
    assert steps >= 1
    assert evaluate_0 == evaluate_1 == backward_0 == backward_1 == 8 * steps
    # Only real samples report metrics; the padding slots backpropagate zero.
    assert (metrics_0, zero_0) == (8 * steps, 0)
    assert (metrics_1, zero_1) == (3 * steps, 5 * steps)
