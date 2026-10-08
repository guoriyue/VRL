"""Several optimizer steps on one collected batch (``actor.optimizer_steps_per_batch``).

Flash-GRPO's reference collects a whole epoch, computes advantages once (global
std over the epoch), shuffles the samples and trains them as two accumulation
windows, each ending in an optimizer step. The trainer reproduces that by
dealing every group's samples across the requested number of updates after ONE
advantage computation; each update prepares its own objective state and steps
the optimizer, and the samples of the updates are disjoint and complete.

Every trainer here is the real online wiring on tiny SANA (``real_trainer``).
Each sample's reward is unique, so the rewards the evaluator replays name the
samples an update trained on; the optimizer step closes an update.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from tests.rollouts.collector._helpers import IndexReward, Trace
from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.rewards import RewardOutput, RewardSample
from vrl.trainers.online import trainer as trainer_module
from vrl.trainers.online.config import OnlineBatchPlan

_GROUP = 4


class _UniqueReward(IndexReward):
    """Every sample ever scored gets its own reward value."""

    def __init__(self) -> None:
        super().__init__()
        self.scored = 0

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        await super().score_batch(samples)
        start = self.scored
        self.scored += len(samples)
        return RewardOutput(scores=tuple(float(start + i) for i in range(len(samples))))


def _trainer(monkeypatch, tmp_path, *, optimizer_steps_per_batch: int) -> TrainerBench:
    return real_trainer(
        monkeypatch,
        tmp_path,
        reward=_UniqueReward(),
        overrides=(
            f"actor.optimizer_steps_per_batch={optimizer_steps_per_batch}",
            "actor.ppo_epochs=1",
            "actor.drop_zero_advantage=false",
            f"rollout.n_samples_per_prompt={_GROUP}",
        ),
    )


def _trace_updates(monkeypatch, tb: TrainerBench) -> Trace:
    trace = Trace(monkeypatch)
    trace.watch(trainer_module, "_compute_rollout_advantages", "advantages")
    trace.watch(tb.trainer.evaluator, "evaluate", "replay")
    trace.watch(tb.trainer, "_clip_and_step", "optimizer_step")
    return trace


def _trained_rewards_per_update(trace: Trace) -> list[set[float]]:
    """The rewards replayed between consecutive optimizer steps."""

    updates: list[set[float]] = [set()]
    for event, args in trace.calls:
        if event == "replay":
            updates[-1].update(args[1].rewards.tolist())
        elif event == "optimizer_step":
            updates.append(set())
    assert updates[-1] == set()
    return updates[:-1]


def test_two_steps_partition_the_batch_after_one_advantage_pass(monkeypatch, tmp_path) -> None:
    tb = _trainer(monkeypatch, tmp_path, optimizer_steps_per_batch=2)
    trace = _trace_updates(monkeypatch, tb)

    async def _run():
        batch = await tb.trainer.collect_training_batch(["p1", "p2"])
        await tb.trainer.train_on_rollout_batch(batch)
        return batch

    batch = asyncio.run(_run())

    assert trace.events.count("advantages") == 1
    assert tb.trainer.state.global_step == 2
    assert tb.trainer.state.step == 1
    # Two updates, each trained on its own sample set; together they are
    # exactly the collected samples, none repeated.
    updates = _trained_rewards_per_update(trace)
    assert len(updates) == 2
    assert not updates[0] & updates[1]
    collected = {reward for group in batch.batches for reward in group.rewards.tolist()}
    assert updates[0] | updates[1] == collected
    # Every group is dealt across both updates (not cut group by group).
    for group in batch.batches:
        members = set(group.rewards.tolist())
        for update in updates:
            assert members & update


def test_one_step_keeps_the_single_update(monkeypatch, tmp_path) -> None:
    tb = _trainer(monkeypatch, tmp_path, optimizer_steps_per_batch=1)
    trace = _trace_updates(monkeypatch, tb)

    asyncio.run(tb.trainer.step(["p1", "p2"]))

    assert tb.trainer.state.global_step == 1
    assert len(_trained_rewards_per_update(trace)) == 1


def test_dealing_is_deterministic_for_a_resumed_counter(monkeypatch, tmp_path) -> None:
    dealt = []
    for name in ("a", "b"):
        # One stack at a time: each tiny SANA stack serves its own snapshot.
        tb = _trainer(monkeypatch, tmp_path / name, optimizer_steps_per_batch=2)
        trace = _trace_updates(monkeypatch, tb)
        asyncio.run(tb.trainer.step(["p1", "p2"]))
        dealt.append(_trained_rewards_per_update(trace))

    assert dealt[0] == dealt[1]


def test_streaming_rejects_several_steps_per_batch() -> None:
    with pytest.raises(ValueError, match="optimizer_steps_per_batch>1"):
        OnlineBatchPlan(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            prompts_per_collection=2,
            optimizer_steps_per_batch=2,
        )
