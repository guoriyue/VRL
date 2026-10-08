"""OnlineTrainer trainable-state / weight-sync wiring: pre-collect sync ordering and getter requirement.

Both run on the online recipe's own wiring over the tiny SANA stack: the real
strategy export is the payload, the real weight syncer pushes it, and the
in-process rollout runtime installs it.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from tests.trainers.online._helpers import real_trainer


def test_initial_rollout_weight_sync_happens_before_collect(monkeypatch, tmp_path) -> None:
    """The first step syncs the driver's trainable state to the rollout before collecting;
    the post-train sync then publishes the updated weights."""

    bench = real_trainer(monkeypatch, tmp_path, overrides=("actor.ppo_epochs=1",))
    initial = {
        f"transformer.{name}": value.detach().clone()
        for name, value in bench.trainable_parameters().items()
    }

    asyncio.run(bench.trainer.step(["a cat"]))

    trace = bench.collector.trace
    assert trace.events.index("update_weights") < trace.events.index("generate")
    pushes = [args for event, args in trace.calls if event == "update_weights"]
    assert [version for _state, version in pushes] == [1, 2]
    first, second = (state for state, _version in pushes)
    assert first.keys() == initial.keys()
    assert all(torch.equal(first[key], initial[key]) for key in initial)
    live = {
        f"transformer.{name}": value.detach()
        for name, value in bench.trainable_parameters().items()
    }
    assert any(not torch.equal(live[key], initial[key]) for key in initial)
    assert all(torch.equal(second[key], live[key]) for key in live)


def test_weight_sync_requires_explicit_trainable_state_getter(monkeypatch, tmp_path) -> None:
    """A trainer with a weight syncer but no trainable-state getter is refused at
    construction, before any collect could run."""

    from vrl.trainers.online.trainer import OnlineTrainer
    from vrl.trainers.weight_sync import RayRuntimeWeightSyncer

    bench = real_trainer(monkeypatch, tmp_path)
    trainer = bench.trainer

    with pytest.raises(ValueError, match="trainable-state getter"):
        OnlineTrainer(
            algorithm=trainer.algorithm,
            collector=bench.collector.collector,
            evaluator=trainer.evaluator,
            model=bench.model,
            weight_syncer=RayRuntimeWeightSyncer(bench.collector.runtime),
            config=trainer.config,
            device="cpu",
            strategy=bench.strategy,
        )

    assert bench.collector.trace.events == []
