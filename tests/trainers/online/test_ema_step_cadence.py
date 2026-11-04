"""The EMA shadow steps once per optimizer step, at that step's counter.

The trainer is the real online wiring on tiny SANA; the real ``EMAWeights``
shadow is observed, not replaced.
"""

from __future__ import annotations

import asyncio

from tests.rollouts.collector._helpers import Trace
from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.trainers.online.ema import EMAWeights

_GROUP = 3


def _trainer(monkeypatch, tmp_path) -> TrainerBench:
    # One collected group of three samples, one sample per replay microbatch:
    # three microbatches per optimizer update.
    return real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(
            f"rollout.n_samples_per_prompt={_GROUP}",
            f"rollout.samples_per_generation_batch={_GROUP}",
            "actor.training_microbatch_size=1",
            "actor.ppo_epochs=1",
            "actor.drop_zero_advantage=false",
            "actor.ema.enable=true",
            "actor.ema.decay=0.9",
            "actor.ema.update_interval=1",
        ),
    )


def _shadow_counters(monkeypatch) -> Trace:
    trace = Trace(monkeypatch)
    trace.watch(EMAWeights, "step", "ema_step")
    return trace


def _counters(trace: Trace) -> list[int]:
    # args: (shadow, parameters, optimization_step)
    return [int(args[2]) for event, args in trace.calls if event == "ema_step"]


def test_shadow_steps_once_per_optimizer_step_not_per_microbatch(monkeypatch, tmp_path) -> None:
    trace = _shadow_counters(monkeypatch)
    tb = _trainer(monkeypatch, tmp_path)

    asyncio.run(tb.trainer.step(["p1"]))
    asyncio.run(tb.trainer.step(["p1"]))

    assert _counters(trace) == [0, 1]
