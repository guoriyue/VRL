"""``actor.ema.step_per_microbatch``: the reference loop's shadow stepping.

Flash-GRPO's reference calls ``ema.step`` after EVERY replay microbatch with
the optimizer-step counter, and increments that counter before the last
microbatch's call (it steps the optimizer, bumps ``global_step``, then steps
the shadow). With ``n`` microbatches per update the shadow therefore sees
``n - 1`` calls at the current counter and one at the next. Off, the trainer
steps the shadow once per optimizer step at the current counter.

The trainer is the real online wiring on tiny SANA; the real ``EMAWeights``
shadow is observed, not replaced.
"""

from __future__ import annotations

import asyncio

from tests.rollouts.collector._helpers import Trace
from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.trainers.online.ema import EMAWeights

_GROUP = 3


def _trainer(monkeypatch, tmp_path, *, step_per_microbatch: bool) -> TrainerBench:
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
            f"actor.ema.step_per_microbatch={str(step_per_microbatch).lower()}",
        ),
    )


def _shadow_counters(monkeypatch) -> Trace:
    trace = Trace(monkeypatch)
    trace.watch(EMAWeights, "step", "ema_step")
    return trace


def _counters(trace: Trace) -> list[int]:
    # args: (shadow, parameters, optimization_step)
    return [int(args[2]) for event, args in trace.calls if event == "ema_step"]


def test_reference_stepping_calls_the_shadow_per_microbatch(monkeypatch, tmp_path) -> None:
    trace = _shadow_counters(monkeypatch)
    tb = _trainer(monkeypatch, tmp_path, step_per_microbatch=True)

    asyncio.run(tb.trainer.step(["p1"]))
    asyncio.run(tb.trainer.step(["p1"]))

    # Update 0: two microbatch calls at counter 0, the last one after the
    # optimizer step at counter 1; update 1 likewise at 1 then 2.
    assert _counters(trace) == [0, 0, 1, 1, 1, 2]


def test_default_stepping_is_once_per_optimizer_step(monkeypatch, tmp_path) -> None:
    trace = _shadow_counters(monkeypatch)
    tb = _trainer(monkeypatch, tmp_path, step_per_microbatch=False)

    asyncio.run(tb.trainer.step(["p1"]))
    asyncio.run(tb.trainer.step(["p1"]))

    assert _counters(trace) == [0, 1]
