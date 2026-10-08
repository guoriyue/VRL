"""The schedules a trust-region algorithm accepts actually train.

Flow-DPPO / GRPO-Guard are *defined* by a clipped/guarded importance ratio
r = pi_new/pi_old; config resolution refuses the strict single-epoch schedule
that makes r identically 1 (tests/config/test_algorithm_schedule_soundness.py).
These cases run the schedules it accepts: a second epoch or continuous staleness
moves the behavior policy, and plain GRPO stays valid at one epoch.

Every trainer here is the online recipe's own wiring on the tiny SANA run; the
algorithm is selected by swapping the recipe preset, exactly as a user would.
"""

from __future__ import annotations

import asyncio
import math

import torch

from tests.trainers.online._helpers import real_trainer

_FLOW_DPPO = "/recipe/online=flow_matching_dppo"
_GRPO_GUARD = "/recipe/online=flow_matching_grpo_guard"
_CONTINUOUS = "/base/rollout/orchestration=continuous"


def _trains_one_step(bench) -> None:
    """The accepted configuration runs a real update that moves the policy."""

    before = {name: value.detach().clone() for name, value in bench.trainable_parameters().items()}

    async def step():
        try:
            return await bench.trainer.step(["a cat", "a dog"])
        finally:
            await bench.trainer.rollout_schedule.shutdown()

    metrics = asyncio.run(step())
    after = bench.trainable_parameters()

    assert math.isfinite(metrics.loss)
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_trust_region_with_multi_epoch_is_allowed(monkeypatch, tmp_path) -> None:
    # ppo_epochs>1 lets the policy move between epochs, so the ratio engages.
    bench = real_trainer(monkeypatch, tmp_path, overrides=(_FLOW_DPPO, "actor.ppo_epochs=2"))

    _trains_one_step(bench)


def test_plain_grpo_single_epoch_is_allowed(monkeypatch, tmp_path) -> None:
    # Plain GRPO at one epoch is honest REINFORCE+group-baseline, not a no-op.
    bench = real_trainer(monkeypatch, tmp_path, overrides=("actor.ppo_epochs=1",))

    _trains_one_step(bench)


def test_continuous_with_staleness_allows_single_epoch_trust_region(monkeypatch, tmp_path) -> None:
    # A stale behavior policy makes the ratio differ from 1 even at one epoch.
    bench = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(_FLOW_DPPO, "actor.ppo_epochs=1", _CONTINUOUS),
    )

    _trains_one_step(bench)
