"""``precision_correction.recompute_old_logprob=on`` is only sound when the
replay forward IS the behavior policy: one PPO epoch over a fresh rollout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.trainers.online.trainer import OnlineTrainer


def _config(*, ppo_epochs: int, schedule_mode: str, max_stale: int, mode: str = "on"):
    return SimpleNamespace(
        precision_correction=PrecisionCorrectionConfig(recompute_old_logprob=mode),
        ppo_epochs=ppo_epochs,
        rollout_orchestration=SimpleNamespace(
            schedule_mode=schedule_mode,
            continuous=SimpleNamespace(max_stale_policy_versions=max_stale),
        ),
    )


def test_strict_single_epoch_is_accepted() -> None:
    OnlineTrainer._validate_recompute_old_logprob(
        _config(ppo_epochs=1, schedule_mode="strict_on_policy", max_stale=0)
    )


def test_multiple_ppo_epochs_are_refused() -> None:
    with pytest.raises(ValueError, match="ppo_epochs=2"):
        OnlineTrainer._validate_recompute_old_logprob(
            _config(ppo_epochs=2, schedule_mode="strict_on_policy", max_stale=0)
        )


def test_continuous_staleness_is_refused() -> None:
    with pytest.raises(ValueError, match="max_stale_policy_versions=1"):
        OnlineTrainer._validate_recompute_old_logprob(
            _config(ppo_epochs=1, schedule_mode="continuous", max_stale=1)
        )


def test_off_mode_ignores_the_schedule() -> None:
    OnlineTrainer._validate_recompute_old_logprob(
        _config(ppo_epochs=4, schedule_mode="continuous", max_stale=2, mode="off")
    )
