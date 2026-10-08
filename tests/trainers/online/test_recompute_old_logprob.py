"""``precision_correction.recompute_old_logprob`` reaches the objective.

Each case resolves a real tiny SANA run and builds the trainer the way the
online recipe does; the factory hands the configured correction to GRPO.
Config resolution refuses ``on`` under off-policy replay
(tests/config/test_algorithm_schedule_soundness.py).
"""

from __future__ import annotations

from tests.trainers.online._helpers import real_trainer

_RECOMPUTE = 'trainer.precision_correction.recompute_old_logprob="on"'


def test_strict_single_epoch_is_accepted(monkeypatch, tmp_path) -> None:
    tb = real_trainer(monkeypatch, tmp_path, overrides=(_RECOMPUTE, "actor.ppo_epochs=1"))

    assert tb.trainer.algorithm.precision_correction.recompute_old_logprob == "on"


def test_off_mode_ignores_the_schedule(monkeypatch, tmp_path) -> None:
    # The tiny recipe trains four PPO epochs; with recomputation off that is fine.
    tb = real_trainer(monkeypatch, tmp_path)

    assert tb.trainer.algorithm.precision_correction.recompute_old_logprob == "off"
    assert tb.trainer.config.ppo_epochs == 4
