"""The flow-SDE evaluator's last denoise step receives no loss under any timestep selection.

``strided`` with ``timestep_fraction < 1`` never reached the last step by floor
arithmetic; ``timestep_fraction == 1`` and the random draws did. The trainer
now drops it for the flow-SDE evaluator; other evaluators keep every step.
"""

from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from tests.trainers.online._helpers import _diffusion_rollout_batch, bare_trainer
from vrl.math.denoise.flow_matching import flow_sde_scale_terms
from vrl.rollouts.evaluators.denoise.sde_logprob import DenoiseSDELogProbEvaluator


def _scheduler(steps: int) -> FlowMatchEulerDiscreteScheduler:
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(steps)
    return scheduler


def _batch(steps: int):
    return _diffusion_rollout_batch(
        rewards=torch.zeros(2),
        group_ids=torch.zeros(2, dtype=torch.long),
        num_steps=steps,
    )


def test_the_last_flow_sde_step_is_the_only_near_deterministic_one() -> None:
    """The premise of the rule, on the measured schedules: only the last step
    falls below 1/50 of step 0's noise scale."""

    for steps, noise_level in ((20, 1.0), (20, 0.7), (35, 1.0), (35, 0.7)):
        scheduler = _scheduler(steps)
        _, _, scale = flow_sde_scale_terms(
            scheduler,
            scheduler.timesteps,
            noise_level=noise_level,
            step_index=list(range(steps)),
        )
        assert scale[-1] / scale[0] < 0.02 < scale[-2] / scale[0]


def test_flow_sde_training_skips_the_last_step_and_other_evaluators_keep_it() -> None:
    batch = _batch(20)
    flow = bare_trainer(evaluator=DenoiseSDELogProbEvaluator(scheduler=_scheduler(20)))
    assert flow._train_replay_indices(batch, 1.0, "strided") == list(range(19))
    assert flow._train_replay_indices(batch, 0.5, "strided") == [2 * i for i in range(10)]
    torch.manual_seed(0)
    random_pick = flow._train_replay_indices(batch, 1.0, "random")
    assert 19 not in random_pick and len(random_pick) == 19

    ddim = bare_trainer(
        evaluator=DenoiseSDELogProbEvaluator(scheduler=_scheduler(20), sde_type="ddim")
    )
    assert ddim._train_replay_indices(batch, 1.0, "strided") == list(range(20))
    other = bare_trainer(evaluator=object())
    assert other._train_replay_indices(batch, 1.0, "strided") == list(range(20))
