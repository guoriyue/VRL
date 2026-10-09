"""An algorithm's schedule requirements are enforced when the config resolves.

Each objective's config class declares, in its ``requirements``, whether it
scores against the current weights (no off-policy staleness) and whether its
loss is an active trust region.
Resolution rejects a schedule that breaks either declaration, and a
``recompute_old_logprob`` correction under off-policy replay, before any model
or Ray worker exists. Every case swaps presets on a real SD3.5 experiment.
"""

from __future__ import annotations

import pytest

from vrl.config.builders import build_configs
from vrl.config.loading import load_config

_BASE = "experiment/sd3_5/online_grpo_ocr"
_CONTINUOUS = "/base/rollout/orchestration=continuous"
_RECOMPUTE = 'trainer.precision_correction.recompute_old_logprob="on"'
# Several epochs or updates per batch replay a whole collected batch, which
# needs the full-batch path rather than the base experiment's streaming.
_FULL_BATCH = "actor.prompts_per_collection=0"


def _build(*overrides: str, experiment: str = _BASE):
    return build_configs(load_config(experiment, overrides=list(overrides)))


@pytest.mark.parametrize("recipe", ["flow_matching_dppo", "flow_matching_grpo_guard"])
def test_trust_region_needs_a_moving_behavior_policy(recipe: str) -> None:
    """Strict scheduling at one epoch makes the ratio identically 1: a no-op trust region."""

    with pytest.raises(ValueError, match="equivalent to plain GRPO"):
        _build(f"/recipe/online={recipe}", "actor.ppo_epochs=1")

    multi_epoch = _build(f"/recipe/online={recipe}", "actor.ppo_epochs=2", _FULL_BATCH)
    assert multi_epoch.trainer.ppo_epochs == 2
    continuous = _build(f"/recipe/online={recipe}", "actor.ppo_epochs=1", _CONTINUOUS)
    assert continuous.trainer.rollout_orchestration.schedule_mode == "continuous"


def test_plain_grpo_single_epoch_is_a_valid_schedule() -> None:
    """Plain GRPO's clip is a safety rail, so one strict epoch is honest REINFORCE."""

    assert _build("actor.ppo_epochs=1").trainer.ppo_epochs == 1


@pytest.mark.parametrize("recipe", ["flow_matching_dppo", "flow_matching_grpo_guard"])
def test_trust_region_needs_the_stored_rollout_proposal_mean(recipe: str) -> None:
    with pytest.raises(ValueError, match=r"rollout\.return_prev_sample_mean=true"):
        _build(
            f"/recipe/online={recipe}",
            "actor.ppo_epochs=2",
            _FULL_BATCH,
            "rollout.return_prev_sample_mean=false",
        )


def test_current_policy_objective_rejects_continuous_staleness() -> None:
    """DiffusionNFT's behaviour policy is the current weights; a stale sample has none."""

    with pytest.raises(ValueError, match="scores against the current weights"):
        _build(_CONTINUOUS, experiment="experiment/flux/online_diffusion_nft_pickscore_validation")


@pytest.mark.parametrize(
    "schedule",
    [
        ("actor.ppo_epochs=2", _FULL_BATCH),
        ("actor.ppo_epochs=1", "actor.optimizer_steps_per_batch=2", _FULL_BATCH),
        ("actor.ppo_epochs=1", _CONTINUOUS),
    ],
)
def test_recomputed_old_logprob_requires_on_policy_replay(schedule) -> None:
    """The pre-update replay is the behavior policy only on the first pass of a fresh batch."""

    with pytest.raises(ValueError, match="recompute_old_logprob='on'"):
        _build(_RECOMPUTE, *schedule)

    on_policy = _build(_RECOMPUTE, "actor.ppo_epochs=1").trainer
    assert on_policy.precision_correction.recompute_old_logprob == "on"
