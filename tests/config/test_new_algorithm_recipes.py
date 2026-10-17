"""The new-algorithm recipes compose into runnable configs.

DanceGRPO / Flow-DPPO / GRPO-Guard each ship a base/algorithm config + an
online recipe. These pin that swapping the recipe into a real diffusion
experiment resolves to the right algorithm type and sets the knobs each
algorithm needs (and nothing trips the unknown-key check).
"""

from __future__ import annotations

import pytest

from vrl.algorithms.grpo.continuous import FlowDPPOConfig, GRPOConfig, GRPOGuardConfig
from vrl.config.builders import BuiltConfigs
from vrl.config.loading import load_config

# A complete diffusion GRPO experiment; we swap only its /recipe/online group so
# the new recipe inherits a full model/sampling/reward/dataset environment.
_BASE = "experiment/sd3_5/online_grpo_ocr"


def _load(recipe: str):
    return load_config(_BASE, overrides=[f"/recipe/online={recipe}"])


def test_dance_grpo_recipe_resolves_with_random_timestep_selection() -> None:
    cfg = _load("flow_matching_dance_grpo")
    assert cfg.algorithm.kind == "dance_grpo"
    built = BuiltConfigs.from_cfg(cfg)
    assert isinstance(built.algorithm, GRPOConfig)
    # The defining knob: random per-update timestep subset reaches TrainerConfig.
    assert built.trainer.timestep_selection == "random"


@pytest.mark.parametrize(
    ("preset", "kind", "config_cls"),
    [
        ("flow_matching_dppo", "flow_dppo", FlowDPPOConfig),
        ("flow_matching_grpo_guard", "grpo_guard", GRPOGuardConfig),
    ],
)
def test_trust_region_recipes_resolve_and_enable_proposal_mean_storage(
    preset: str, kind: str, config_cls: type
) -> None:
    # A trust-region objective needs a moving behavior policy: a second epoch
    # over the full batch (config resolution refuses one strict epoch).
    cfg = load_config(
        _BASE,
        overrides=[
            f"/recipe/online={preset}",
            "actor.ppo_epochs=2",
            "actor.prompts_per_collection=0",
        ],
    )
    assert cfg.algorithm.kind == kind
    assert isinstance(BuiltConfigs.from_cfg(cfg).algorithm, config_cls)
    # Required for the trust-region loss; without it generation never stores the
    # rollout proposal mean and the loss fails fast.
    assert cfg.rollout.return_prev_sample_mean is True
