"""The new-algorithm recipes compose into runnable configs.

DanceGRPO / Flow-DPPO / GRPO-Guard each ship a base/algorithm config + an
online recipe. These pin that swapping the recipe into a real diffusion
experiment resolves to the right algorithm type and sets the knobs each
algorithm needs (and nothing trips the unknown-key check).
"""

from __future__ import annotations

import logging

import pytest

from vrl.algorithms.grpo.continuous import FlowDPPOConfig, GRPOConfig, GRPOGuardConfig
from vrl.config.builders import build_algorithm_config, build_configs
from vrl.config.loading import load_config

# A complete diffusion GRPO experiment; we swap only its /recipe/online group so
# the new recipe inherits a full model/sampling/reward/dataset environment.
_BASE = "experiment/diffusion/sd3_5/online_grpo_ocr"


def _load(recipe: str):
    return load_config(_BASE, overrides=[f"/recipe/online={recipe}"])


def test_dance_grpo_recipe_resolves_with_random_timestep_selection() -> None:
    cfg = _load("flow_matching_dance_grpo")
    assert cfg.algorithm.kind == "dance_grpo"
    assert isinstance(build_algorithm_config(cfg), GRPOConfig)
    # The defining knob: random per-update timestep subset reaches TrainerConfig.
    assert build_configs(cfg)["trainer"].timestep_selection == "random"


def test_flow_dppo_recipe_resolves_and_enables_proposal_mean_storage() -> None:
    cfg = _load("flow_matching_dppo")
    assert cfg.algorithm.kind == "flow_dppo"
    assert isinstance(build_algorithm_config(cfg), FlowDPPOConfig)
    # Required for the trust-region loss; without it generation never stores the
    # rollout proposal mean and the loss fails fast.
    assert cfg.rollout.return_prev_sample_mean is True


def test_grpo_guard_recipe_resolves_and_enables_proposal_mean_storage() -> None:
    cfg = _load("flow_matching_grpo_guard")
    assert cfg.algorithm.kind == "grpo_guard"
    assert isinstance(build_algorithm_config(cfg), GRPOGuardConfig)
    assert cfg.rollout.return_prev_sample_mean is True


@pytest.mark.parametrize(
    "recipe",
    ["flow_matching_dance_grpo", "flow_matching_dppo", "flow_matching_grpo_guard"],
)
def test_recipes_have_no_unknown_config_keys(recipe: str, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="vrl.config.unknown_keys"):
        cfg = _load(recipe)
        build_configs(cfg)
    unknown = [r for r in caplog.records if "unknown config keys" in r.getMessage()]
    assert not unknown, [r.getMessage() for r in unknown]
