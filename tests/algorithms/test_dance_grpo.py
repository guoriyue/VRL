"""DanceGRPO = FlowGRPO loss + random denoise-timestep selection.

verl-omni registers dance_grpo to the same FlowGRPOLoss as flow_grpo, so vrl
reuses the GRPO algorithm/config unchanged; the only behavioral delta is the
trainer's per-update random timestep subset (vs the default strided subset).
These pin (1) the selection helper and (2) that the dance_grpo kind dispatches
to GRPO without inventing a new config type.
"""

from __future__ import annotations

import pytest
import torch

from vrl.trainers.online import OnlineTrainer

_pick = OnlineTrainer._train_timestep_indices


# --------------------------------------------------------- timestep selection


def test_full_fraction_trains_all_steps_regardless_of_selection() -> None:
    # timestep_fraction == 1 -> every step trained, selection is a no-op.
    assert _pick(10, 1.0, "strided") == list(range(10))
    assert _pick(10, 1.0, "random") == list(range(10))


def test_strided_is_deterministic_and_evenly_spaced() -> None:
    a = _pick(10, 0.5, "strided")
    b = _pick(10, 0.5, "strided")
    assert a == b  # fixed every call
    assert a == [0, 2, 4, 6, 8]
    assert len(a) == 5


def test_random_picks_a_valid_sorted_subset_of_the_right_size() -> None:
    torch.manual_seed(0)
    picks = _pick(20, 0.5, "random")
    assert len(picks) == 10
    assert picks == sorted(picks)  # returned sorted; loop sums over indices
    assert len(set(picks)) == 10  # no repeats (randperm, not sampling w/ replace)
    assert all(0 <= i < 20 for i in picks)


def test_random_resamples_across_calls() -> None:
    # Fresh subset each update is the whole point — decorrelates step coverage.
    torch.manual_seed(0)
    first = _pick(50, 0.4, "random")
    second = _pick(50, 0.4, "random")
    assert first != second


def test_at_least_one_step_even_at_tiny_fraction() -> None:
    assert len(_pick(10, 0.0, "strided")) == 1
    assert len(_pick(10, 0.0, "random")) == 1


# ------------------------------------------------ timestep_selection validation


def _trainer_config(**overrides):
    from vrl.trainers.core.types import (
        DebugConfig,
        EMAConfig,
        OptimConfig,
    )
    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

    base = dict(
        batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
        timestep_fraction=1.0,
        drop_zero_advantage=False,
        output_dir="outputs/",
        optim=OptimConfig(lr=0.01),
        ema=EMAConfig(),
        debug=DebugConfig(),
    )
    base.update(overrides)
    return TrainerConfig(**base)


def test_trainer_config_accepts_random() -> None:
    assert _trainer_config(timestep_selection="random").timestep_selection == "random"


def test_trainer_config_rejects_unknown_selection() -> None:
    with pytest.raises(ValueError, match="timestep_selection"):
        _trainer_config(timestep_selection="bogus")


# ---------------------------------------------------------- kind dispatch


def test_dance_grpo_kind_is_accepted() -> None:
    from vrl.config.schema import AlgorithmConfig

    assert AlgorithmConfig(kind="dance_grpo").kind == "dance_grpo"


def test_dance_grpo_builds_a_plain_grpo_config() -> None:
    from omegaconf import OmegaConf

    from vrl.algorithms.grpo.continuous import GRPOConfig
    from vrl.algorithms.grpo.token import TokenGRPOConfig
    from vrl.config.schema import AlgorithmConfig

    cfg = OmegaConf.create({"algorithm": {"kind": "dance_grpo", "clip_ratio": 0.2}})
    built = AlgorithmConfig.model_validate(
        OmegaConf.to_container(cfg.algorithm, resolve=True),
    ).hyperparameters
    assert isinstance(built, GRPOConfig)
    assert not isinstance(built, TokenGRPOConfig)  # not the token variant
