"""Whole-tree unknown-key walker: every depth, one mechanism."""

from __future__ import annotations

import logging

from omegaconf import OmegaConf

from vrl.config.unknown_keys import find_unknown_keys, warn_unknown_keys


def test_unknown_keys_are_found_at_every_depth() -> None:
    """Typos at top level, section level, and nested blocks are all named."""
    cfg = OmegaConf.create(
        {
            "samplng": {"num_steps": 10},
            "sampling": {"num_stps": 5},
            "rollout": {"sde": {"type": "sde", "window_sze": 3}},
            "distributed": {
                "resources": {"reward": {"num_gpus": 1, "share_with_rolout": True}},
            },
            "actor": {"mixed_precision": "bf16", "optim": {"lr": 1e-4, "lrr": 2}},
        },
    )
    unknown = find_unknown_keys(cfg)
    assert unknown == [
        "actor.mixed_precision",
        "actor.optim.lrr",
        "distributed.resources.reward.share_with_rolout",
        "rollout.sde.window_sze",
        "sampling.num_stps",
        "samplng",
    ]


def test_open_blocks_accept_arbitrary_keys() -> None:
    """worker_config / non-modeled reward kwargs / cosmos are open by design."""
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"ocr": 1.0},
                "kwargs": {
                    "ocr": {"anything": 1},
                    "kling_video_reward": {
                        "execution": "pool",
                        "reward_name": "r",
                        "score_key": "s",
                        "worker_config": {"any_worker_key": True},
                    },
                },
            },
            "cosmos": {"reference_mode": "global"},
        },
    )
    assert find_unknown_keys(cfg) == []


def test_warn_unknown_keys_logs_one_line(caplog) -> None:
    cfg = OmegaConf.create({"sampling": {"num_stps": 5}})
    with caplog.at_level(logging.WARNING):
        unknown = warn_unknown_keys(cfg)
    assert unknown == ["sampling.num_stps"]
    assert any("sampling.num_stps" in r.getMessage() for r in caplog.records)


# ── Anti-rot sweeps: the registry must track the real code and real configs ──


def test_every_config_path_read_by_code_is_registered() -> None:
    """Anti-rot: cfg paths the code reads must exist in the known-key tree."""
    from vrl.config.lint import unregistered_code_paths

    assert unregistered_code_paths() == []


def test_all_experiment_configs_have_zero_unknown_keys() -> None:
    """Anti-rot: every shipped experiment must load with zero unknown keys."""
    from vrl.config.lint import unknown_yaml_keys

    assert unknown_yaml_keys() == {}
