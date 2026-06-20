"""Tests for canonical rollout family registry metadata."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import RolloutConfig, build_rollout_config_from_cfg
from vrl.rollouts.families import (
    FAMILY_REGISTRY,
    get_rollout_family_entry,
    normalize_rollout_family,
    registered_rollout_families,
)


def test_family_registry_covers_current_rollout_families() -> None:
    """Checks family registry covers current rollout families."""
    assert registered_rollout_families() == (
        "sd3_5",
        "wan_2_1",
        "wan_2_1_i2v",
        "cosmos-predict2",
        "cosmos-predict2.5",
        "cosmos-predict2-anima",
        "janus_pro",
        "janus_pro_r1",
        "nextstep_1",
    )

    for family in registered_rollout_families():
        entry = FAMILY_REGISTRY[family]
        expected_model_prefix = (
            "vrl.models.diffusion."
            if entry.collector.kind == "diffusion"
            else "vrl.models.ar."
        )
        assert entry.family == family
        assert entry.task
        assert entry.collector.request_prefix
        assert entry.executor_cls.startswith(expected_model_prefix)
        assert entry.runtime_builder.startswith(expected_model_prefix)
        assert entry.runtime_spec_extractor.startswith(expected_model_prefix)
        assert ":" in entry.gatherer.import_path


def test_family_aliases_resolve_to_canonical_entries() -> None:
    """Checks family aliases resolve to canonical entries."""
    expected_aliases = {
        "wan": "wan_2_1",
        "wan_i2v": "wan_2_1_i2v",
        "cosmos": "cosmos-predict2",
        "cosmos_predict2": "cosmos-predict2",
        "cosmos_predict2_5": "cosmos-predict2.5",
        "anima": "cosmos-predict2-anima",
        "cosmos_anima": "cosmos-predict2-anima",
        "janus": "janus_pro",
        "janus_r1": "janus_pro_r1",
        "nextstep": "nextstep_1",
    }
    for alias, expected in expected_aliases.items():
        assert normalize_rollout_family(alias) == expected
        assert get_rollout_family_entry(alias) is FAMILY_REGISTRY[expected]


def test_rollout_config_is_projected_from_yaml() -> None:
    """Checks rollout config is projected from YAML."""
    cfg = OmegaConf.create(
        {
            "sampling": {
                "width": 1280,
                "height": 704,
                "num_frames": 93,
                "num_steps": 35,
                "guidance_scale": 7.0,
                "max_sequence_length": 512,
                "fps": 16,
                "cfg": True,
            },
            "rollout": {
                "sample_batch_size": 8,
                "noise_level": 1.0,
                "same_latent": False,
                "sde": {
                    "type": "flow_grpo",
                    "window_size": 0,
                    "window_range": [0, 10],
                },
            },
            "algorithm": {"kl_reward_coef": 0.25},
        },
    )

    rollout_config = build_rollout_config_from_cfg(cfg, family="cosmos-predict2")

    assert rollout_config.require("width") == 1280
    assert rollout_config.require("num_steps") == 35
    assert rollout_config.require("sample_batch_size") == 8
    assert rollout_config.require("sde_window_range") == (0, 10)
    assert rollout_config.require("return_kl") is True


def test_request_sampling_is_projected_from_resolved_yaml_config() -> None:
    """Checks request sampling is projected from resolved YAML config."""
    cfg = OmegaConf.create(
        {
            "sampling": {
                "width": 1280,
                "height": 704,
                "num_frames": 93,
                "num_steps": 35,
                "guidance_scale": 7.0,
                "fps": 16,
                "cfg": True,
            },
            "rollout": {
                "n_samples_per_prompt": 4,
                "rollout_batch_size": 1,
                "sample_batch_size": 8,
                "reward_view": "image",
                "noise_level": 1.0,
                "same_latent": False,
                "sde": {
                    "type": "flow_grpo",
                    "window_size": 0,
                    "window_range": [0, 10],
                },
            },
            "algorithm": {"kl_reward_coef": 0.0},
        },
    )

    sampling = build_rollout_config_from_cfg(
        cfg,
        family="cosmos-predict2",
    ).request_sampling()

    assert sampling["width"] == 1280
    assert sampling["num_frames"] == 93
    assert sampling["fps"] == 16
    assert sampling["sample_batch_size"] == 8
    assert sampling["sde_type"] == "flow_grpo"
    assert sampling["sde_window_range"] == (0, 10)
    assert sampling["return_kl"] is False
    assert "kl_reward_coef" not in sampling
    assert "n" not in sampling
    assert "reward_view" not in sampling
    assert "rollout_batch_size" not in sampling


def test_registry_keeps_return_artifacts_as_wiring_metadata() -> None:
    """Checks registry keeps return artifacts as wiring metadata."""
    for family in FAMILY_REGISTRY:
        assert FAMILY_REGISTRY[family].collector.return_artifacts == (
            "output",
            "trajectory",
        )


def test_migrated_collectors_build_direct_trajectory_collectors() -> None:
    """Checks migrated collectors build direct trajectory collectors."""
    for family in FAMILY_REGISTRY:
        collector = build_rollout_collector(
            family,
            model=None,
            reward_fn=None,
            config=RolloutConfig(
                family=family,
                values={
                    "n_samples_per_prompt": 1,
                    "sample_batch_size": 1,
                },
            ),
        )
        assert collector.family == family
        assert callable(collector.collect)


def test_unknown_family_raises_clear_error() -> None:
    """Checks unknown family raises clear error."""
    with pytest.raises(ValueError, match="unsupported rollout family"):
        get_rollout_family_entry("not_a_family")
