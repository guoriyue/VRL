"""Tests for canonical rollout family registry metadata."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.rollouts.collector.factory import (
    COLLECTOR_REGISTRY,
    build_rollout_collector,
)
from vrl.rollouts.family_registry import (
    FAMILY_REGISTRY,
    get_rollout_family_entry,
    normalize_rollout_family,
    registered_rollout_families,
)
from vrl.rollouts.settings import RolloutSettings
from vrl.scripts.common.factory import build_collector_config_from_cfg


def test_family_registry_covers_current_rollout_families() -> None:
    assert registered_rollout_families() == (
        "sd3_5",
        "wan_2_1",
        "cosmos-predict2",
        "cosmos-predict2.5",
        "janus_pro",
        "janus_pro_r1",
        "nextstep_1",
    )

    for family in registered_rollout_families():
        entry = FAMILY_REGISTRY[family]
        assert entry.family == family
        assert entry.task
        assert entry.collector.request_prefix
        assert entry.executor_cls.startswith("vrl.models.families.")
        assert entry.runtime_builder.startswith("vrl.models.families.")
        assert entry.runtime_spec_extractor.startswith("vrl.models.families.")
        assert ":" in entry.gatherer.import_path


def test_family_aliases_resolve_to_canonical_entries() -> None:
    expected_aliases = {
        "wan": "wan_2_1",
        "cosmos": "cosmos-predict2",
        "cosmos_predict2": "cosmos-predict2",
        "cosmos_predict2_5": "cosmos-predict2.5",
        "janus": "janus_pro",
        "janus_r1": "janus_pro_r1",
        "nextstep": "nextstep_1",
    }
    for alias, expected in expected_aliases.items():
        assert normalize_rollout_family(alias) == expected
        assert get_rollout_family_entry(alias) is FAMILY_REGISTRY[expected]


def test_collector_registry_reuses_family_registry_metadata() -> None:
    assert set(COLLECTOR_REGISTRY) == set(FAMILY_REGISTRY)
    for family, family_entry in FAMILY_REGISTRY.items():
        collector_entry = COLLECTOR_REGISTRY[family]
        assert collector_entry.family == family_entry.family
        assert collector_entry.task == family_entry.task
        assert collector_entry.kind == family_entry.collector.kind
        assert collector_entry.request_prefix == family_entry.collector.request_prefix
        assert collector_entry.return_artifacts == family_entry.collector.return_artifacts


def test_rollout_settings_are_projected_from_yaml() -> None:
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
                    "type": "sde",
                    "window_size": 0,
                    "window_range": [0, 10],
                },
            },
            "algorithm": {"kl_reward": 0.25},
        },
    )

    settings = build_collector_config_from_cfg(cfg, "cosmos-predict2")

    assert settings.width == 1280
    assert settings.num_steps == 35
    assert settings.sample_batch_size == 8
    assert settings.sde_window_range == (0, 10)
    assert settings.return_kl is True


def test_request_sampling_is_projected_from_resolved_yaml_settings() -> None:
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
                "n": 4,
                "rollout_batch_size": 1,
                "sample_batch_size": 8,
                "noise_level": 1.0,
                "same_latent": False,
                "sde": {
                    "type": "sde",
                    "window_size": 0,
                    "window_range": [0, 10],
                },
            },
            "algorithm": {"kl_reward": 0.0},
        },
    )

    sampling = build_collector_config_from_cfg(
        cfg,
        "cosmos-predict2",
    ).request_sampling()

    assert sampling["width"] == 1280
    assert sampling["num_frames"] == 93
    assert sampling["fps"] == 16
    assert sampling["sample_batch_size"] == 8
    assert sampling["sde_type"] == "sde"
    assert sampling["sde_window_range"] == (0, 10)
    assert sampling["return_kl"] is False
    assert "kl_reward" not in sampling
    assert "n" not in sampling
    assert "rollout_batch_size" not in sampling


def test_registry_keeps_return_artifacts_as_wiring_metadata() -> None:
    for family in FAMILY_REGISTRY:
        assert FAMILY_REGISTRY[family].collector.return_artifacts == (
            "output",
            "trajectory",
        )


def test_migrated_collectors_build_direct_trajectory_collectors() -> None:
    for family in FAMILY_REGISTRY:
        collector = build_rollout_collector(
            family,
            model=None,
            reward_fn=None,
            config=RolloutSettings(
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
    with pytest.raises(ValueError, match="unsupported rollout family"):
        get_rollout_family_entry("not_a_family")
