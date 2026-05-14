"""Tests for canonical rollout family registry metadata."""

from __future__ import annotations

import pytest

from vrl.models.registry import registered_models
from vrl.rollouts.collector.configs import (
    CosmosPredict2CollectorConfig,
    JanusProCollectorConfig,
    JanusProR1CollectorConfig,
    NextStep1CollectorConfig,
    SD3_5CollectorConfig,
    Wan_2_1CollectorConfig,
)
from vrl.rollouts.collector.factory import (
    COLLECTOR_REGISTRY,
    build_rollout_collector,
    collector_config_cls,
)
from vrl.rollouts.family_registry import (
    DIFFUSION_COMMON_SAMPLING_FIELDS,
    DIFFUSION_MIGRATED_RETURN_ARTIFACTS,
    DIFFUSION_VIDEO_SAMPLING_FIELDS,
    FAMILY_REGISTRY,
    JANUS_PRO_MIGRATED_RETURN_ARTIFACTS,
    JANUS_PRO_R1_MIGRATED_RETURN_ARTIFACTS,
    NEXTSTEP_MIGRATED_RETURN_ARTIFACTS,
    SD3_MIGRATED_RETURN_ARTIFACTS,
    get_rollout_family_entry,
    normalize_rollout_family,
    registered_rollout_families,
)
from vrl.rollouts.packers.trajectory import TrajectoryRolloutPacker


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

    expected_config_cls_path = {
        "sd3_5": "vrl.rollouts.collector.configs:SD3_5CollectorConfig",
        "wan_2_1": "vrl.rollouts.collector.configs:Wan_2_1CollectorConfig",
        "cosmos-predict2": "vrl.rollouts.collector.configs:CosmosPredict2CollectorConfig",
        "cosmos-predict2.5": "vrl.rollouts.collector.configs:CosmosPredict2CollectorConfig",
        "janus_pro": "vrl.rollouts.collector.configs:JanusProCollectorConfig",
        "janus_pro_r1": "vrl.rollouts.collector.configs:JanusProR1CollectorConfig",
        "nextstep_1": "vrl.rollouts.collector.configs:NextStep1CollectorConfig",
    }
    for family, config_cls_path in expected_config_cls_path.items():
        entry = FAMILY_REGISTRY[family]
        assert entry.family == family
        assert entry.task
        assert entry.collector.config_cls == config_cls_path
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
    expected_config_cls = {
        "sd3_5": SD3_5CollectorConfig,
        "wan_2_1": Wan_2_1CollectorConfig,
        "cosmos-predict2": CosmosPredict2CollectorConfig,
        "cosmos-predict2.5": CosmosPredict2CollectorConfig,
        "janus_pro": JanusProCollectorConfig,
        "janus_pro_r1": JanusProR1CollectorConfig,
        "nextstep_1": NextStep1CollectorConfig,
    }
    assert set(COLLECTOR_REGISTRY) == set(FAMILY_REGISTRY)
    for family, family_entry in FAMILY_REGISTRY.items():
        collector_entry = COLLECTOR_REGISTRY[family]
        assert collector_entry.family == family_entry.family
        assert collector_entry.task == family_entry.task
        assert collector_entry.kind == family_entry.collector.kind
        assert collector_entry.config_cls is expected_config_cls[family]
        assert collector_config_cls(family) is expected_config_cls[family]


def test_family_registry_is_derived_from_model_registrations() -> None:
    registered_rollouts = {
        rollout.family
        for model in registered_models()
        for rollout in model.rollouts
    }

    assert registered_rollouts == set(FAMILY_REGISTRY)


def test_diffusion_request_shape_is_registry_declared() -> None:
    assert FAMILY_REGISTRY["sd3_5"].collector.sampling_fields == (DIFFUSION_COMMON_SAMPLING_FIELDS)
    assert FAMILY_REGISTRY["wan_2_1"].collector.sampling_fields == (
        DIFFUSION_VIDEO_SAMPLING_FIELDS
    )
    assert FAMILY_REGISTRY["cosmos-predict2"].collector.sampling_fields == (
        *DIFFUSION_VIDEO_SAMPLING_FIELDS,
        "fps",
    )
    assert FAMILY_REGISTRY["cosmos-predict2.5"].collector.sampling_fields == (
        *DIFFUSION_VIDEO_SAMPLING_FIELDS,
        "fps",
    )
    for family in ("sd3_5", "wan_2_1", "cosmos-predict2", "cosmos-predict2.5"):
        collector = FAMILY_REGISTRY[family].collector
        assert "noise_level" in collector.sampling_fields
        assert "sde_type" in collector.sampling_fields
        assert "return_kl" in collector.sampling_fields
    assert FAMILY_REGISTRY["sd3_5"].collector.return_artifacts == SD3_MIGRATED_RETURN_ARTIFACTS
    for family in ("wan_2_1", "cosmos-predict2", "cosmos-predict2.5"):
        assert FAMILY_REGISTRY[family].collector.return_artifacts == (
            DIFFUSION_MIGRATED_RETURN_ARTIFACTS
        )
    assert FAMILY_REGISTRY["janus_pro"].collector.return_artifacts == (
        JANUS_PRO_MIGRATED_RETURN_ARTIFACTS
    )
    assert FAMILY_REGISTRY["janus_pro_r1"].collector.return_artifacts == (
        JANUS_PRO_R1_MIGRATED_RETURN_ARTIFACTS
    )
    assert FAMILY_REGISTRY["nextstep_1"].collector.return_artifacts == (
        NEXTSTEP_MIGRATED_RETURN_ARTIFACTS
    )


def test_migrated_collectors_use_trajectory_packer_by_default() -> None:
    for family in FAMILY_REGISTRY:
        collector = build_rollout_collector(family, model=None, reward_fn=None)
        assert isinstance(collector.packer, TrajectoryRolloutPacker)


def test_unknown_family_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported rollout family"):
        get_rollout_family_entry("not_a_family")
