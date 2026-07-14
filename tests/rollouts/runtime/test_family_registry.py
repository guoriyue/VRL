"""Tests for canonical model-family registry behavior."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest
from omegaconf import OmegaConf

from vrl.families.names import (
    _FAMILY_BY_ALIAS,
    normalize_model_family,
)
from vrl.families.registry import (
    FAMILY_REGISTRY,
    ARFamilyBuild,
    DiffusionFamilyBuild,
    get_model_family_entry,
)
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import (
    RolloutCollectorConfig,
    build_rollout_config_from_cfg,
)
from vrl.utils.config import import_from_path


def test_family_entry_rejects_a_collector_kind_build_mismatch() -> None:
    entry = get_model_family_entry("sana")

    with pytest.raises(ValueError, match="does not match its family build"):
        replace(entry, collector_kind="ar_discrete")


def test_family_name_import_does_not_load_runtime_registry() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import vrl.families.names; "
                "assert 'vrl.families.registry' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_family_registry_entries_have_complete_protocol_wiring() -> None:
    assert FAMILY_REGISTRY
    assert len(FAMILY_REGISTRY) == len(set(FAMILY_REGISTRY))

    for family, entry in FAMILY_REGISTRY.items():
        assert entry.family == family
        assert entry.task
        assert callable(entry.new_gatherer().gather_chunks)
        if entry.collector_kind == "diffusion":
            assert isinstance(entry.family_build, DiffusionFamilyBuild)
            assert entry.executor_cls.startswith(
                ("vrl.models.diffusion.", "vrl.generation.diffusion."),
            )
        else:
            assert isinstance(entry.family_build, ARFamilyBuild)
            assert entry.executor_cls.startswith("vrl.models.ar.")


def test_family_aliases_resolve_to_canonical_entries() -> None:
    assert _FAMILY_BY_ALIAS
    for alias, family in _FAMILY_BY_ALIAS.items():
        assert normalize_model_family(alias) == family
        assert get_model_family_entry(alias) is FAMILY_REGISTRY[family]


def test_rollout_config_is_projected_from_yaml() -> None:
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
            },
            "rollout": {
                "samples_per_chunk": 8,
                "noise_level": 1.0,
                "same_latent": False,
                "sde": {
                    "type": "flow_grpo",
                    "window_size": 0,
                    "window_range": [0, 10],
                },
                "trajectory_storage": {"device": "cpu", "dtype": "float16"},
            },
            "algorithm": {"kl_reward_coef": 0.25},
        },
    )

    rollout = build_rollout_config_from_cfg(cfg)

    assert rollout.values["width"] == 1280
    assert rollout.values["num_steps"] == 35
    assert rollout.values["samples_per_chunk"] == 8
    assert rollout.values["sde_window_range"] == [0, 10]
    assert rollout.values["return_kl"] is True
    assert rollout.values["trajectory_storage"] == {
        "device": "cpu",
        "dtype": "float16",
    }


def test_request_sampling_excludes_driver_only_rollout_values() -> None:
    cfg = OmegaConf.create(
        {
            "sampling": {"width": 1280, "num_frames": 93, "fps": 16},
            "rollout": {
                "n_samples_per_prompt": 4,
                "prompts_per_batch": 1,
                "microbatch_size": 1,
                "host_memory_budget_fraction": 0.8,
                "samples_per_chunk": 8,
                "sde": {"type": "flow_grpo", "window_range": [0, 10]},
            },
            "algorithm": {"kl_reward_coef": 0.0},
        },
    )

    sampling = build_rollout_config_from_cfg(cfg).request_sampling()

    assert sampling["width"] == 1280
    assert sampling["samples_per_chunk"] == 8
    assert sampling["sde_type"] == "flow_grpo"
    assert sampling["sde_window_range"] == [0, 10]
    assert sampling["return_kl"] is False
    for driver_key in (
        "host_memory_budget_fraction",
        "kl_reward_coef",
        "microbatch_size",
        "n_samples_per_prompt",
        "prompts_per_batch",
    ):
        assert driver_key not in sampling


def test_all_registry_entries_build_collectors_from_the_same_entry() -> None:
    for entry in FAMILY_REGISTRY.values():
        collector = build_rollout_collector(
            entry,
            reward_fn=None,
            config=RolloutCollectorConfig(
                values={"n_samples_per_prompt": 1, "samples_per_chunk": 1},
            ),
        )
        assert collector.request_builder.entry is entry
        assert callable(collector.collect_unscored)
        assert callable(collector.score_rollouts)


def test_unknown_family_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported model family"):
        get_model_family_entry("not_a_family")


def test_ar_family_build_descriptors_are_importable() -> None:
    ar_entries = [
        entry for entry in FAMILY_REGISTRY.values() if entry.collector_kind != "diffusion"
    ]
    assert len(ar_entries) >= 6
    for entry in ar_entries:
        build = entry.family_build
        assert isinstance(build, ARFamilyBuild)
        for path in (
            build.model_cls,
            build.replay_cls,
            build.config_cls,
            build.config_builder,
        ):
            assert callable(import_from_path(path))


def test_diffusion_family_build_descriptors_have_one_replay_path() -> None:
    diffusion_entries = [
        entry for entry in FAMILY_REGISTRY.values() if entry.collector_kind == "diffusion"
    ]
    assert len(diffusion_entries) >= 10
    for entry in diffusion_entries:
        build = entry.family_build
        assert isinstance(build, DiffusionFamilyBuild)
        if build.replay_cls is not None:
            assert callable(import_from_path(build.replay_cls))
            assert build.transformer_classname
            assert build.replay_runtime_builder is None
        else:
            assert build.replay_runtime_builder
            assert callable(import_from_path(build.replay_runtime_builder))
