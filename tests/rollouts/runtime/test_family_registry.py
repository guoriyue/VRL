"""Tests for canonical model-family registry behavior."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest
from omegaconf import OmegaConf

from vrl.families.names import (
    _FAMILY_BY_ALIAS,
    normalize_model_family,
)
from vrl.families.registry import (
    FAMILY_REGISTRY,
    DenoiseFamilyBuild,
    TokenFamilyBuild,
    get_model_family_entry,
)
from vrl.families.semantics import GenerationRegime, PolicySemantics
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import (
    RolloutCollectorConfig,
    build_rollout_config_from_cfg,
)
from vrl.trajectory import TrajectoryStoragePolicy
from vrl.utils.config import cfg_get, import_from_path


def test_family_entry_rejects_a_policy_step_build_mismatch() -> None:
    entry = get_model_family_entry("sana")

    with pytest.raises(ValueError, match="does not match its family build"):
        replace(
            entry,
            policy_semantics=PolicySemantics(
                generation_regime="token_autoregressive",
                step_kind="token",
                action_distribution="categorical",
                trajectory_layout="token",
            ),
        )


def test_generation_regime_vocabulary_uses_paper_familiar_names() -> None:
    assert set(get_args(GenerationRegime)) == {
        "full_sequence",
        "token_autoregressive",
        "chunk_autoregressive",
    }


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


def test_production_code_does_not_reference_legacy_taxonomy_paths() -> None:
    root = Path(__file__).resolve().parents[3] / "vrl"
    legacy_directories = (
        root / "models" / "ar",
        root / "models" / "diffusion",
        root / "generation" / "ar",
        root / "generation" / "diffusion",
        root / "math" / "ar",
        root / "math" / "diffusion",
        root / "rollouts" / "evaluators" / "ar",
        root / "rollouts" / "evaluators" / "diffusion",
        root / "scripts" / "ar",
        root / "scripts" / "diffusion",
    )
    stale_directories = [path.relative_to(root) for path in legacy_directories if path.exists()]
    assert not stale_directories, f"legacy taxonomy directories: {stale_directories}"

    legacy_paths = (
        "vrl.models." + "ar",
        "vrl.models." + "diffusion",
        "vrl.generation." + "ar",
        "vrl.generation." + "diffusion",
        "vrl.math." + "ar",
        "vrl.math." + "diffusion",
    )
    violations = [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if any(legacy_path in path.read_text(encoding="utf-8") for legacy_path in legacy_paths)
    ]

    assert not violations, f"legacy taxonomy path references: {violations}"


def test_family_registry_entries_have_complete_protocol_wiring() -> None:
    assert FAMILY_REGISTRY
    assert len(FAMILY_REGISTRY) == len(set(FAMILY_REGISTRY))

    for family, entry in FAMILY_REGISTRY.items():
        assert entry.family == family
        assert entry.task
        assert callable(entry.new_gatherer().gather_chunks)
        assert entry.policy_semantics.generation_regime in {
            "full_sequence",
            "token_autoregressive",
            "chunk_autoregressive",
        }
        if entry.policy_semantics.step_kind == "denoise":
            assert isinstance(entry.family_build, DenoiseFamilyBuild)
            assert entry.executor_cls.startswith(
                (
                    "vrl.models.families.",
                    "vrl.generation.bindings.full_sequence_denoise.",
                ),
            )
        else:
            assert isinstance(entry.family_build, TokenFamilyBuild)
            assert entry.executor_cls.startswith(
                ("vrl.models.families.", "vrl.generation.bindings.token_autoregressive."),
            )


def test_chunk_family_trainability_and_process_boundaries_are_explicit() -> None:
    causvid = get_model_family_entry("causvid").runtime_capabilities
    magi = get_model_family_entry("magi_1").runtime_capabilities

    assert causvid.supports_policy_replay is True
    assert causvid.runs_in_isolated_subprocess is False
    assert magi.supports_policy_replay is False
    assert magi.runs_in_isolated_subprocess is True


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

    assert rollout.request_sampling["width"] == 1280
    assert rollout.request_sampling["num_steps"] == 35
    assert rollout.request_sampling["samples_per_chunk"] == 8
    assert rollout.request_sampling["sde_window_range"] == [0, 10]
    assert rollout.request_sampling["return_kl"] is True
    assert rollout.kl_reward_coef == pytest.approx(0.25)
    assert rollout.trajectory_storage == TrajectoryStoragePolicy(
        device="cpu",
        dtype="float16",
    )
    assert rollout.generation_sampling()["trajectory_storage"] == {
        "device": "cpu",
        "dtype": "float16",
    }


def test_request_sampling_projects_only_generation_owned_rollout_values() -> None:
    cfg = OmegaConf.create(
        {
            "sampling": {"width": 1280, "num_frames": 93, "fps": 16},
            "rollout": {
                "n_samples_per_prompt": 4,
                "prompts_per_batch": 1,
                "samples_per_chunk": 8,
                "sde": {"type": "flow_grpo", "window_range": [0, 10]},
            },
            "algorithm": {"kl_reward_coef": 0.0},
        },
    )

    rollout = build_rollout_config_from_cfg(cfg)
    sampling = rollout.generation_sampling()

    assert sampling["width"] == 1280
    assert sampling["samples_per_chunk"] == 8
    assert sampling["sde_type"] == "flow_grpo"
    assert sampling["sde_window_range"] == [0, 10]
    assert sampling["return_kl"] is False
    for driver_key in ("kl_reward_coef", "n_samples_per_prompt", "prompts_per_batch"):
        assert driver_key not in sampling
    assert "trajectory_storage" not in sampling
    assert rollout.trajectory_storage == TrajectoryStoragePolicy()


def test_collector_config_get_adapts_local_and_request_state() -> None:
    rollout = RolloutCollectorConfig(
        request_sampling={"num_steps": 4},
        kl_reward_coef=0.5,
        trajectory_storage=TrajectoryStoragePolicy(device="cpu"),
    )

    assert cfg_get(rollout, "num_steps") == 4
    assert cfg_get(rollout, "kl_reward_coef") == pytest.approx(0.5)
    assert cfg_get(rollout, "trajectory_storage") == TrajectoryStoragePolicy(
        device="cpu",
    )
    assert cfg_get(rollout, "missing", "fallback") == "fallback"


def test_all_registry_entries_build_collectors_from_the_same_entry() -> None:
    for entry in FAMILY_REGISTRY.values():
        collector = build_rollout_collector(
            entry,
            reward_fn=None,
            config=RolloutCollectorConfig(
                request_sampling={"samples_per_chunk": 1},
            ),
        )
        assert collector.request_builder.entry is entry
        assert callable(collector.collect_unscored)
        assert callable(collector.score_rollouts)


def test_unknown_family_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported model family"):
        get_model_family_entry("not_a_family")


def test_token_family_build_descriptors_are_importable() -> None:
    token_entries = [
        entry
        for entry in FAMILY_REGISTRY.values()
        if isinstance(entry.family_build, TokenFamilyBuild)
    ]
    assert len(token_entries) >= 6
    for entry in token_entries:
        build = entry.family_build
        assert isinstance(build, TokenFamilyBuild)
        for path in (
            build.model_cls,
            build.replay_cls,
            build.config_cls,
            build.config_builder,
        ):
            assert callable(import_from_path(path))


def test_denoise_family_build_descriptors_have_one_replay_path() -> None:
    denoise_entries = [
        entry
        for entry in FAMILY_REGISTRY.values()
        if isinstance(entry.family_build, DenoiseFamilyBuild)
    ]
    assert len(denoise_entries) >= 10
    for entry in denoise_entries:
        build = entry.family_build
        assert isinstance(build, DenoiseFamilyBuild)
        if build.replay_cls is not None:
            assert callable(import_from_path(build.replay_cls))
            assert build.transformer_classname
            assert build.replay_runtime_builder is None
        else:
            assert build.replay_runtime_builder
            assert callable(import_from_path(build.replay_runtime_builder))
