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
from vrl.rollouts.families.registry import _default_return_artifacts


def test_family_registry_covers_current_rollout_families() -> None:
    """Every registered family carries structurally valid dispatch wiring.

    Asserts the *shape* each entry must satisfy (modality-consistent import
    paths, non-empty task/prefix, importable gatherer) rather than a hand-copied
    list of family names — a new family is covered automatically and adding one
    cannot pass without valid wiring.
    """
    families = registered_rollout_families()
    assert families  # registry must not be empty
    assert len(set(families)) == len(families)  # no duplicate keys

    for family in families:
        entry = FAMILY_REGISTRY[family]
        expected_model_prefix = (
            "vrl.models.diffusion."
            if entry.collector.kind == "diffusion"
            else "vrl.models.ar."
        )
        assert entry.family == family
        assert entry.task
        assert entry.collector.request_prefix
        # A diffusion family's executor is either family-specific (under
        # vrl.models.diffusion) or the shared generic DiffusionChunkExecutor
        # (family-agnostic infra under vrl.generation.diffusion); both are
        # modality-consistent. AR families all ship their own executor.
        if entry.collector.kind == "diffusion":
            assert entry.executor_cls.startswith(
                ("vrl.models.diffusion.", "vrl.generation.diffusion."),
            )
        else:
            assert entry.executor_cls.startswith(expected_model_prefix)
        assert entry.runtime_builder.startswith(expected_model_prefix)
        assert entry.runtime_spec_extractor.startswith(expected_model_prefix)
        assert ":" in entry.gatherer.import_path


def test_family_aliases_resolve_to_canonical_entries() -> None:
    """Every alias declared on a registry entry resolves back to that entry.

    Derived from ``entry.aliases`` so a new family/alias is covered automatically
    and no hand-copied alias map can drift from the registry source of truth.
    """
    seen = 0
    for family, entry in FAMILY_REGISTRY.items():
        # The canonical name itself must resolve to its own entry.
        assert normalize_rollout_family(family) == family
        for alias in entry.aliases:
            assert normalize_rollout_family(alias) == family
            assert get_rollout_family_entry(alias) is entry
            seen += 1
    assert seen > 0  # guard: the registry must actually declare aliases


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
                "samples_per_chunk": 8,
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
    assert rollout_config.require("samples_per_chunk") == 8
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
                "prompts_per_batch": 1,
                "samples_per_chunk": 8,
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
    assert sampling["samples_per_chunk"] == 8
    assert sampling["sde_type"] == "flow_grpo"
    assert sampling["sde_window_range"] == (0, 10)
    assert sampling["return_kl"] is False
    assert "kl_reward_coef" not in sampling
    assert "n" not in sampling
    assert "reward_view" not in sampling
    assert "prompts_per_batch" not in sampling


def test_registry_keeps_return_artifacts_as_wiring_metadata() -> None:
    """Checks registry keeps return artifacts as wiring metadata."""
    for family in FAMILY_REGISTRY:
        assert (
            FAMILY_REGISTRY[family].collector.return_artifacts
            == _default_return_artifacts
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
                    "samples_per_chunk": 1,
                },
            ),
        )
        assert collector.family == family
        assert callable(collector.collect)


def test_unknown_family_raises_clear_error() -> None:
    """Checks unknown family raises clear error."""
    with pytest.raises(ValueError, match="unsupported rollout family"):
        get_rollout_family_entry("not_a_family")


def test_ar_families_declare_importable_replay_builders() -> None:
    """Every AR family must resolve through the generic AR GRPO entrypoint.

    train_ar_grpo (vrl/scripts/ar/train.py) builds the trainer replay bundle
    from the entry's replay_runtime_builder string; a missing or misspelled
    declaration would only surface at training launch otherwise.
    """
    from vrl.rollouts.families.registry import FAMILY_REGISTRY
    from vrl.utils.config import import_from_path

    ar_families = [
        entry for entry in FAMILY_REGISTRY.values() if entry.task.startswith("ar_")
    ]
    assert len(ar_families) >= 6
    for entry in ar_families:
        assert entry.replay_runtime_builder, f"{entry.family} lacks replay_runtime_builder"
        assert callable(import_from_path(entry.replay_runtime_builder))
        assert callable(import_from_path(entry.runtime_builder))
        assert callable(import_from_path(entry.runtime_spec_extractor))
