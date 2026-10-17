"""Tests for canonical model-family registry behavior."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict

import pytest
from omegaconf import OmegaConf

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.models.families.names import (
    _FAMILY_BY_ALIAS,
    normalize_model_family,
)
from vrl.models.families.registry import (
    FAMILY_REGISTRY,
    DenoiseFamilyBuild,
    get_model_family_entry,
)
from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)
from vrl.models.interfaces.runtime import ModelBuild
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trajectory.storage import TrajectoryStoragePolicy


def _typed_model_build_inputs(payload):
    cfg = OmegaConf.create(payload)
    root = parse_config(cfg)
    return cfg, root, PrecisionPolicy.from_section(root.precision)


@pytest.mark.parametrize("family", ["cosmos-predict2.5", "flux", "sd3_5"])
def test_rollout_and_replay_builds_share_one_checkpoint_identity(family) -> None:
    _, root, precision = _typed_model_build_inputs(
        {
            "model": {
                "family": family,
                "path": "example/model",
                "revision": "a" * 40,
                "use_lora": True,
                "lora": {"rank": 2, "alpha": 4, "target_modules": ["to_q"]},
            },
            "rollout": {"sde": {"type": "flow_grpo"}},
            "precision": {"float32_precision": "ieee", "training": {"dtype": "fp32"}},
        },
    )

    identities = []
    for for_rollout in (False, True):
        build = get_model_family_entry(family).resolve_model_build(
            root,
            "cpu",
            precision=precision,
            for_rollout=for_rollout,
        )
        # Ray reconstructs the same build from the dataclass payload.
        restored = ModelBuild(**asdict(build))
        identities.append(restored.checkpoint_identity())
    assert identities[0] == identities[1]


def test_model_build_projects_typed_sections_without_losing_falsy_presence() -> None:
    _, root, precision = _typed_model_build_inputs(
        {
            "model": {
                "family": "sana",
                "path": "unit-checkpoint",
                "revision": None,
                "use_lora": False,
                "lora": {
                    "path": None,
                    "target_modules": [],
                    "dropout": 0,
                },
                "memory": {
                    "vae_decode": {
                        "tiling": False,
                        "slicing": False,
                    },
                },
                "torch_compile": {
                    "enable": False,
                    "mode": "",
                },
                "executor": {
                    "batch_passthrough_keys": [],
                },
            },
            "sampling": {
                "guidance_scale": 0,
                "num_steps": 1,
                "max_sequence_length": None,
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
            },
        },
    )

    build = get_model_family_entry("sana").resolve_model_build(
        root,
        "cpu",
        precision=precision,
    )

    assert build.model_name_or_path == "unit-checkpoint"
    assert build.revision is None
    assert build.model_config == {
        "use_lora": False,
        "lora": {
            "path": None,
            "target_modules": [],
            "dropout": 0.0,
        },
        "torch_compile": {
            "enable": False,
            "mode": "",
        },
    }
    assert build.generation_memory == GenerationMemoryPolicy(
        vae_decode=VaeDecodeMemory(tiling=False, slicing=False),
    )
    assert build.sampling_config == {
        "guidance_scale": 0,
        "num_steps": 1,
        "max_sequence_length": None,
    }

    replay_build = get_model_family_entry("sana").resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=False,
    )
    assert replay_build.generation_memory is None
    assert "memory" not in (replay_build.model_config or {})


def test_model_build_rejects_entry_and_typed_family_mismatch() -> None:
    _, root, precision = _typed_model_build_inputs(
        {
            "model": {"family": "sana", "path": "unit-checkpoint"},
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
            },
        },
    )

    with pytest.raises(ValueError, match="does not match registry entry"):
        get_model_family_entry("sd3_5").resolve_model_build(
            root,
            "cpu",
            precision=precision,
        )


@pytest.mark.parametrize("path", [None, ""])
def test_denoise_model_build_requires_a_nonempty_path(path) -> None:
    _, root, precision = _typed_model_build_inputs(
        {
            "model": {"family": "sana", "path": path},
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=r"^config missing required field: model\.path$",
    ):
        get_model_family_entry("sana").resolve_model_build(
            root,
            "cpu",
            precision=precision,
        )


def test_model_section_imports_do_not_load_model_runtimes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from vrl.config.model_schema import "
                "LoraSection, ModelExecutorSection, ModelMemorySection, "
                "TorchCompileSection, VaeDecodeMemorySection; "
                "from vrl.models.families.registry import FAMILY_REGISTRY; "
                "from vrl.utils.config import import_from_path; "
                "from vrl.config.model_schema import ModelSection; "
                "from vrl.config.sampling_schema import SamplingSection; "
                "sections = [import_from_path(entry.model_section_cls) "
                "for entry in FAMILY_REGISTRY.values()]; "
                "assert all(issubclass(section, ModelSection) for section in sections); "
                "assert all(issubclass(import_from_path(entry.sampling_section_cls), "
                "SamplingSection) for entry in FAMILY_REGISTRY.values()); "
                # Registry-derived, not a second hand-written family list: a new
                # family is covered the moment it registers a section path.
                "assert all(section.__module__ == 'vrl.config.model_schema' "
                "or section.__module__.startswith('vrl.models.families.') "
                "and section.__module__.endswith('.config') "
                "for section in sections); "
                "assert all(section.__module__ == 'vrl.config.model_schema' "
                "for section in (LoraSection, ModelExecutorSection, "
                "ModelMemorySection, TorchCompileSection, "
                "VaeDecodeMemorySection)); "
                "assert 'torch' not in sys.modules; "
                "assert 'diffusers' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'peft' not in sys.modules; "
                "assert 'safetensors' not in sys.modules; "
                "assert 'huggingface_hub' not in sys.modules; "
                "assert not any("
                "name.startswith('vrl.models.families.') and "
                "any(part in {'model', 'runtime', 'runner', 'adapter'} "
                "for part in name.split('.')) "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_denoise_family_build_rejects_scheduler_without_transformer() -> None:
    with pytest.raises(ValueError, match="scheduler_classname requires transformer_classname"):
        DenoiseFamilyBuild(
            model_cls="example:Model", replay_cls="example:Replay", scheduler_classname="Scheduler"
        )


def test_family_aliases_resolve_to_canonical_entries() -> None:
    assert _FAMILY_BY_ALIAS
    for alias, family in _FAMILY_BY_ALIAS.items():
        assert normalize_model_family(alias) == family
        assert get_model_family_entry(alias) is FAMILY_REGISTRY[family]


def test_rollout_config_is_projected_from_yaml() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "wan_2_1"},
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
                "n_samples_per_prompt": 4,
                "prompts_per_batch": 1,
                "samples_per_generation_batch": 8,
                "noise_level": 1.0,
                "sde": {
                    "type": "flow_grpo",
                    "window_size": 0,
                    "window_range": [0, 10],
                },
                "trajectory_storage": {"device": "cpu", "dtype": "float16"},
            },
            "algorithm": {"kind": "grpo"},
        },
    )

    rollout = RolloutCollectorConfig.from_root(parse_config(cfg))

    assert rollout.request_sampling["width"] == 1280
    assert rollout.request_sampling["num_steps"] == 35
    assert rollout.samples_per_generation_batch == 8
    assert "samples_per_generation_batch" not in rollout.request_sampling
    assert rollout.denoise == DenoiseRequestOptions(
        noise_level=1.0,
        sde_type="flow_grpo",
        sde_window_size=0,
        sde_window_range=(0, 10),
    )
    # Only generation-owned values reach the request; driver and storage
    # settings stay on the collector config.
    for driver_key in ("noise_level", "n_samples_per_prompt", "prompts_per_batch"):
        assert driver_key not in rollout.request_sampling
    assert "trajectory_storage" not in rollout.request_sampling
    assert rollout.trajectory_storage == TrajectoryStoragePolicy(
        device="cpu",
        dtype="float16",
    )


def test_request_fps_override_updates_reward_metadata() -> None:
    collector_request = GenerationRequestBuilder(
        entry=get_model_family_entry("cosmos-predict2"),
        config=RolloutCollectorConfig(request_sampling={"fps": 16}),
    ).build(
        ["robot arm moves"],
        1,
        request_overrides={"fps": 12},
    )

    assert collector_request.request.sampling["fps"] == 12
    assert collector_request.metadata["video_fps"] == 12
