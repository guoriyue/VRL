"""Tests for canonical model-family registry behavior."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.sampling_schema import SamplingSection
from vrl.config.schema import parse_config
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.models.families.names import (
    _FAMILY_BY_ALIAS,
    normalize_model_family,
)
from vrl.models.families.registry import (
    FAMILY_REGISTRY,
    DenoiseFamilyBuild,
    GenerationRuntimeCapabilities,
    TokenFamilyBuild,
    get_model_family_entry,
)
from vrl.models.families.semantics import PolicySemantics
from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trajectory import TrajectoryStoragePolicy
from vrl.utils.config import import_from_path


def _typed_model_build_inputs(payload):
    cfg = OmegaConf.create(payload)
    root = parse_config(cfg)
    return cfg, root, PrecisionPolicy.from_section(root.precision)


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
                "num_steps": 0,
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
        "num_steps": 0,
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


@pytest.mark.parametrize("model_payload", [{"family": "emu3"}, {"family": "emu3", "path": None}])
def test_token_model_build_derives_only_an_omitted_or_null_default_path(
    model_payload,
) -> None:
    _, root, precision = _typed_model_build_inputs(
        {
            "model": model_payload,
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
            },
        },
    )

    build = get_model_family_entry("emu3").resolve_model_build(
        root,
        "cpu",
        precision=precision,
    )

    assert build.model_name_or_path == "BAAI/Emu3-Gen-hf"


def test_token_model_build_rejects_an_explicit_empty_path() -> None:
    _, root, precision = _typed_model_build_inputs(
        {
            "model": {"family": "emu3", "path": ""},
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
        get_model_family_entry("emu3").resolve_model_build(
            root,
            "cpu",
            precision=precision,
        )


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


def test_generation_runtime_capabilities_reject_unknown_memory_sections() -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown model.memory section\(s\): transformer_offload",
    ):
        GenerationRuntimeCapabilities(
            supported_model_memory_sections=frozenset({"transformer_offload"}),
        )


def test_family_registry_entries_have_complete_protocol_wiring() -> None:
    assert FAMILY_REGISTRY
    assert len(FAMILY_REGISTRY) == len(set(FAMILY_REGISTRY))

    for family, entry in FAMILY_REGISTRY.items():
        assert entry.family == family
        assert entry.task
        assert entry.model_section_cls
        assert entry.sampling_section_cls
        assert callable(entry.new_gatherer().gather_batches)
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


def test_family_registry_entries_own_importable_model_sections() -> None:
    from vrl.config.model_schema import ModelSection

    for entry in FAMILY_REGISTRY.values():
        section_cls = import_from_path(entry.model_section_cls)
        assert isinstance(section_cls, type)
        assert issubclass(section_cls, ModelSection)


def test_family_registry_entries_own_importable_sampling_sections() -> None:
    for entry in FAMILY_REGISTRY.values():
        section_cls = import_from_path(entry.sampling_section_cls)
        assert isinstance(section_cls, type)
        assert issubclass(section_cls, SamplingSection)


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
                "sections = [import_from_path(entry.model_section_cls) "
                "for entry in FAMILY_REGISTRY.values()]; "
                "[import_from_path(entry.sampling_section_cls) "
                "for entry in FAMILY_REGISTRY.values()]; "
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


def test_policy_replay_support_is_derived_from_family_recipes() -> None:
    causvid = get_model_family_entry("causvid")
    magi = get_model_family_entry("magi_1")
    sana = get_model_family_entry("sana")
    llamagen = get_model_family_entry("llamagen")

    assert causvid.supports_policy_replay is True
    assert magi.supports_policy_replay is False
    assert sana.supports_policy_replay is True
    assert llamagen.supports_policy_replay is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "replay_cls": "example:Replay",
            "transformer_classname": "Transformer",
            "replay_runtime_builder": "example:build_replay",
        },
        {
            "replay_runtime_builder": "example:build_replay",
            "replay_unavailable_reason": "generation only",
        },
    ],
)
def test_denoise_family_build_rejects_ambiguous_replay_modes(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        DenoiseFamilyBuild(model_cls="example:Model", **kwargs)


def test_generation_only_entry_fails_before_dynamic_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = get_model_family_entry("magi_1")

    def fail_import(path: str) -> None:
        pytest.fail(f"generation-only replay unexpectedly imported {path}")

    monkeypatch.setattr("vrl.utils.config.import_from_path", fail_import)

    with pytest.raises(RuntimeError, match="final-video inference only"):
        entry.build_replay(SimpleNamespace(family="magi_1"))


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
                "samples_per_generation_batch": 8,
                "noise_level": 1.0,
                "sde": {
                    "type": "flow_grpo",
                    "window_size": 0,
                    "window_range": [0, 10],
                },
                "trajectory_storage": {"device": "cpu", "dtype": "float16"},
            },
            "algorithm": {"kind": "grpo", "kl_reward_coef": 0.25},
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
        return_kl=True,
    )
    assert "noise_level" not in rollout.request_sampling
    assert rollout.kl_reward_coef == pytest.approx(0.25)
    assert rollout.trajectory_storage == TrajectoryStoragePolicy(
        device="cpu",
        dtype="float16",
    )


def test_typed_collector_projects_only_the_selected_sampling_schema() -> None:
    root = parse_config(
        OmegaConf.create(
            {
                "model": {"family": "janus_pro"},
                "sampling": {
                    "attention_backend": "torch_native",
                    "image_token_num": 16,
                },
            },
        ),
    )

    rollout = RolloutCollectorConfig.from_root(root)

    assert rollout.request_sampling == {
        "attention_backend": "torch_native",
        "image_token_num": 16,
    }


def test_request_sampling_is_the_family_section_vocabulary() -> None:
    """The request keys are exactly the family's sampling fields: a key another
    family owns is rejected at parse, not silently dropped by the projection."""
    with pytest.raises(ValueError, match=r"unknown sampling\.attention_backend"):
        parse_config(
            OmegaConf.create(
                {
                    "model": {"family": "glm_image"},
                    "sampling": {"attention_backend": "torch_native", "image_height": 256},
                },
            ),
        )

    rollout = RolloutCollectorConfig.from_root(
        parse_config(
            OmegaConf.create(
                {"model": {"family": "glm_image"}, "sampling": {"image_height": 256}}
            ),
        ),
    )

    assert rollout.request_sampling == {"image_height": 256}


def test_cosmos_predict2_recipe_keeps_request_and_reward_fps_at_16() -> None:
    cfg = load_config("experiment/cosmos_predict2/online_grpo_v2w_reference_480p")
    root = parse_config(cfg)
    collector_config = RolloutCollectorConfig.from_root(root)
    collector_request = GenerationRequestBuilder(
        entry=get_model_family_entry("cosmos-predict2"),
        config=collector_config,
    ).build(["robot arm moves"], 1)

    assert collector_config.request_sampling["fps"] == 16
    assert collector_request.request.sampling["fps"] == 16
    assert collector_request.metadata["video_fps"] == 16


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


def test_request_sampling_projects_only_generation_owned_rollout_values() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "wan_2_1"},
            "sampling": {"width": 1280, "num_frames": 93, "fps": 16},
            "rollout": {
                "n_samples_per_prompt": 4,
                "prompts_per_batch": 1,
                "samples_per_generation_batch": 8,
                "sde": {"type": "flow_grpo", "window_range": [0, 10]},
            },
            "algorithm": {"kind": "grpo", "kl_reward_coef": 0.0},
        },
    )

    rollout = RolloutCollectorConfig.from_root(parse_config(cfg))
    sampling = rollout.request_sampling

    assert sampling["width"] == 1280
    assert rollout.samples_per_generation_batch == 8
    assert rollout.denoise is not None
    assert rollout.denoise.sde_type == "flow_grpo"
    assert rollout.denoise.sde_window_range == (0, 10)
    assert rollout.denoise.return_kl is False
    for driver_key in ("kl_reward_coef", "n_samples_per_prompt", "prompts_per_batch"):
        assert driver_key not in sampling
    assert "trajectory_storage" not in sampling
    assert rollout.trajectory_storage == TrajectoryStoragePolicy()


def test_all_registry_entries_build_collectors_from_the_same_entry() -> None:
    for entry in FAMILY_REGISTRY.values():
        collector = build_rollout_collector(
            entry,
            reward_runtime=RewardFunctionRuntime(None),
            config=RolloutCollectorConfig(samples_per_generation_batch=1),
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


def test_denoise_family_build_descriptors_have_one_explicit_replay_mode() -> None:
    denoise_entries = [
        entry
        for entry in FAMILY_REGISTRY.values()
        if isinstance(entry.family_build, DenoiseFamilyBuild)
    ]
    assert len(denoise_entries) >= 10
    for entry in denoise_entries:
        build = entry.family_build
        assert isinstance(build, DenoiseFamilyBuild)
        if not entry.supports_policy_replay:
            assert build.replay_cls is None
            assert build.transformer_classname is None
            assert build.replay_runtime_builder is None
            assert build.replay_unavailable_reason
            continue
        if build.replay_cls is not None:
            assert callable(import_from_path(build.replay_cls))
            assert build.transformer_classname
            assert build.replay_runtime_builder is None
            assert build.replay_unavailable_reason is None
        else:
            assert build.replay_runtime_builder
            assert callable(import_from_path(build.replay_runtime_builder))
            assert build.replay_unavailable_reason is None
