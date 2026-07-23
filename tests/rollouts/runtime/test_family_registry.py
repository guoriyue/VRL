"""Tests for canonical model-family registry behavior."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.config.sampling_schema import SamplingSection
from vrl.config.schema import parse_config
from vrl.families.names import (
    _FAMILY_BY_ALIAS,
    normalize_model_family,
)
from vrl.families.registry import (
    FAMILY_REGISTRY,
    SHARED_MODEL_SECTION_CLS,
    DenoiseFamilyBuild,
    GenerationRuntimeCapabilities,
    TokenFamilyBuild,
    get_model_family_entry,
)
from vrl.families.semantics import GenerationRegime, PolicySemantics
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import (
    RolloutCollectorConfig,
    build_rollout_config_from_cfg,
)
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trajectory import TrajectoryStoragePolicy
from vrl.utils.config import cfg_get, import_from_path


def _typed_model_build_inputs(payload):
    cfg = OmegaConf.create(payload)
    root = parse_config(cfg)
    return cfg, root, resolve_precision_policy(root)


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
                    "chunk_passthrough_keys": [],
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
    assert build.model_config == {
        "revision": None,
        "use_lora": False,
        "lora": {
            "path": None,
            "target_modules": [],
            "dropout": 0.0,
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
    }
    assert build.sampling_config == {
        "guidance_scale": 0,
        "num_steps": 0,
        "max_sequence_length": None,
    }


def test_model_build_rejects_raw_config_and_wrong_precision_types() -> None:
    cfg, root, precision = _typed_model_build_inputs(
        {
            "model": {"family": "sana", "path": "unit-checkpoint"},
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
            },
        },
    )
    entry = get_model_family_entry("sana")

    with pytest.raises(TypeError, match="validated RootConfig"):
        entry.resolve_model_build(cfg, "cpu", precision=precision)
    with pytest.raises(TypeError, match="resolved PrecisionPolicy"):
        entry.resolve_model_build(root, "cpu", precision=object())


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


def test_model_build_cannot_bypass_unknown_model_keys_with_raw_config() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "sana",
                "path": "unit-checkpoint",
                "use_lorra": True,
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
            },
        },
    )

    with pytest.raises(ValueError, match=r"unknown model\.use_lorra"):
        parse_config(cfg)
    with pytest.raises(TypeError, match="validated RootConfig"):
        get_model_family_entry("sana").resolve_model_build(
            cfg,
            "cpu",
            precision=resolve_precision_policy(cfg),
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


def test_executor_config_support_is_not_duplicated_as_a_capability_field() -> None:
    assert (
        "supports_model_executor_config" not in GenerationRuntimeCapabilities.__dataclass_fields__
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
        assert entry.model_section_cls
        assert entry.sampling_section_cls
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


def test_sampling_sections_share_only_matching_request_vocabularies() -> None:
    predict2 = import_from_path(
        get_model_family_entry("cosmos-predict2").sampling_section_cls,
    )
    cosmos3 = import_from_path(get_model_family_entry("cosmos3").sampling_section_cls)
    sana = import_from_path(get_model_family_entry("sana").sampling_section_cls)
    hunyuan_image = import_from_path(
        get_model_family_entry("hunyuan_image").sampling_section_cls,
    )
    janus = import_from_path(get_model_family_entry("janus_pro").sampling_section_cls)
    glm = import_from_path(get_model_family_entry("glm_image").sampling_section_cls)

    assert predict2 is cosmos3
    assert sana is not hunyuan_image
    assert janus is not glm


def test_migrated_model_sections_are_owned_by_their_family_packages() -> None:
    expected_paths = {
        "causvid": "vrl.models.families.causvid.config:CausVidModelSection",
        "wan_2_1": "vrl.models.families.wan_2_1.config:WanModelSection",
        "wan_2_1_i2v": "vrl.models.families.wan_2_1.config:WanModelSection",
        "cosmos-predict2.5": (
            "vrl.models.families.cosmos.predict2_5.config:CosmosPredict25ModelSection"
        ),
        "cosmos-predict2-anima": (
            "vrl.models.families.cosmos.anima.config:CosmosAnimaModelSection"
        ),
        "echo": "vrl.models.families.echo.config:EchoModelSection",
        "flux": "vrl.models.families.flux.config:FluxModelSection",
        "janus_pro": "vrl.models.families.janus_pro.config:JanusProModelSection",
        "janus_pro_r1": "vrl.models.families.janus_pro.config:JanusProModelSection",
        "llamagen": "vrl.models.families.llamagen.config:LlamaGenModelSection",
        "magi_1": "vrl.models.families.magi_1.config:Magi1ModelSection",
        "nextstep_1": "vrl.models.families.nextstep_1.config:NextStep1ModelSection",
    }

    assert {
        family: get_model_family_entry(family).model_section_cls for family in expected_paths
    } == expected_paths


def test_migrated_token_runtime_configs_are_owned_by_their_family_packages() -> None:
    expected_paths = {
        "emu3": "vrl.models.families.emu3.config:Emu3Config",
        "glm_image": "vrl.models.families.glm_image.config:GlmImageConfig",
    }

    for family, expected_path in expected_paths.items():
        entry = get_model_family_entry(family)
        assert isinstance(entry.family_build, TokenFamilyBuild)
        assert entry.family_build.config_cls == expected_path
        assert entry.model_section_cls == SHARED_MODEL_SECTION_CLS

    llamagen = get_model_family_entry("llamagen")
    assert isinstance(llamagen.family_build, TokenFamilyBuild)
    assert llamagen.family_build.config_cls == "vrl.models.families.llamagen.config:LlamaGenConfig"

    nextstep = get_model_family_entry("nextstep_1")
    assert isinstance(nextstep.family_build, TokenFamilyBuild)
    assert (
        nextstep.family_build.config_cls == "vrl.models.families.nextstep_1.config:NextStep1Config"
    )

    janus = get_model_family_entry("janus_pro")
    janus_r1 = get_model_family_entry("janus_pro_r1")
    assert isinstance(janus.family_build, TokenFamilyBuild)
    assert janus.family_build is janus_r1.family_build
    assert janus.family_build.config_cls == "vrl.models.families.janus_pro.config:JanusProConfig"


def test_migrated_family_packages_keep_their_public_facades() -> None:
    expected_exports = {
        "vrl.models.families.causvid": {
            "CausVidCausalRunner",
            "CausVidGeometry",
            "CausVidModel",
            "CausVidModelSection",
            "CausVidReplayModel",
            "CausVidRunResult",
            "CausVidSchedule",
        },
        "vrl.models.families.echo": {
            "EchoModel",
            "EchoModelSection",
            "EchoReplayModel",
            "EchoSamplingState",
        },
        "vrl.models.families.emu3": {
            "Emu3ChunkExecutor",
            "Emu3Config",
            "Emu3Model",
            "Emu3ReplayModel",
            "emu3_allowed_token_mask",
            "emu3_forced_token_schedule",
            "emu3_grid_token_num",
        },
        "vrl.models.families.flux": {"FluxModelSection"},
        "vrl.models.families.glm_image": {
            "GlmImageChunkExecutor",
            "GlmImageConfig",
            "GlmImageModel",
            "GlmImageReplayModel",
            "glm_image_decode_position_schedule",
            "glm_image_grid_dims",
            "glm_image_prefill_position_ids",
            "glm_image_token_num",
        },
        "vrl.models.families.janus_pro": {
            "JANUS_R1_SEGMENTS",
            "JanusProChunkExecutor",
            "JanusProConfig",
            "JanusProModel",
            "JanusProModelSection",
            "JanusProR1ChunkExecutor",
            "JanusProR1ChunkGatherer",
            "image_token_logits_from_hidden",
        },
        "vrl.models.families.llamagen": {
            "LLAMAGEN_CAPTION_TOKEN_NUM",
            "LLAMAGEN_IMAGE_TOKEN_NUM",
            "LLAMAGEN_IMAGE_VOCAB_SIZE",
            "LlamaGenChunkExecutor",
            "LlamaGenConfig",
            "LlamaGenModel",
            "LlamaGenModelSection",
            "LlamaGenReplayModel",
        },
        "vrl.models.families.magi_1": {
            "Magi1Model",
            "Magi1ModelSection",
            "Magi1SubprocessConfig",
            "Magi1SubprocessModel",
        },
        "vrl.models.families.nextstep_1": {
            "NextStep1ChunkExecutor",
            "NextStep1ChunkGatherer",
            "NextStep1Config",
            "NextStep1Model",
            "NextStep1ModelSection",
        },
    }

    for module_name, expected in expected_exports.items():
        module = importlib.import_module(module_name)
        assert expected == set(module.__all__)
        assert expected <= set(dir(module))


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
                "from vrl.families.registry import FAMILY_REGISTRY; "
                "from vrl.utils.config import import_from_path; "
                "[import_from_path(entry.model_section_cls) "
                "for entry in FAMILY_REGISTRY.values()]; "
                "[import_from_path(entry.sampling_section_cls) "
                "for entry in FAMILY_REGISTRY.values()]; "
                "from vrl.models.families.causvid import CausVidModelSection; "
                "from vrl.models.families.cosmos.anima "
                "import CosmosAnimaModelSection; "
                "from vrl.models.families.cosmos.predict2_5 "
                "import CosmosPredict25ModelSection; "
                "from vrl.models.families.echo import EchoModelSection; "
                "from vrl.models.families.flux import FluxModelSection; "
                "from vrl.models.families.janus_pro import JanusProModelSection; "
                "from vrl.models.families.llamagen import LlamaGenModelSection; "
                "from vrl.models.families.magi_1 import Magi1ModelSection; "
                "from vrl.models.families.nextstep_1 import NextStep1ModelSection; "
                "from vrl.models.families.wan_2_1 import WanModelSection; "
                "assert CausVidModelSection.__module__.endswith('.causvid.config'); "
                "assert WanModelSection.__module__.endswith('.wan_2_1.config'); "
                "assert CosmosPredict25ModelSection.__module__.endswith("
                "'.predict2_5.config'); "
                "assert CosmosAnimaModelSection.__module__.endswith('.anima.config'); "
                "assert EchoModelSection.__module__.endswith('.echo.config'); "
                "assert FluxModelSection.__module__.endswith('.flux.config'); "
                "assert JanusProModelSection.__module__.endswith('.janus_pro.config'); "
                "assert LlamaGenModelSection.__module__.endswith('.llamagen.config'); "
                "assert Magi1ModelSection.__module__.endswith('.magi_1.config'); "
                "assert NextStep1ModelSection.__module__.endswith("
                "'.nextstep_1.config'); "
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


def test_token_runtime_config_imports_do_not_load_model_runtimes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from vrl.families.registry import get_model_family_entry; "
                "from vrl.utils.config import import_from_path; "
                "emu_entry = get_model_family_entry('emu3'); "
                "glm_entry = get_model_family_entry('glm_image'); "
                "janus_entry = get_model_family_entry('janus_pro'); "
                "llamagen_entry = get_model_family_entry('llamagen'); "
                "nextstep_entry = get_model_family_entry('nextstep_1'); "
                "emu_cls = import_from_path(emu_entry.family_build.config_cls); "
                "glm_cls = import_from_path(glm_entry.family_build.config_cls); "
                "janus_cls = import_from_path(janus_entry.family_build.config_cls); "
                "llamagen_cls = import_from_path("
                "llamagen_entry.family_build.config_cls); "
                "nextstep_cls = import_from_path("
                "nextstep_entry.family_build.config_cls); "
                "from vrl.models.families.emu3 import Emu3Config; "
                "from vrl.models.families.glm_image import GlmImageConfig; "
                "from vrl.models.families.janus_pro "
                "import JANUS_R1_SEGMENTS, JanusProConfig; "
                "from vrl.models.families.llamagen import LlamaGenConfig; "
                "from vrl.models.families.nextstep_1 import NextStep1Config; "
                "assert emu_cls is Emu3Config; "
                "assert glm_cls is GlmImageConfig; "
                "assert janus_cls is JanusProConfig; "
                "assert llamagen_cls is LlamaGenConfig; "
                "assert nextstep_cls is NextStep1Config; "
                "assert Emu3Config.__module__.endswith('.emu3.config'); "
                "assert GlmImageConfig.__module__.endswith('.glm_image.config'); "
                "assert JanusProConfig.__module__.endswith('.janus_pro.config'); "
                "assert LlamaGenConfig.__module__.endswith('.llamagen.config'); "
                "assert NextStep1Config.__module__.endswith('.nextstep_1.config'); "
                "assert JANUS_R1_SEGMENTS == "
                "('initial_image', 'selfcheck_text', 'final_image'); "
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


def test_family_entry_rejects_an_empty_model_section_path() -> None:
    entry = get_model_family_entry("sana")

    with pytest.raises(ValueError, match="requires a model section class"):
        replace(entry, model_section_cls="")


def test_family_entry_rejects_an_empty_sampling_section_path() -> None:
    entry = get_model_family_entry("sana")

    with pytest.raises(ValueError, match="requires a sampling section class"):
        replace(entry, sampling_section_cls="")


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
        {"replay_cls": "example:Replay"},
        {"transformer_classname": "Transformer"},
        {
            "replay_cls": "example:Replay",
            "transformer_classname": "Transformer",
            "replay_runtime_builder": "example:build_replay",
        },
        {
            "replay_cls": "example:Replay",
            "transformer_classname": "Transformer",
            "replay_unavailable_reason": "generation only",
        },
        {
            "replay_runtime_builder": "example:build_replay",
            "replay_unavailable_reason": "generation only",
        },
        {"replay_unavailable_reason": "  "},
        {
            "replay_runtime_builder": "example:build_replay",
            "scheduler_classname": "Scheduler",
        },
        {
            "replay_unavailable_reason": "generation only",
            "scheduler_classname": "Scheduler",
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

    rollout = build_rollout_config_from_cfg(root)

    assert rollout.request_sampling == {
        "attention_backend": "torch_native",
        "image_token_num": 16,
    }


def test_raw_collector_adapter_derives_fields_from_present_family() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "glm_image"},
            "sampling": {
                "attention_backend": "torch_native",
                "image_height": 256,
            },
        },
    )

    rollout = build_rollout_config_from_cfg(cfg)

    assert rollout.request_sampling == {"image_height": 256}


def test_cosmos_predict2_recipe_keeps_request_and_reward_fps_at_16() -> None:
    cfg = load_config("experiment/cosmos_predict2/online_grpo_v2w_reference_480p")
    root = parse_config(cfg)
    collector_config = build_rollout_config_from_cfg(root)
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
