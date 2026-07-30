"""Tests for rollout runtime input construction."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.families.registry import (
    FAMILY_REGISTRY,
    GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR,
    ModelFamilyEntry,
    get_model_family_entry,
)
from vrl.generation.bindings.full_sequence_denoise import DiffusionChunkGatherer
from vrl.generation.bindings.token_autoregressive.executor import ARDiscreteChunkGatherer
from vrl.generation.protocols import GenerationChunkExecutor
from vrl.generation.ray import RayGenerationLauncher, RayGenerationLaunchInputs
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launcher import build_executor_kwargs
from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
from vrl.models.families.janus_pro.runtime import JanusProR1ChunkGatherer
from vrl.models.families.nextstep_1.runtime import NextStep1ChunkGatherer
from vrl.ray.placement import RolePlacement
from vrl.ray.resources import resolve_distributed_resources
from vrl.rollouts.collector.config import RolloutCollectorConfig


def _capture_launch_inputs(
    cfg: Any,
    entry: ModelFamilyEntry,
) -> tuple[RayGenerationLaunchInputs, dict[str, Any]]:
    """Intercept the public launch boundary without starting Ray actors."""

    captured: list[RayGenerationLaunchInputs] = []
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
    config = RayGenerationConfig.from_cfg(
        root,
        resources=resolve_distributed_resources(cfg),
    )
    runtime_device = "cuda" if config.resources.rollout_gpus_per_worker > 0 else "cpu"
    identity_build = entry.resolve_model_build(
        root,
        runtime_device,
        precision=precision,
    )
    expected_model_identity = resolve_checkpoint_model_identity(identity_build)

    def capture_launch(
        _launcher: RayGenerationLauncher,
        resolved_config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement,
    ) -> RayGenerationLaunchInputs:
        assert isinstance(placement, RolePlacement)
        assert resolved_config is config
        captured.append(launch_inputs)
        return launch_inputs

    with patch.object(RayGenerationLauncher, "launch", new=capture_launch):
        result = RayGenerationLauncher(init_ray=False).launch_from_cfg(
            root,
            precision=precision,
            config=config,
            entry=entry,
            driver_bundle=SimpleNamespace(
                model=SimpleNamespace(device="cpu"),
                trainable_modules={},
            ),
            expected_model_identity=expected_model_identity,
            placement=RolePlacement(
                placement_group=object(),
                bundle_indices=(),
                expected_gpu_ids=(),
            ),
        )

    assert captured == [result]
    return captured[0], expected_model_identity


@pytest.mark.parametrize(
    ("experiment", "family", "expected_gatherer"),
    [
        ("sd3_5/online_grpo_ocr", "sd3_5", DiffusionChunkGatherer),
        (
            "sana/online_grpo_aesthetic",
            "sana",
            DiffusionChunkGatherer,
        ),
        ("wan_2_1/online_grpo_ocr", "wan_2_1", DiffusionChunkGatherer),
        (
            "wan_2_1/online_grpo_kling_video_reward",
            "wan_2_1",
            DiffusionChunkGatherer,
        ),
        (
            "wan_2_1/online_grpo_physics_i2v",
            "wan_2_1_i2v",
            DiffusionChunkGatherer,
        ),
        (
            "wan_2_1/online_grpo_i2v_smoke_single_gpu",
            "wan_2_1_i2v",
            DiffusionChunkGatherer,
        ),
        (
            "cosmos_predict2/online_grpo_kling_video_reward",
            "cosmos-predict2",
            DiffusionChunkGatherer,
        ),
        (
            "anima_preview3/online_grpo_aesthetic",
            "cosmos-predict2-anima",
            DiffusionChunkGatherer,
        ),
        (
            "anima_preview3/online_grpo_aesthetic_nsfw_safety",
            "cosmos-predict2-anima",
            DiffusionChunkGatherer,
        ),
        (
            "janus_pro/online_grpo_ocr",
            "janus_pro",
            ARDiscreteChunkGatherer,
        ),
        (
            "janus_pro/online_r1_grpo_ocr",
            "janus_pro_r1",
            JanusProR1ChunkGatherer,
        ),
        (
            "nextstep_1/online_grpo_ocr",
            "nextstep_1",
            NextStep1ChunkGatherer,
        ),
    ],
)
def test_rollout_runtime_inputs_are_serializable_and_registry_backed(
    experiment: str,
    family: str,
    expected_gatherer: type,
) -> None:
    """Checks rollout runtime inputs are serializable and registry-backed."""
    cfg = load_config(
        f"experiment/{experiment}",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "distributed.resources.reward.num_gpus=0",
            "distributed.resources.reward.gpus_per_worker=0",
            "distributed.rollout.cpus_per_worker=1",
            "rollout.samples_per_chunk=2",
        ],
    )
    entry = get_model_family_entry(family)

    inputs, expected_model_identity = _capture_launch_inputs(cfg, entry)

    assert isinstance(inputs, RayGenerationLaunchInputs)
    restored = pickle.loads(pickle.dumps(inputs))
    assert isinstance(restored, RayGenerationLaunchInputs)
    assert restored.launch_contract == inputs.launch_contract
    assert restored.launch_contract.family == family
    assert restored.launch_contract.expected_model_identity == expected_model_identity
    # Family identity lives once in the outer contract; worker-side executor
    # wiring comes from the registry, while this nested payload is per-run data.
    assert "family" not in restored.launch_contract.model_build
    assert restored.launch_contract.policy_version == 0
    if entry.runtime_capabilities.accepts_samples_per_chunk:
        assert restored.launch_contract.executor_kwargs["samples_per_chunk"] == 2
    else:
        assert "samples_per_chunk" not in restored.launch_contract.executor_kwargs
    assert isinstance(restored.gatherer, expected_gatherer)
    assert not isinstance(restored.gatherer, GenerationChunkExecutor)


def test_diffusion_launch_contract_uses_resolved_config_parameter_dtype() -> None:
    """The worker payload derives ordinary parameter dtype from rollout precision."""
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.resources.visible_devices=[0,1]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=1",
            "distributed.resources.rollout.gpus_per_worker=1",
            "distributed.resources.rollout.num_workers=1",
        ],
    )

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert inputs.launch_contract.model_build["device"] == "cuda"
    assert inputs.launch_contract.model_build["parameter_dtype"] == "bfloat16"
    assert inputs.launch_contract.model_build["precision"] == {
        "dtype": "bf16",
        "float32_precision": "tf32",
        "quantization": None,
        "outer_autocast": True,
    }


def test_sana_launch_contract_carries_parameter_and_rollout_precision() -> None:
    """SANA's native FP16 policy and separate BF16 Gemma survive the Ray boundary."""
    cfg = load_config(
        "experiment/sana/online_grpo_aesthetic",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "distributed.resources.reward.num_gpus=0",
            "distributed.resources.reward.gpus_per_worker=0",
        ],
    )

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sana"),
    )

    model_build = inputs.launch_contract.model_build
    assert model_build["parameter_dtype"] == "float16"
    assert model_build["precision"] == {
        "dtype": "fp16",
        "float32_precision": "ieee",
        "quantization": None,
        "outer_autocast": False,
    }
    assert model_build["rollout"] == {
        "prompt_encoder_dtype": "bfloat16",
        "base_weight_sync": False,
    }
    assert pickle.loads(pickle.dumps(model_build)) == model_build


def test_sana_fp8_rollout_keeps_native_policy_and_bf16_prompt_encoder() -> None:
    """FP8 swaps GEMMs without changing SANA or Gemma's base dtype policies."""
    cfg = load_config(
        "experiment/sana/online_grpo_aesthetic",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "distributed.resources.reward.num_gpus=0",
            "distributed.resources.reward.gpus_per_worker=0",
        ],
    )
    cfg.precision.rollout.quantization = {"format": "fp8"}

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sana"),
    )

    model_build = inputs.launch_contract.model_build
    assert model_build["parameter_dtype"] == "float16"
    assert model_build["precision"] == {
        "dtype": "fp16",
        "float32_precision": "ieee",
        "quantization": {"format": "fp8", "recipe": "rowwise"},
        "outer_autocast": False,
    }
    assert model_build["rollout"]["prompt_encoder_dtype"] == "bfloat16"
    assert "quantization" not in model_build["rollout"]


def test_generation_chunk_auto_reaches_ray_runtime_without_executor_coercion() -> None:
    """Ray owns generation auto; the fixed executor fallback must not parse it."""
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            # This test only exercises input routing; resource validation has
            # dedicated coverage and the repository verify lane hides all GPUs.
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "distributed.resources.reward.num_gpus=0",
            "distributed.resources.reward.gpus_per_worker=0",
            "rollout.samples_per_chunk=auto",
        ],
    )

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    assert "samples_per_chunk" not in inputs.launch_contract.executor_kwargs
    assert RolloutCollectorConfig.from_cfg(cfg).request_sampling["samples_per_chunk"] == "auto"


@pytest.mark.parametrize(
    ("experiment", "family"),
    [
        ("sd3_5/online_grpo_ocr", "sd3_5"),
        ("wan_2_1/online_grpo_ocr", "wan_2_1"),
        ("wan_2_1/online_grpo_physics_i2v", "wan_2_1_i2v"),
        ("wan_2_1/online_grpo_i2v_smoke_single_gpu", "wan_2_1_i2v"),
        ("cosmos_predict2/online_grpo_kling_video_reward", "cosmos-predict2"),
        (
            "cosmos_predict2_5/online_nft_kling_video_reward",
            "cosmos-predict2.5",
        ),
        ("anima_preview3/online_grpo_aesthetic", "cosmos-predict2-anima"),
    ],
)
def test_model_torch_compile_applies_to_all_diffusion_rollout_families(
    experiment: str,
    family: str,
) -> None:
    """Checks model.torch_compile is the single compile source for rollout workers."""
    cfg = load_config(
        f"experiment/{experiment}",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "distributed.resources.reward.num_gpus=0",
            "distributed.resources.reward.gpus_per_worker=0",
            "model.torch_compile.enable=true",
            "model.torch_compile.mode=default",
        ],
    )
    entry = get_model_family_entry(family)

    inputs, _ = _capture_launch_inputs(
        cfg,
        entry,
    )

    assert entry.policy_semantics.step_kind == "denoise"
    assert entry.policy_semantics.generation_regime == "full_sequence"
    model_config = inputs.launch_contract.model_build["model_config"]
    assert model_config["torch_compile"] == {
        "enable": True,
        "mode": "default",
    }


def test_executor_kwargs_use_configured_chunk_size() -> None:
    """The public config is the only chunk-size input to the launch contract."""
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "rollout.samples_per_chunk=8",
        ],
    )

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert inputs.launch_contract.executor_kwargs["samples_per_chunk"] == 8


def test_generic_executor_kwargs_project_the_complete_model_block() -> None:
    cfg = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "flux",
                    "executor": {
                        "num_frames": 17,
                        "max_sequence_length": 256,
                        "fps": 24,
                        "chunk_passthrough_keys": ["text_ids"],
                    },
                    "memory": {"vae_decode": {"tiling": False}},
                },
                "rollout": {"samples_per_chunk": 3},
            },
        ),
    )

    assert build_executor_kwargs(get_model_family_entry("flux"), cfg) == {
        "samples_per_chunk": 3,
        "num_frames": 17,
        "max_sequence_length": 256,
        "fps": 24,
        "chunk_passthrough_keys": ["text_ids"],
    }


@pytest.mark.parametrize(
    "family",
    [
        family
        for family, entry in FAMILY_REGISTRY.items()
        if entry.executor_cls != GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR
    ],
)
def test_custom_executors_reject_model_executor_instead_of_silently_dropping_it(
    family: str,
) -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "executor": {"max_sequence_length": 123},
            },
        },
    )

    with pytest.raises(ValueError, match=r"does not support model\.executor"):
        build_executor_kwargs(get_model_family_entry(family), cfg)


@pytest.mark.parametrize(
    "family",
    [
        family
        for family, entry in FAMILY_REGISTRY.items()
        if not entry.runtime_capabilities.supported_model_memory_sections
    ],
)
def test_executor_projection_defensively_rejects_unsupported_memory(
    family: str,
) -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "memory": {"vae_decode": {"tiling": False}},
            },
        },
    )

    with pytest.raises(ValueError, match=r"does not support model\.memory"):
        build_executor_kwargs(get_model_family_entry(family), cfg)


def test_custom_executor_keeps_independent_supported_memory_config() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "memory": {"vae_decode": {"tiling": True}},
            },
            "rollout": {"samples_per_chunk": 2},
        },
    )

    assert build_executor_kwargs(get_model_family_entry("wan_2_1_i2v"), cfg) == {
        "samples_per_chunk": 2,
    }
