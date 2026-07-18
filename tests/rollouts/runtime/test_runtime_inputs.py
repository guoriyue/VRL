"""Tests for rollout runtime input construction."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from vrl.config.loading import load_config
from vrl.families.registry import ModelFamilyEntry, get_model_family_entry
from vrl.generation.ar.executor import ARDiscreteChunkGatherer
from vrl.generation.diffusion import DiffusionChunkGatherer
from vrl.generation.protocols import GenerationChunkExecutor
from vrl.generation.ray import RayGenerationLauncher, RayGenerationLaunchInputs
from vrl.generation.ray.config import RayGenerationConfig
from vrl.models.ar.janus_pro.runtime import JanusProR1ChunkGatherer
from vrl.models.ar.nextstep_1.runtime import NextStep1ChunkGatherer
from vrl.ray.placement import RolePlacement
from vrl.ray.resources import resolve_distributed_resources
from vrl.rollouts.collector.config import build_rollout_config_from_cfg


def _capture_launch_inputs(
    cfg: Any,
    entry: ModelFamilyEntry,
) -> RayGenerationLaunchInputs:
    """Intercept the public launch boundary without starting Ray actors."""

    captured: list[RayGenerationLaunchInputs] = []

    def capture_launch(
        _launcher: RayGenerationLauncher,
        _config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement,
    ) -> RayGenerationLaunchInputs:
        assert isinstance(placement, RolePlacement)
        captured.append(launch_inputs)
        return launch_inputs

    with patch.object(RayGenerationLauncher, "launch", new=capture_launch):
        result = RayGenerationLauncher(init_ray=False).launch_from_cfg(
            cfg,
            resources=resolve_distributed_resources(cfg),
            entry=entry,
            driver_bundle=SimpleNamespace(
                model=SimpleNamespace(device="cpu"),
                trainable_modules={},
            ),
            placement=RolePlacement(
                placement_group=object(),
                bundle_indices=(),
                expected_gpu_ids=(),
            ),
        )

    assert captured == [result]
    return captured[0]


@pytest.mark.parametrize(
    ("experiment", "family", "expected_gatherer"),
    [
        ("diffusion/sd3_5/online_grpo_ocr", "sd3_5", DiffusionChunkGatherer),
        (
            "diffusion/sana/online_grpo_aesthetic",
            "sana",
            DiffusionChunkGatherer,
        ),
        ("diffusion/wan_2_1/online_grpo_ocr", "wan_2_1", DiffusionChunkGatherer),
        (
            "diffusion/wan_2_1/online_grpo_kling_video_reward",
            "wan_2_1",
            DiffusionChunkGatherer,
        ),
        (
            "diffusion/wan_2_1/online_grpo_physics_i2v",
            "wan_2_1_i2v",
            DiffusionChunkGatherer,
        ),
        (
            "diffusion/wan_2_1/online_grpo_i2v_smoke_single_gpu",
            "wan_2_1_i2v",
            DiffusionChunkGatherer,
        ),
        (
            "diffusion/cosmos_predict2/online_grpo_kling_video_reward",
            "cosmos-predict2",
            DiffusionChunkGatherer,
        ),
        (
            "diffusion/anima_preview3/online_grpo_aesthetic",
            "cosmos-predict2-anima",
            DiffusionChunkGatherer,
        ),
        (
            "diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety",
            "cosmos-predict2-anima",
            DiffusionChunkGatherer,
        ),
        (
            "ar/janus_pro/online_grpo_ocr",
            "janus_pro",
            ARDiscreteChunkGatherer,
        ),
        (
            "ar/janus_pro/online_r1_grpo_ocr",
            "janus_pro_r1",
            JanusProR1ChunkGatherer,
        ),
        (
            "ar/nextstep_1/online_grpo_ocr",
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

    inputs = _capture_launch_inputs(cfg, entry)

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert pickle.loads(pickle.dumps(inputs.launch_contract)) == inputs.launch_contract
    assert inputs.launch_contract.family == family
    # Family identity lives once in the outer contract; worker-side executor
    # wiring comes from the registry, while this nested payload is per-run data.
    assert "family" not in inputs.launch_contract.model_build
    assert inputs.launch_contract.policy_version == 0
    if entry.collector_kind == "diffusion":
        assert inputs.launch_contract.executor_kwargs["samples_per_chunk"] == 2
    else:
        assert "samples_per_chunk" not in inputs.launch_contract.executor_kwargs
    assert isinstance(inputs.gatherer, expected_gatherer)
    assert not isinstance(inputs.gatherer, GenerationChunkExecutor)


def test_diffusion_launch_contract_uses_resolved_config_parameter_dtype() -> None:
    """The worker payload derives ordinary parameter dtype from rollout precision."""
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.resources.visible_devices=[0,1]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=1",
            "distributed.resources.rollout.gpus_per_worker=1",
            "distributed.resources.rollout.num_workers=1",
        ],
    )

    inputs = _capture_launch_inputs(
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
        "experiment/diffusion/sana/online_grpo_aesthetic",
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

    inputs = _capture_launch_inputs(
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
        "experiment/diffusion/sana/online_grpo_aesthetic",
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

    inputs = _capture_launch_inputs(
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
        "experiment/diffusion/sd3_5/online_grpo_ocr",
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

    inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    assert "samples_per_chunk" not in inputs.launch_contract.executor_kwargs
    assert build_rollout_config_from_cfg(cfg).request_sampling()["samples_per_chunk"] == "auto"


@pytest.mark.parametrize(
    ("experiment", "family"),
    [
        ("diffusion/sd3_5/online_grpo_ocr", "sd3_5"),
        ("diffusion/wan_2_1/online_grpo_ocr", "wan_2_1"),
        ("diffusion/wan_2_1/online_grpo_physics_i2v", "wan_2_1_i2v"),
        ("diffusion/wan_2_1/online_grpo_i2v_smoke_single_gpu", "wan_2_1_i2v"),
        ("diffusion/cosmos_predict2/online_grpo_kling_video_reward", "cosmos-predict2"),
        (
            "diffusion/cosmos_predict2_5/online_nft_kling_video_reward",
            "cosmos-predict2.5",
        ),
        ("diffusion/anima_preview3/online_grpo_aesthetic", "cosmos-predict2-anima"),
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

    inputs = _capture_launch_inputs(
        cfg,
        entry,
    )

    assert entry.collector_kind == "diffusion"
    model_config = inputs.launch_contract.model_build["model_config"]
    assert model_config["torch_compile"] == {
        "enable": True,
        "mode": "default",
    }


def test_executor_kwargs_use_configured_chunk_size() -> None:
    """The public config is the only chunk-size input to the launch contract."""
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "rollout.samples_per_chunk=8",
        ],
    )

    inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert inputs.launch_contract.executor_kwargs["samples_per_chunk"] == 8
