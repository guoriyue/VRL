"""Tests for rollout runtime input construction."""

from __future__ import annotations

import pickle

import pytest

from vrl.config.loading import load_config
from vrl.generation.ar.executor import ARDiscreteChunkGatherer
from vrl.generation.diffusion import DiffusionChunkGatherer
from vrl.generation.protocols import GenerationChunkExecutor
from vrl.generation.ray import RayGenerationLauncher, RayGenerationLaunchInputs
from vrl.models.ar.janus_pro.runtime import JanusProR1ChunkGatherer
from vrl.models.ar.nextstep_1.runtime import NextStep1ChunkGatherer
from vrl.rollouts.collector.config import build_rollout_config_from_cfg
from vrl.rollouts.families import get_rollout_family_entry


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
        ],
    )
    entry = get_rollout_family_entry(family)

    inputs = RayGenerationLauncher.build_inputs(
        cfg,
        entry,
        executor_kwargs={"samples_per_chunk": 2},
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert pickle.loads(pickle.dumps(inputs.launch_contract)) == inputs.launch_contract
    assert inputs.launch_contract.family == family
    assert inputs.launch_contract.model_build["family"] == inputs.launch_contract.family
    # registry is the single source of truth for the canonical task string
    assert inputs.launch_contract.task == entry.task
    assert inputs.launch_contract.policy_version == 0
    assert inputs.launch_contract.runtime_builder == entry.runtime_builder
    assert inputs.launch_contract.executor_cls == entry.executor_cls
    # Generic-executor families also carry their model.executor yaml block
    # (family/task/num_frames/...) in executor_kwargs; families with their own
    # executor carry only the cfg-derived kwargs. Both must thread the
    # cfg-derived samples_per_chunk.
    assert inputs.launch_contract.executor_kwargs["samples_per_chunk"] == 2
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

    inputs = RayGenerationLauncher.build_inputs(
        cfg,
        get_rollout_family_entry("sd3_5"),
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert inputs.launch_contract.model_build is not None
    assert inputs.launch_contract.model_build["device"] == "cuda"
    assert inputs.launch_contract.model_build["parameter_dtype"] == "bfloat16"
    assert inputs.launch_contract.model_build["rollout"]["autocast_dtype"] == "bfloat16"


def test_sana_launch_contract_carries_parameter_and_rollout_precision() -> None:
    """SANA's fp16 parameters and bf16 autocast survive the Ray boundary."""
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

    inputs = RayGenerationLauncher.build_inputs(
        cfg,
        get_rollout_family_entry("sana"),
    )

    model_build = inputs.launch_contract.model_build
    assert model_build is not None
    assert model_build["parameter_dtype"] == "float16"
    assert model_build["rollout"] == {
        "autocast_dtype": "bfloat16",
        "prompt_encoder_dtype": "bfloat16",
        "quantization_format": None,
        "quantization_recipe": None,
        "base_weight_sync": False,
    }
    assert pickle.loads(pickle.dumps(model_build)) == model_build


def test_sana_fp8_rollout_keeps_bf16_outer_autocast() -> None:
    """FP8 swaps GEMMs; unswapped rollout ops remain under bf16 autocast."""
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
    cfg.precision = {
        "training": {"dtype": "bf16"},
        "rollout": {
            "dtype": "bf16",
            "quantization": {"format": "fp8"},
            "prompt_encoders": {"dtype": "bf16"},
        },
    }

    inputs = RayGenerationLauncher.build_inputs(
        cfg,
        get_rollout_family_entry("sana"),
    )

    model_build = inputs.launch_contract.model_build
    assert model_build is not None
    assert model_build["parameter_dtype"] == "float16"
    assert model_build["rollout"]["autocast_dtype"] == "bfloat16"
    assert model_build["rollout"]["prompt_encoder_dtype"] == "bfloat16"
    assert model_build["rollout"]["quantization_format"] == "fp8"


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

    inputs = RayGenerationLauncher.build_inputs(
        cfg,
        get_rollout_family_entry("sd3_5"),
    )

    assert "samples_per_chunk" not in inputs.launch_contract.executor_kwargs
    assert (
        build_rollout_config_from_cfg(cfg, family="sd3_5").request_sampling()["samples_per_chunk"]
        == "auto"
    )


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
    entry = get_rollout_family_entry(family)

    inputs = RayGenerationLauncher.build_inputs(
        cfg,
        entry,
    )

    assert entry.collector.kind == "diffusion"
    assert inputs.launch_contract.model_build is not None
    model_config = inputs.launch_contract.model_build["model_config"]
    assert model_config["torch_compile"] == {
        "enable": True,
        "mode": "default",
    }


def test_explicit_executor_kwargs_override_registry_defaults() -> None:
    """Checks explicit executor kwargs override registry defaults."""
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

    inputs = RayGenerationLauncher.build_inputs(
        cfg,
        get_rollout_family_entry("sd3_5"),
        executor_kwargs={"samples_per_chunk": 3},
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    # Explicit executor_kwargs override the cfg-derived value (3, not the
    # rollout.samples_per_chunk=8 above); the generic executor's config keys
    # ride alongside.
    assert inputs.launch_contract.executor_kwargs["samples_per_chunk"] == 3
