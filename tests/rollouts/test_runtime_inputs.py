"""Tests for rollout runtime input construction."""

from __future__ import annotations

import pickle

import pytest
import torch

from vrl.config.loading import load_config
from vrl.generation.diffusion import DiffusionChunkGatherer
from vrl.generation.protocols import PipelineExecutor
from vrl.models.ar.janus_pro.runtime import JanusProChunkGatherer
from vrl.models.ar.nextstep_1.runtime import NextStep1ChunkGatherer
from vrl.rollouts.families import (
    RayGenerationLaunchInputs,
    build_ray_generation_inputs_for_family,
    get_rollout_family_entry,
)


@pytest.mark.parametrize(
    ("experiment", "family", "expected_task", "expected_gatherer"),
    [
        ("diffusion/sd3_5/online_grpo_ocr", "sd3_5", "t2i", DiffusionChunkGatherer),
        ("diffusion/wan_2_1/online_grpo_ocr", "wan_2_1", "t2v", DiffusionChunkGatherer),
        (
            "diffusion/wan_2_1/online_grpo_video_reward",
            "wan_2_1",
            "t2v",
            DiffusionChunkGatherer,
        ),
        (
            "diffusion/cosmos_predict2/online_grpo_video_reward",
            "cosmos-predict2",
            "v2w",
            DiffusionChunkGatherer,
        ),
        (
            "diffusion/anima_preview3/online_grpo_aesthetic",
            "cosmos-predict2-anima",
            "t2i",
            DiffusionChunkGatherer,
        ),
        (
            "diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety",
            "cosmos-predict2-anima",
            "t2i",
            DiffusionChunkGatherer,
        ),
        (
            "ar/janus_pro/online_grpo_ocr",
            "janus_pro",
            "ar_t2i",
            JanusProChunkGatherer,
        ),
        (
            "ar/nextstep_1/online_grpo_ocr",
            "nextstep_1",
            "ar_t2i",
            NextStep1ChunkGatherer,
        ),
    ],
)
def test_rollout_runtime_inputs_are_serializable_and_registry_backed(
    experiment: str,
    family: str,
    expected_task: str,
    expected_gatherer: type,
) -> None:
    cfg = load_config(
        f"experiment/{experiment}",
        overrides=[
            "distributed.backend=ray",
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

    inputs = build_ray_generation_inputs_for_family(
        cfg,
        family,
        weight_dtype=torch.bfloat16,
        executor_kwargs={"sample_batch_size": 2},
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert pickle.loads(pickle.dumps(inputs.launch_contract)) == inputs.launch_contract
    assert inputs.launch_contract.family == family
    assert inputs.launch_contract.task == expected_task
    assert inputs.launch_contract.policy_version == 0
    assert inputs.launch_contract.runtime_builder == entry.runtime_builder
    assert inputs.launch_contract.executor_cls == entry.executor_cls
    assert inputs.launch_contract.executor_kwargs == {"sample_batch_size": 2}
    assert isinstance(inputs.gatherer, expected_gatherer)
    assert not isinstance(inputs.gatherer, PipelineExecutor)


def test_diffusion_launch_contract_uses_worker_primitive_device_and_dtype() -> None:
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.backend=ray",
            "distributed.resources.visible_devices=[0,1]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=1",
            "distributed.resources.rollout.gpus_per_worker=1",
            "distributed.resources.rollout.num_workers=1",
        ],
    )

    inputs = build_ray_generation_inputs_for_family(
        cfg,
        "sd3_5",
        weight_dtype=torch.float16,
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert inputs.launch_contract.model_build is not None
    assert inputs.launch_contract.model_build["device"] == "cuda"
    assert inputs.launch_contract.model_build["dtype"] == "float16"


def test_cosmos_runtime_inputs_include_reference_image_from_cfg() -> None:
    cfg = load_config(
        "experiment/diffusion/cosmos_predict2/online_grpo_video_reward",
        overrides=[
            "distributed.backend=ray",
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "distributed.resources.reward.num_gpus=0",
            "distributed.resources.reward.gpus_per_worker=0",
            "model.reference_image=/tmp/reference.png",
        ],
    )

    inputs = build_ray_generation_inputs_for_family(
        cfg,
        "cosmos-predict2",
        weight_dtype=torch.bfloat16,
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert inputs.launch_contract.family == "cosmos-predict2"
    assert inputs.launch_contract.executor_kwargs["reference_image"] == "/tmp/reference.png"
    assert inputs.launch_contract.model_build is not None
    assert inputs.launch_contract.model_build["extra"]["reference_image"] == "/tmp/reference.png"


def test_explicit_executor_kwargs_override_registry_defaults() -> None:
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.backend=ray",
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
            "rollout.sample_batch_size=8",
        ],
    )

    inputs = build_ray_generation_inputs_for_family(
        cfg,
        "sd3_5",
        weight_dtype=torch.bfloat16,
        executor_kwargs={"sample_batch_size": 3},
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert inputs.launch_contract.executor_kwargs == {"sample_batch_size": 3}
