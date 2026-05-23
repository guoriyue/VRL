from __future__ import annotations

from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig
from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.rollouts.families import build_ray_generation_inputs_for_family


def test_janus_pro_r1_experiment_loads_with_multisegment_algorithm() -> None:
    cfg = load_config("experiment/online/ocr/ar_multisegment_token_grpo")
    built = build_configs(cfg)

    assert cfg.trainer.entrypoint == (
        "vrl.scripts.ar.janus_pro.train:train_janus_pro_r1_ocr_grpo"
    )
    assert cfg.algorithm.kind == "token_grpo_multisegment"
    assert isinstance(built["algorithm"], MultiSegmentTokenGRPOConfig)
    assert built["algorithm"].train_segments == {
        "initial_image": True,
        "selfcheck_text": False,
        "final_image": True,
    }
    assert cfg.eval.fixed.enabled is True
    assert len(cfg.eval.fixed.prompts) == 8


def test_janus_pro_r1_runtime_inputs_select_r1_task_and_executor() -> None:
    cfg = load_config(
        "experiment/online/ocr/ar_multisegment_token_grpo",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
        ],
    )
    inputs = build_ray_generation_inputs_for_family(
        cfg,
        "janus_pro_r1",
        weight_dtype=str(cfg.model.dtype),
    )

    assert inputs.launch_contract.family == "janus_pro_r1"
    assert inputs.launch_contract.task == "ar_t2i_r1"
    assert inputs.launch_contract.executor_cls == (
        "vrl.models.ar.janus_pro.runtime:JanusProR1PipelineExecutor"
    )


def test_janus_pro_r1_codex_qa_experiment_loads() -> None:
    cfg = load_config("experiment/online/codex_qa/ar_multisegment_token_grpo")
    built = build_configs(cfg)

    assert cfg.reward.components.codex_image_qa == 1.0
    assert cfg.trainer.entrypoint == (
        "vrl.scripts.ar.janus_pro.train:train_janus_pro_r1_codex_qa_grpo"
    )
    assert cfg.reward.kwargs.codex_image_qa.command[0] == "codex"
    assert cfg.data.manifest == "datasets/drawbench/train_192.txt"
    assert isinstance(built["algorithm"], MultiSegmentTokenGRPOConfig)


def test_janus_pro_r1_codex_qa_profile_wrapper_enables_profiler() -> None:
    cfg = load_config("profile/one_epoch/r1_codex_qa")
    built = build_configs(cfg)

    assert cfg.trainer.entrypoint == (
        "vrl.scripts.ar.janus_pro.train:train_janus_pro_r1_codex_qa_grpo"
    )
    assert cfg.trainer.total_epochs == 1
    assert cfg.trainer.output_dir == "outputs/profile_janus_r1_codex_qa_1epoch"
    assert built["trainer"].torch_profiler.enabled is True
    assert cfg.rollout.torch_profiler.enabled is True


def test_janus_pro_r1_codex_qa_rollout_profiler_reaches_launch_contract() -> None:
    cfg = load_config(
        "profile/one_epoch/r1_codex_qa",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
        ],
    )
    inputs = build_ray_generation_inputs_for_family(
        cfg,
        "janus_pro_r1",
        weight_dtype=str(cfg.model.dtype),
    )

    assert inputs.launch_contract.extra["profiler_output_dir"] == (
        "outputs/profile_janus_r1_codex_qa_1epoch"
    )
    assert inputs.launch_contract.extra["torch_profiler"]["enabled"] is True
