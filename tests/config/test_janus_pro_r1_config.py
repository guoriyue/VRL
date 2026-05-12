from __future__ import annotations

from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig
from vrl.config.loader import build_configs, load_config
from vrl.rollouts.runtime.launch_inputs import build_rollout_runtime_inputs


def test_janus_pro_r1_experiment_loads_with_multisegment_algorithm() -> None:
    cfg = load_config("experiment/janus_pro_1b_r1_ocr_grpo")
    built = build_configs(cfg)

    assert cfg.trainer.entrypoint == (
        "vrl.scripts.janus_pro.train:train_janus_pro_r1_ocr_grpo"
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
        "experiment/janus_pro_1b_r1_ocr_grpo",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.gpus_per_worker=0",
            "distributed.resources.rollout.num_workers=1",
        ],
    )
    inputs = build_rollout_runtime_inputs(
        cfg,
        "janus_pro_r1",
        weight_dtype=str(cfg.model.dtype),
    )

    assert inputs.runtime_spec.family == "janus_pro_r1"
    assert inputs.runtime_spec.task == "ar_t2i_r1"
    assert inputs.runtime_spec.executor_cls == (
        "vrl.models.families.janus_pro.r1_executor:JanusProR1PipelineExecutor"
    )
