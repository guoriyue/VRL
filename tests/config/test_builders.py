from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

import vrl.config.builders as builders
from vrl.config.builders import build_configs
from vrl.config.loading import load_config


def test_build_rejects_strict_resume_with_a_warm_start_adapter() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    OmegaConf.update(cfg, "trainer.resume_from", "/tmp/checkpoint-4")
    OmegaConf.update(cfg, "trainer.resume_strict", True)
    OmegaConf.update(cfg, "model.lora.path", "/tmp/warm-start-adapter")

    with pytest.raises(
        ValueError,
        match=r"trainer\.resume_from cannot be combined with model\.lora\.path",
    ):
        build_configs(cfg)


def test_build_clears_nonstrict_resume_adapter_in_raw_and_typed_sources() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    OmegaConf.update(cfg, "trainer.resume_from", "/tmp/checkpoint-4")
    OmegaConf.update(cfg, "trainer.resume_strict", False)
    OmegaConf.update(cfg, "model.lora.path", "/tmp/warm-start-adapter")

    built = build_configs(cfg)

    assert built.resume.checkpoint_path == "/tmp/checkpoint-4"
    assert built.resume.strict is False
    assert cfg.model.lora.path == ""
    assert built.root.model is not None
    assert built.root.model.lora is not None
    assert built.root.model.lora.path == ""


def test_build_preserves_warm_start_adapter_without_full_resume() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    OmegaConf.update(cfg, "trainer.resume_from", "")
    OmegaConf.update(cfg, "model.lora.path", "/tmp/warm-start-adapter")

    built = build_configs(cfg)

    assert built.resume.checkpoint_path is None
    assert cfg.model.lora.path == "/tmp/warm-start-adapter"
    assert built.root.model is not None
    assert built.root.model.lora is not None
    assert built.root.model.lora.path == "/tmp/warm-start-adapter"


def test_online_build_has_named_reward_and_trainer_fields() -> None:
    built = build_configs(load_config("experiment/sana/online_grpo_aesthetic"))

    assert built.trainer is not None
    assert built.reward is not None
    assert built.reward.weights["aesthetic"] == pytest.approx(1.0)
    assert built.reward.weights["pickscore"] == pytest.approx(0.0)
    assert set(built.reward.kwargs) == {"aesthetic", "pickscore"}


def test_reward_runtime_config_normalizes_every_component_and_derives_transport() -> None:
    reward = builders.RewardRuntimeConfig.from_cfg(
        OmegaConf.create(
            {
                "reward": {
                    "components": {"remote": 0.0, "local": 1.0},
                    "inference": {
                        "remote": {
                            "kind": "http",
                            "endpoint": "http://127.0.0.1:8300",
                            "expected_model": "remote-model",
                        },
                    },
                },
            },
        ),
    )

    assert reward.weights == {"remote": 0.0, "local": 1.0}
    assert reward.kwargs["local"] == {}
    assert set(reward.kwargs) == {"remote", "local"}
    assert set(reward.inference_configs) == {"remote", "local"}
    assert reward.all_external_inference is False
    external_only = builders.RewardRuntimeConfig.from_cfg(
        OmegaConf.create(
            {
                "reward": {
                    "components": {"remote": 0.0},
                    "inference": {
                        "remote": {
                            "kind": "http",
                            "endpoint": "http://127.0.0.1:8300",
                            "expected_model": "remote-model",
                        },
                    },
                },
            },
        ),
    )
    assert external_only.all_external_inference is True


def test_reward_builder_rejects_kwargs_without_a_component() -> None:
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"aesthetic": 1.0},
                "kwargs": {"aestheic": {}},
            },
        },
    )

    with pytest.raises(ValueError, match=r"reward\.kwargs\.aestheic"):
        builders.RewardRuntimeConfig.from_cfg(cfg)


def test_offline_dpo_uses_the_same_build_result_without_online_state() -> None:
    built = build_configs(load_config("experiment/wan_2_1/offline_dpo_pickapic"))

    assert built.root.algorithm is not None
    assert built.root.algorithm.kind == "diffusion_dpo"
    assert type(built.algorithm).__name__ == "DiffusionDPOConfig"
    assert built.trainer is None
    assert built.reward is None
    assert built.resume.checkpoint_path is None
    assert built.resume.strict is True


def test_online_build_rejects_missing_or_all_zero_reward() -> None:
    missing = load_config("experiment/sd3_5/online_grpo_ocr")
    del missing["reward"]
    with pytest.raises(ValueError, match="online recipe requires a reward section"):
        build_configs(missing)

    all_zero = load_config("experiment/sd3_5/online_grpo_ocr")
    for name in all_zero.reward.components:
        all_zero.reward.components[name] = 0.0
    with pytest.raises(ValueError, match="At least one reward component"):
        build_configs(all_zero)


@pytest.mark.parametrize(
    ("rollout_dtype", "stages_match", "correction_mode"),
    [
        ("bf16", True, "off"),
        ("fp16", False, "truncate"),
    ],
)
def test_precision_role_split_is_resolved_once_into_trainer(
    rollout_dtype: str,
    stages_match: bool,
    correction_mode: str,
) -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[f"precision.rollout.dtype={rollout_dtype}"],
    )

    built = build_configs(cfg)

    assert built.trainer is not None
    assert built.precision.stages_match is stages_match
    assert built.trainer.train_precision == built.precision.training.label
    assert built.trainer.rollout_precision == built.precision.rollout.label
    assert built.trainer.precision_correction.tis_mode == correction_mode


def test_typed_root_is_a_strategy_builder_input() -> None:
    from vrl.trainers.distributed import DistributedTrainingContext
    from vrl.trainers.strategy import DDPStrategy, build_strategy

    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.training.strategy=ddp",
            "distributed.training.ddp.find_unused_parameters=true",
        ],
    )
    built = build_configs(cfg)
    context = DistributedTrainingContext(
        strategy="ddp",
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )

    strategy = build_strategy(built.root, context)

    assert isinstance(strategy, DDPStrategy)
    assert strategy._find_unused_parameters is True
