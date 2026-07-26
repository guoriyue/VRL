from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

import vrl.config.builders as builders
import vrl.config.validation as validation
from vrl.config.builders import BuiltConfigs, RewardRuntimeConfig, build_configs
from vrl.config.loading import load_config
from vrl.config.reward_inference import RewardInferenceConfig
from vrl.config.schema import RootConfig


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


def test_build_resolves_public_config_precision_and_reward_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    calls = {"parse": 0, "precision": 0, "reward": 0, "standalone_reward_validation": 0}

    def counted(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        validation,
        "parse_config",
        counted("parse", validation.parse_config),
    )
    monkeypatch.setattr(
        validation,
        "resolve_precision_policy",
        counted("precision", validation.resolve_precision_policy),
    )
    monkeypatch.setattr(
        builders,
        "build_reward_config",
        counted("reward", builders.build_reward_config),
    )

    def unexpected_reward_reparse(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls["standalone_reward_validation"] += 1
        raise AssertionError("typed RootConfig.reward must not be parsed again")

    monkeypatch.setattr(
        validation,
        "validate_reward_config",
        unexpected_reward_reparse,
    )

    built = build_configs(cfg)

    assert isinstance(built, BuiltConfigs)
    assert isinstance(built.root, RootConfig)
    assert isinstance(built.reward, RewardRuntimeConfig)
    assert calls == {
        "parse": 1,
        "precision": 1,
        "reward": 1,
        "standalone_reward_validation": 0,
    }


def test_online_build_has_named_reward_and_trainer_fields() -> None:
    built = build_configs(load_config("experiment/sana/online_grpo_aesthetic"))

    assert built.trainer is not None
    assert built.reward is not None
    assert built.reward.weights["aesthetic"] == pytest.approx(1.0)
    assert built.reward.weights["pickscore"] == pytest.approx(0.0)
    assert set(built.reward.kwargs) == {"aesthetic", "pickscore"}


def test_reward_runtime_config_normalizes_every_component_and_derives_transport() -> None:
    reward = builders.build_reward_config(
        OmegaConf.create(
            {
                "reward": {
                    "components": {"remote": 0.0, "local": 1.0},
                    "kwargs": {
                        "remote": {
                            "inference": {
                                "kind": "http",
                                "endpoint": "http://127.0.0.1:8300",
                                "expected_model": "remote-model",
                            },
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
    external_only = builders.build_reward_config(
        OmegaConf.create(
            {
                "reward": {
                    "components": {"remote": 0.0},
                    "kwargs": {"remote": reward.kwargs["remote"]},
                },
            },
        ),
    )
    assert external_only.all_external_inference is True


def test_reward_runtime_config_rejects_inconsistent_component_maps() -> None:
    with pytest.raises(ValueError, match=r"kwargs keys must match component keys"):
        RewardRuntimeConfig(
            weights={"aesthetic": 1.0},
            kwargs={},
            inference_configs={"aesthetic": RewardInferenceConfig()},
        )

    with pytest.raises(ValueError, match=r"inference_configs keys must match component keys"):
        RewardRuntimeConfig(
            weights={"aesthetic": 1.0},
            kwargs={"aesthetic": {}},
            inference_configs={},
        )


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
        builders.build_reward_config(cfg)


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


@pytest.mark.parametrize("find_unused_parameters", [False, True])
def test_typed_root_is_a_strategy_builder_input(find_unused_parameters: bool) -> None:
    from vrl.trainers.distributed import DistributedTrainingContext
    from vrl.trainers.strategy import DDPStrategy, build_strategy

    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.training.strategy=ddp",
            (
                "distributed.training.ddp.find_unused_parameters="
                f"{str(find_unused_parameters).lower()}"
            ),
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
    assert strategy._find_unused_parameters is find_unused_parameters
