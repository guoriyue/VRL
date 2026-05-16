"""Core config-loader safety tests.

This file intentionally keeps broad experiment coverage in loop-style tests
instead of parametrizing the same assertion into dozens of collected tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.algorithms.diffusion_nft import DiffusionNFTConfig
from vrl.algorithms.dpo import DiffusionDPOConfig
from vrl.algorithms.grpo.continuous import GRPOConfig
from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig
from vrl.algorithms.grpo.token import TokenGRPOConfig
from vrl.config.builders import build_algorithm_config, build_configs
from vrl.config.loading import load_config
from vrl.config.validation import (
    optional_none,
    validate_reward_config,
    validate_training_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"
CONFIGS_ROOT = REPO_ROOT / "configs"

EXPECTED_ALGO_TYPE = {
    "grpo": GRPOConfig,
    "token_grpo": TokenGRPOConfig,
    "token_grpo_multisegment": MultiSegmentTokenGRPOConfig,
    "diffusion_dpo": DiffusionDPOConfig,
    "diffusion_nft": DiffusionNFTConfig,
}


def _experiment_names() -> list[str]:
    return sorted(
        p.relative_to(EXPERIMENT_DIR).with_suffix("").as_posix()
        for p in EXPERIMENT_DIR.rglob("*.yaml")
    )


def test_config_groups_are_not_flattened() -> None:
    flattened = [
        path
        for group in ("experiment", "model", "sampling")
        for path in (CONFIGS_ROOT / group).glob("*.yaml")
    ]

    assert flattened == []
    assert not (CONFIGS_ROOT / "profiling").exists()


def test_only_model_configs_are_split_by_model_name() -> None:
    model_name_tokens = ("sd3", "wan", "janus", "nextstep", "cosmos", "predict2")
    offenders = [
        path.relative_to(CONFIGS_ROOT).as_posix()
        for group in ("experiment", "sampling", "profile", "reward", "dataset")
        for path in (CONFIGS_ROOT / group).rglob("*.yaml")
        if any(
            token in path.relative_to(CONFIGS_ROOT / group).as_posix()
            for token in model_name_tokens
        )
    ]

    assert offenders == []


def test_experiments_compose_reward_and_dataset_groups() -> None:
    inline_fields = []
    for path in EXPERIMENT_DIR.rglob("*.yaml"):
        raw = OmegaConf.load(path)
        for key in ("reward", "data"):
            if key in raw:
                inline_fields.append(f"{path.relative_to(CONFIGS_ROOT).as_posix()}:{key}")

    assert inline_fields == []


def test_all_experiments_load_and_validate() -> None:
    for name in _experiment_names():
        cfg = load_config(f"experiment/{name}")
        assert "model" in cfg, f"{name} missing model.*"
        assert "trainer" in cfg, f"{name} missing trainer.*"
        assert "algorithm" in cfg, f"{name} missing algorithm.*"
        assert "data" in cfg, f"{name} missing data.* source"
        if name.startswith("online/"):
            assert "reward" in cfg, f"{name} missing reward.* source"
        assert "path" in cfg.model, f"{name} missing model.path"
        assert "entrypoint" in cfg.trainer, f"{name} missing trainer.entrypoint"
        assert "output_dir" in cfg.trainer, f"{name} missing trainer.output_dir"
        assert "kind" in cfg.algorithm, f"{name} missing algorithm.kind"
        assert "adv_estimator" not in cfg.algorithm, f"{name} still uses adv_estimator"
        validate_training_config(cfg)


def test_algorithm_config_dispatches_representative_kinds() -> None:
    examples = {
        "online/ocr/image_flow_grpo": GRPOConfig,
        "online/ocr/ar_discrete_token_grpo": TokenGRPOConfig,
        "online/ocr/ar_multisegment_token_grpo": MultiSegmentTokenGRPOConfig,
        "offline/dpo/diffusion": DiffusionDPOConfig,
        "online/ocr/video_diffusion_nft": DiffusionNFTConfig,
    }
    for name, expected_type in examples.items():
        cfg = load_config(f"experiment/{name}")
        algo_cfg = build_algorithm_config(cfg)
        assert isinstance(algo_cfg, expected_type)
        assert isinstance(algo_cfg, EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)])
        if name == "online/ocr/ar_discrete_token_grpo":
            assert algo_cfg.kl_estimator == "k2"


def test_unified_train_entrypoint_reads_yaml_entrypoint() -> None:
    from vrl.scripts.train import _import_callable, resolve_train_target

    cfg = load_config("experiment/online/ocr/image_flow_grpo")
    target = resolve_train_target(cfg)

    assert target.import_path == cfg.trainer.entrypoint
    assert callable(_import_callable(target.import_path))


def test_cli_overrides_reach_typed_trainer_config() -> None:
    cfg = load_config(
        "experiment/online/ocr/image_flow_grpo",
        overrides=[
            "trainer.resume_from=/tmp/checkpoint-10",
            "trainer.torch_profiler.enabled=true",
            "trainer.torch_profiler.activities=[cpu]",
        ],
    )
    trainer = build_configs(cfg)["trainer"]

    assert trainer.resume_from == "/tmp/checkpoint-10"
    assert trainer.torch_profiler.enabled is True
    assert trainer.torch_profiler.activities == ("cpu",)


def test_invalid_algorithm_kind_fails_fast() -> None:
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo", "adv_estimator": "dpo"}})
    with pytest.raises(ValueError, match="adv_estimator"):
        build_algorithm_config(cfg)

    cfg = OmegaConf.create({"algorithm": {"kind": "qpo"}})
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
        build_algorithm_config(cfg)


def test_reward_backbone_kwargs_are_required() -> None:
    cfg = load_config("experiment/online/aesthetic/video_diffusion_grpo")
    del cfg.reward.kwargs.aesthetic["model_name"]

    with pytest.raises(ValueError, match="aesthetic"):
        validate_reward_config(cfg)


def test_required_training_fields_fail_fast() -> None:
    cfg = load_config("experiment/online/ocr/video_diffusion_grpo")
    cfg.trainer.output_dir = "???"
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        validate_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    cfg = load_config("experiment/offline/dpo/diffusion")
    cfg.data.max_train_samples = None

    assert optional_none(cfg, "data.max_train_samples") is None
    validate_training_config(cfg)
