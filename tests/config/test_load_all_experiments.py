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
from vrl.config.loader import (
    build_algorithm_config,
    build_configs,
    load_config,
    optional_none,
    validate_reward_config,
    validate_training_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"

EXPECTED_ALGO_TYPE = {
    "grpo": GRPOConfig,
    "token_grpo": TokenGRPOConfig,
    "token_grpo_multisegment": MultiSegmentTokenGRPOConfig,
    "diffusion_dpo": DiffusionDPOConfig,
    "diffusion_nft": DiffusionNFTConfig,
}


def _experiment_names() -> list[str]:
    return sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))


def test_all_experiments_load_and_validate() -> None:
    for name in _experiment_names():
        cfg = load_config(f"experiment/{name}")
        assert "model" in cfg, f"{name} missing model.*"
        assert "trainer" in cfg, f"{name} missing trainer.*"
        assert "algorithm" in cfg, f"{name} missing algorithm.*"
        assert "data" in cfg, f"{name} missing data.* source"
        assert "path" in cfg.model, f"{name} missing model.path"
        assert "entrypoint" in cfg.trainer, f"{name} missing trainer.entrypoint"
        assert "output_dir" in cfg.trainer, f"{name} missing trainer.output_dir"
        assert "kind" in cfg.algorithm, f"{name} missing algorithm.kind"
        assert "adv_estimator" not in cfg.algorithm, f"{name} still uses adv_estimator"
        validate_training_config(cfg)


def test_algorithm_config_dispatches_representative_kinds() -> None:
    examples = {
        "sd3_5_ocr_grpo": GRPOConfig,
        "janus_pro_1b_ocr_grpo": TokenGRPOConfig,
        "janus_pro_1b_r1_ocr_grpo": MultiSegmentTokenGRPOConfig,
        "wan_2_1_1_3b_dpo": DiffusionDPOConfig,
        "cosmos_predict2_5_2b_diffusionnft": DiffusionNFTConfig,
    }
    for name, expected_type in examples.items():
        cfg = load_config(f"experiment/{name}")
        algo_cfg = build_algorithm_config(cfg)
        assert isinstance(algo_cfg, expected_type)
        assert isinstance(algo_cfg, EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)])


def test_unified_train_entrypoint_reads_yaml_entrypoint() -> None:
    from vrl.scripts.train import _import_callable, resolve_train_target

    cfg = load_config("experiment/sd3_5_ocr_grpo")
    target = resolve_train_target(cfg)

    assert target.import_path == cfg.trainer.entrypoint
    assert callable(_import_callable(target.import_path))


def test_cli_overrides_reach_typed_trainer_config() -> None:
    cfg = load_config(
        "experiment/sd3_5_ocr_grpo",
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
    cfg = load_config("experiment/cosmos_predict2_2b_grpo")
    del cfg.reward.kwargs.aesthetic["model_name"]

    with pytest.raises(ValueError, match="aesthetic"):
        validate_reward_config(cfg)


def test_required_training_fields_fail_fast() -> None:
    cfg = load_config("experiment/wan_2_1_1_3b_ocr_grpo")
    del cfg.trainer["output_dir"]
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        validate_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    cfg = load_config("experiment/wan_2_1_1_3b_dpo")
    cfg.data.max_train_samples = None

    assert optional_none(cfg, "data.max_train_samples") is None
    validate_training_config(cfg)
