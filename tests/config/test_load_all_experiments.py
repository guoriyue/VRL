"""Core config-loader safety tests.

This file intentionally keeps broad experiment coverage in loop-style tests
instead of parametrizing the same assertion into dozens of collected tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
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


def test_cosmos_diffusion_nft_video_reward_validation_config() -> None:
    cfg = load_config("experiment/online/ocr/video_diffusion_nft")

    assert cfg.reward.kwargs.video_reward.inference_runtime == "ray"
    assert cfg.reward.kwargs.video_reward.artifact_dir == (
        f"{cfg.trainer.output_dir}/reward_artifacts"
    )
    assert cfg.reward.kwargs.video_reward.debug_dir == (
        f"{cfg.trainer.output_dir}/reward_debug"
    )
    assert cfg.distributed.resources.reward.num_gpus == 1
    assert cfg.distributed.resources.reward.share_with_rollout is True
    assert cfg.distributed.reward.release_after_score is True
    assert cfg.trainer.total_epochs == 1


def test_cosmos_optimization_check_records_trainable_change(tmp_path: Path) -> None:
    from vrl.scripts.cosmos.train import (
        _capture_optimization_check_before,
        _capture_optimization_check_metrics,
        _write_optimization_check_after,
    )

    module = torch.nn.Linear(1, 1, bias=False)
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"video_reward": 1.0},
                "kwargs": {
                    "video_reward": {
                        "inference_runtime": "ray",
                        "reward_name": "cosmos_reason1",
                        "artifact_dir": str(tmp_path / "reward_artifacts"),
                        "debug_dir": str(tmp_path / "reward_debug"),
                        "worker_config": {"reward_model_version": "reward-v1"},
                    },
                },
            },
        },
    )
    stack = SimpleNamespace(
        cfg=cfg,
        bundle=SimpleNamespace(trainable_modules={"module": module}),
        output_dir=tmp_path,
        trainer=SimpleNamespace(state=SimpleNamespace(global_step=1, step=1)),
    )
    state: dict[str, object] = {}

    _capture_optimization_check_before(state, stack, 0)
    with torch.no_grad():
        module.weight.add_(1.0)
    _capture_optimization_check_metrics(
        state,
        {"grad_norm": 2.0, "reward_mean": 1.0, "reward_std": 0.5, "advantage_mean": 0.25},
        SimpleNamespace(grad_norm=2.0, reward_mean=1.0, reward_std=0.5, advantage_mean=0.25),
    )
    _write_optimization_check_after(state, stack, 0)

    payload = json.loads((tmp_path / "optimization_check.json").read_text())
    assert payload["global_step"] == 1
    assert payload["grad_norm"] == pytest.approx(2.0)
    assert payload["reward_std"] == pytest.approx(0.5)
    assert payload["trainable_sha256_changed"] is True
    assert payload["reward_runtime"] == "ray"
    assert payload["reward_artifacts_manifest"].endswith("reward_artifacts/manifest.jsonl")


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
    del cfg.reward.kwargs.video_reward["score_key"]

    with pytest.raises(ValueError, match="video_reward"):
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
