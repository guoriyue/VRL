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
    task_configs = list((CONFIGS_ROOT / "task").glob("*.yaml"))

    assert flattened == []
    assert task_configs == []
    assert not (CONFIGS_ROOT / "profiling").exists()


def test_experiments_are_grouped_by_model_family() -> None:
    expected = {
        "ar/janus_pro/online_grpo_ocr",
        "ar/janus_pro/online_r1_grpo_codex_qa",
        "ar/janus_pro/online_r1_grpo_ocr",
        "ar/nextstep_1/online_grpo_ocr",
        "diffusion/anima_preview3/online_grpo_aesthetic",
        "diffusion/anima_preview3/online_grpo_anatomy",
        "diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety",
        "diffusion/cosmos_predict2/online_grpo_video_reward",
        "diffusion/cosmos_predict2_5/online_nft_video_reward",
        "diffusion/sd3_5/online_grpo_geneval",
        "diffusion/sd3_5/online_grpo_ocr",
        "diffusion/sd3_5/online_grpo_ocr_prompt_alignment",
        "diffusion/sd3_5/online_grpo_pickscore",
        "diffusion/wan_2_1/offline_dpo_pickapic",
        "diffusion/wan_2_1/online_grpo_ocr",
    }

    assert set(_experiment_names()) == expected
    assert {Path(name).parts[0] for name in _experiment_names()} == {"ar", "diffusion"}


def test_experiments_use_dataset_groups_and_only_override_reward_weights() -> None:
    inline_data = []
    inline_reward_kwargs = []
    for path in EXPERIMENT_DIR.rglob("*.yaml"):
        raw = OmegaConf.load(path)
        if "data" in raw:
            inline_data.append(path.relative_to(CONFIGS_ROOT).as_posix())
        reward = raw.get("reward", None)
        if reward is not None and "kwargs" in reward:
            inline_reward_kwargs.append(path.relative_to(CONFIGS_ROOT).as_posix())

    assert inline_data == []
    assert inline_reward_kwargs == []


def test_reward_configs_are_single_reward_building_blocks() -> None:
    offenders = []
    for path in (CONFIGS_ROOT / "reward").rglob("*.yaml"):
        raw = OmegaConf.load(path)
        components = raw.get("reward", {}).get("components", {})
        if len(components) != 1:
            offenders.append(path.relative_to(CONFIGS_ROOT).as_posix())

    assert offenders == []


def test_all_experiments_load_and_validate() -> None:
    for name in _experiment_names():
        cfg = load_config(f"experiment/{name}")
        assert "model" in cfg, f"{name} missing model.*"
        assert "trainer" in cfg, f"{name} missing trainer.*"
        assert "algorithm" in cfg, f"{name} missing algorithm.*"
        assert "data" in cfg, f"{name} missing data.* source"
        if str(cfg.algorithm.kind) != "diffusion_dpo":
            assert "reward" in cfg, f"{name} missing reward.* source"
        assert "path" in cfg.model, f"{name} missing model.path"
        assert "entrypoint" in cfg.trainer, f"{name} missing trainer.entrypoint"
        assert "output_dir" in cfg.trainer, f"{name} missing trainer.output_dir"
        assert "kind" in cfg.algorithm, f"{name} missing algorithm.kind"
        assert "adv_estimator" not in cfg.algorithm, f"{name} still uses adv_estimator"
        validate_training_config(cfg)


def test_algorithm_config_dispatches_representative_kinds() -> None:
    examples = {
        "diffusion/sd3_5/online_grpo_ocr": GRPOConfig,
        "ar/janus_pro/online_grpo_ocr": TokenGRPOConfig,
        "ar/janus_pro/online_r1_grpo_ocr": MultiSegmentTokenGRPOConfig,
        "diffusion/wan_2_1/offline_dpo_pickapic": DiffusionDPOConfig,
        "diffusion/cosmos_predict2_5/online_nft_video_reward": DiffusionNFTConfig,
    }
    for name, expected_type in examples.items():
        cfg = load_config(f"experiment/{name}")
        algo_cfg = build_algorithm_config(cfg)
        assert isinstance(algo_cfg, expected_type)
        assert isinstance(algo_cfg, EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)])
        if name == "ar/janus_pro/online_grpo_ocr":
            assert algo_cfg.kl_estimator == "k2"


def test_cosmos_diffusion_nft_video_reward_validation_config() -> None:
    cfg = load_config("experiment/diffusion/cosmos_predict2_5/online_nft_video_reward")

    assert cfg.reward.kwargs.video_reward.inference_runtime == "ray"
    assert cfg.reward.kwargs.video_reward.artifact_dir == (
        f"{cfg.trainer.output_dir}/reward_artifacts"
    )
    assert cfg.reward.kwargs.video_reward.debug_dir == (
        f"{cfg.trainer.output_dir}/reward_debug"
    )
    assert cfg.reward.kwargs.video_reward.reward_name == "cosmos_reason1"
    assert cfg.reward.kwargs.video_reward.worker_config.scorer == "import_path"
    assert "model_name" not in cfg.reward.kwargs.video_reward
    assert cfg.distributed.resources.reward.num_gpus == 1
    assert cfg.distributed.resources.reward.share_with_rollout is True
    assert cfg.distributed.reward.release_after_score is True
    assert cfg.trainer.total_epochs == 1


def test_anima_safe_reward_config_uses_cpu_nsfw_penalty() -> None:
    from vrl.scripts.common.factory import build_reward_from_cfg

    cfg = load_config("experiment/diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety")
    built = build_configs(cfg)

    reward_weights, reward_kwargs = built["reward"]
    assert reward_weights == {
        "aesthetic": pytest.approx(1.0),
        "nsfw_safety": pytest.approx(0.5),
    }
    assert reward_kwargs["nsfw_safety"]["classifier_device"] == "cpu"
    assert reward_kwargs["nsfw_safety"]["threshold"] == pytest.approx(0.35)
    assert cfg.data.manifest == "datasets/danbooru/safety/train.jsonl"
    assert cfg.data.eval_manifest == "datasets/danbooru/safety/eval_baseline.jsonl"
    assert cfg.rollout.sample_batch_size == 1
    reward_fn = build_reward_from_cfg(cfg, built=built, device="cpu")
    assert [name for name, _, _ in reward_fn.rewards] == ["aesthetic", "nsfw_safety"]


def test_anima_anatomy_reward_config_uses_anime_components() -> None:
    from vrl.scripts.common.factory import build_reward_from_cfg

    cfg = load_config("experiment/diffusion/anima_preview3/online_grpo_anatomy")
    built = build_configs(cfg)

    reward_weights, reward_kwargs = built["reward"]
    assert reward_weights == {
        "anime_anatomy_structure": pytest.approx(1.0),
    }
    assert cfg.data.manifest == "datasets/danbooru/anatomy/train_prompts.jsonl"
    assert cfg.data.eval_manifest == "datasets/danbooru/anatomy/eval_prompts.jsonl"
    assert "backend" not in reward_kwargs["anime_anatomy_structure"]
    assert "model_repo" not in reward_kwargs["anime_anatomy_structure"]
    reward_fn = build_reward_from_cfg(cfg, built=built, device="cpu")
    assert [name for name, _, _ in reward_fn.rewards] == ["anime_anatomy_structure"]


def test_anima_config_keeps_artifact_names_without_local_paths() -> None:
    cfg = load_config("experiment/diffusion/anima_preview3/online_grpo_aesthetic")
    model_yaml = OmegaConf.to_yaml(cfg.model)

    assert cfg.model.path == "circlestone-labs/Anima"
    assert cfg.model.transformer_file == (
        "split_files/diffusion_models/anima-preview3-base.safetensors"
    )
    assert cfg.model.text_encoder_file == "split_files/text_encoders/qwen_3_06b_base.safetensors"
    assert cfg.model.vae_file == "split_files/vae/qwen_image_vae.safetensors"
    assert cfg.model.transformer_path == ""
    assert cfg.model.text_encoder_path == ""
    assert cfg.model.vae_path == ""
    assert "tokenizer_root" not in cfg.model
    assert "anima-inference" not in model_yaml
    assert "ComfyUI" not in model_yaml
    assert cfg.model.qwen_tokenizer_path == "Qwen/Qwen2.5-0.5B"
    assert cfg.model.t5_tokenizer_path == "google-t5/t5-base"


def test_cosmos_optimization_check_records_trainable_change(tmp_path: Path) -> None:
    from vrl.scripts.diffusion.cosmos.train import (
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

    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    target = resolve_train_target(cfg)

    assert target.import_path == cfg.trainer.entrypoint
    assert callable(_import_callable(target.import_path))


def test_cli_overrides_reach_typed_trainer_config() -> None:
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
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
    cfg = load_config("experiment/diffusion/cosmos_predict2/online_grpo_video_reward")
    del cfg.reward.kwargs.video_reward["score_key"]

    with pytest.raises(ValueError, match="video_reward"):
        validate_reward_config(cfg)


def test_negative_reward_component_weights_are_rejected() -> None:
    cfg = load_config("experiment/diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety")
    cfg.reward.components.nsfw_safety = -0.5

    with pytest.raises(ValueError, match=r"reward\.components\.nsfw_safety must be >= 0"):
        validate_reward_config(cfg)


def test_required_training_fields_fail_fast() -> None:
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_ocr")
    cfg.trainer.output_dir = "???"
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        validate_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    cfg = load_config("experiment/diffusion/wan_2_1/offline_dpo_pickapic")
    cfg.data.max_train_samples = None

    assert optional_none(cfg, "data.max_train_samples") is None
    validate_training_config(cfg)


def test_dpo_recipe_declares_trainer_only_resource_plan() -> None:
    from vrl.ray.resources import resolve_distributed_resources

    cfg = load_config(
        "experiment/diffusion/wan_2_1/offline_dpo_pickapic",
        overrides=["distributed.resources.visible_devices=[0]"],
    )
    resolved = resolve_distributed_resources(cfg)

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == ()
    assert resolved.rollout_num_workers == 0
