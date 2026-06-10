"""Core config-loader safety tests.

This file intentionally keeps broad experiment coverage in loop-style tests
instead of parametrizing the same assertion into dozens of collected tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    """Checks config groups are not flattened."""
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
    """Checks experiments are grouped by model family."""
    expected = {
        "ar/janus_pro/online_grpo_ocr",
        "ar/janus_pro/online_r1_grpo_ocr",
        "ar/nextstep_1/online_grpo_ocr",
        "diffusion/anima_preview3/online_grpo_aesthetic",
        "diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety",
        "diffusion/cosmos_predict2/online_grpo_v2w_reference",
        "diffusion/cosmos_predict2/online_grpo_kling_video_reward",
        "diffusion/cosmos_predict2_5/online_nft_kling_video_reward",
        "diffusion/cosmos_predict2_5/online_nft_motion_physics",
        "diffusion/sd3_5/online_grpo_geneval",
        "diffusion/sd3_5/online_grpo_ocr",
        "diffusion/sd3_5/online_grpo_ocr_crossnode_debug",
        "diffusion/sd3_5/online_grpo_pickscore",
        "diffusion/wan_2_1/offline_dpo_pickapic",
        "diffusion/wan_2_1/online_grpo_ocr",
        "diffusion/wan_2_1/online_grpo_physics",
        "diffusion/wan_2_1/online_grpo_physics_i2v",
        "diffusion/wan_2_1/online_grpo_kling_video_reward",
    }

    assert set(_experiment_names()) == expected
    assert {Path(name).parts[0] for name in _experiment_names()} == {"ar", "diffusion"}


def test_experiments_use_dataset_groups_and_only_override_reward_weights() -> None:
    """Checks experiments use dataset groups and only override reward weights."""
    inline_data = []
    inline_reward_kwargs = []
    allowed_reward_kwargs = {
        "experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference.yaml",
        "experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml",
        "experiment/diffusion/cosmos_predict2_5/online_nft_motion_physics.yaml",
        "experiment/diffusion/wan_2_1/online_grpo_physics.yaml",
        "experiment/diffusion/wan_2_1/online_grpo_physics_i2v.yaml",
        "experiment/diffusion/wan_2_1/online_grpo_kling_video_reward.yaml",
    }
    for path in EXPERIMENT_DIR.rglob("*.yaml"):
        raw = OmegaConf.load(path)
        if "data" in raw:
            inline_data.append(path.relative_to(CONFIGS_ROOT).as_posix())
        reward = raw.get("reward", None)
        if reward is not None and "kwargs" in reward:
            rel = path.relative_to(CONFIGS_ROOT).as_posix()
            if rel not in allowed_reward_kwargs:
                inline_reward_kwargs.append(rel)

    assert inline_data == []
    assert inline_reward_kwargs == []


def test_reward_configs_are_single_reward_building_blocks() -> None:
    """Checks reward configs are single reward building blocks."""
    offenders = []
    for path in (CONFIGS_ROOT / "reward").rglob("*.yaml"):
        raw = OmegaConf.load(path)
        components = raw.get("reward", {}).get("components", {})
        if len(components) != 1:
            offenders.append(path.relative_to(CONFIGS_ROOT).as_posix())

    assert offenders == []


def test_all_experiments_load_and_validate() -> None:
    """Checks all experiments load and validate."""
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


def test_experiments_do_not_use_legacy_precision_fields() -> None:
    """Checks live YAML uses top-level precision only."""
    offenders = []
    for name in _experiment_names():
        cfg = load_config(f"experiment/{name}")
        actor = cfg.get("actor", {})
        if "mixed_precision" in actor or "bf16" in actor:
            offenders.append(name)
    assert offenders == []


def test_online_diffusion_experiments_have_rollout_compile_strategy() -> None:
    """Checks online diffusion rollouts inherit the shared compile strategy block."""
    offenders = []
    for name in _experiment_names():
        cfg = load_config(f"experiment/{name}")
        if not name.startswith("diffusion/") or str(cfg.algorithm.kind) == "diffusion_dpo":
            continue
        compile_cfg = cfg.get("rollout", {}).get("denoise_compile")
        if compile_cfg is None:
            offenders.append(name)
            continue
        if dict(compile_cfg) != {
            "enable": False,
            "mode": "default",
        }:
            offenders.append(name)

    assert offenders == []


def test_model_memory_policy_defaults_are_yaml_backed() -> None:
    """Checks model memory policy defaults are YAML backed."""
    import torch

    from vrl.models.diffusion.common.vae_decode_memory import (
        VaeDecodeMemory,
        vae_decode_memory_from_config,
    )
    from vrl.models.diffusion.cosmos.anima.runtime import extract_anima_runtime_spec
    from vrl.models.diffusion.sd3_5.runtime import extract_sd3_5_runtime_spec
    from vrl.models.diffusion.wan_2_1.runtime import extract_wan_2_1_runtime_spec

    def vae_decode_policy(spec: Any) -> VaeDecodeMemory:
        memory = spec.memory or {}
        return vae_decode_memory_from_config(memory.get("vae_decode"))

    # SD3.5 carries only a frozen_offload memory block — no VAE decode knobs,
    # so the decode policy must fall back to the all-off default.
    sd3 = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    sd3_spec = extract_sd3_5_runtime_spec(sd3, "cpu", torch.float32)
    assert vae_decode_policy(sd3_spec) == VaeDecodeMemory()

    # Wan and Anima enable VAE tiling+slicing purely from their YAML blocks.
    wan = load_config("experiment/diffusion/wan_2_1/online_grpo_ocr")
    wan_spec = extract_wan_2_1_runtime_spec(wan, "cpu", torch.bfloat16)
    assert vae_decode_policy(wan_spec) == VaeDecodeMemory(tiling=True, slicing=True)

    anima = load_config("experiment/diffusion/anima_preview3/online_grpo_aesthetic")
    anima_spec = extract_anima_runtime_spec(anima, "cpu", torch.bfloat16)
    assert vae_decode_policy(anima_spec) == VaeDecodeMemory(tiling=True, slicing=True)


def test_rollout_orchestration_group_override_uses_rollout_namespace() -> None:
    """Checks rollout orchestration group override uses rollout namespace."""
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=["/base/rollout/orchestration=continuous"],
    )

    orchestration = cfg.trainer.rollout_orchestration
    assert orchestration.mode == "continuous"
    # weight_sync_barrier is no longer spelled in YAML; the typed config
    # derives the only barrier the mode supports.
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    typed = RolloutOrchestrationConfig(
        **OmegaConf.to_container(orchestration, resolve=True),
    )
    assert typed.weight_sync_barrier == "pause_admission_and_drain_inflight"


def test_algorithm_config_dispatches_representative_kinds() -> None:
    """Checks algorithm config dispatches representative kinds."""
    examples = {
        "diffusion/sd3_5/online_grpo_ocr": GRPOConfig,
        "ar/janus_pro/online_grpo_ocr": TokenGRPOConfig,
        "ar/janus_pro/online_r1_grpo_ocr": MultiSegmentTokenGRPOConfig,
        "diffusion/wan_2_1/offline_dpo_pickapic": DiffusionDPOConfig,
        "diffusion/cosmos_predict2_5/online_nft_kling_video_reward": DiffusionNFTConfig,
    }
    for name, expected_type in examples.items():
        cfg = load_config(f"experiment/{name}")
        algo_cfg = build_algorithm_config(cfg)
        assert isinstance(algo_cfg, expected_type)
        assert isinstance(algo_cfg, EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)])
        if name == "ar/janus_pro/online_grpo_ocr":
            assert algo_cfg.kl_estimator == "k2"
        if name == "ar/janus_pro/online_r1_grpo_ocr":
            assert cfg.trainer.entrypoint == (
                "vrl.scripts.ar.janus_pro.train:train_janus_pro_r1_ocr_grpo"
            )
            assert algo_cfg.train_segments == {
                "initial_image": True,
                "selfcheck_text": False,
                "final_image": True,
            }


def test_cosmos_diffusion_nft_video_reward_validation_config() -> None:
    """Checks Cosmos diffusion NFT video reward validation config."""
    cfg = load_config("experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward")

    validate_training_config(cfg)
    assert cfg.data.task_type == "text_to_video"
    assert cfg.data.manifest == "datasets/videophy/train.txt"
    assert cfg.data.eval_manifest == "datasets/videophy/eval.txt"
    assert cfg.data.source_report == "datasets/videophy/report.json"
    assert cfg.model.use_lora is True
    assert cfg.reward.kwargs.kling_video_reward.inference_runtime == "ray"
    assert cfg.reward.kwargs.kling_video_reward.artifact_format == "mp4"
    assert cfg.reward.kwargs.kling_video_reward.artifact_dir == (
        f"{cfg.trainer.output_dir}/reward_artifacts"
    )
    assert cfg.reward.kwargs.kling_video_reward.debug_dir == (
        f"{cfg.trainer.output_dir}/reward_debug"
    )
    assert cfg.reward.kwargs.kling_video_reward.reward_name == "KlingTeam/VideoReward@main"
    assert "model_factory" not in cfg.reward.kwargs.kling_video_reward.worker_config
    assert "reward_model_name" not in cfg.reward.kwargs.kling_video_reward.worker_config
    assert cfg.reward.kwargs.kling_video_reward.worker_config.local_files_only is True
    assert cfg.distributed.resources.reward.num_gpus == 1
    # share_with_rollout is the placement decision; the release lifecycle flags
    # derive from it in resolve_distributed_resources (pinned in
    # tests/ray/test_resources.py), so the YAML no longer sets them.
    assert cfg.distributed.resources.reward.share_with_rollout is True
    # cosmos-rl parity (cosmos-predict2-5-2b-720-nft.toml): n_generation=12,
    # mini_batch=6 prompts/step, 10 epochs, bf16 params. The 72/step effective
    # batch runs single-GPU via sequential gen + gradient accumulation.
    assert cfg.rollout.n == 12
    assert cfg.rollout.rollout_batch_size == 6
    assert cfg.rollout.sample_batch_size == 1
    assert cfg.trainer.total_epochs == 10
    assert cfg.precision == "bf16"
    assert cfg.production.kling_video_reward.enabled is True


def test_cosmos_motion_physics_config() -> None:
    # DanceGRPO-style motion-quality core + explicit physics commonsense term,
    # on the cosmos-rl NFT regime (10-step no-CFG, not the 4-step variant).
    """Checks Cosmos motion physics config."""
    cfg = load_config("experiment/diffusion/cosmos_predict2_5/online_nft_motion_physics")

    validate_training_config(cfg)
    assert cfg.algorithm.kind == "diffusion_nft"
    assert cfg.sampling.num_steps == 10
    assert cfg.sampling.cfg is False
    assert cfg.actor.timestep_fraction == 0.99
    # Weighted compound: Kling Motion-Quality (0.7) + VideoCon-Physics (0.3).
    assert cfg.reward.components.kling_video_reward == 0.7
    assert cfg.reward.components.videocon_physics == 0.3
    assert cfg.reward.kwargs.kling_video_reward.score_key == "motion_quality"
    assert cfg.reward.kwargs.kling_video_reward.artifact_format == "mp4"
    assert cfg.reward.kwargs.videocon_physics.score_key == "physical_commonsense"
    assert cfg.reward.kwargs.videocon_physics.artifact_format == "mp4"
    assert cfg.production.kling_video_reward.enabled is True


def test_cosmos_v2w_reference_route_config() -> None:
    """Checks Cosmos V2W reference route config."""
    cfg = load_config("experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference")

    validate_training_config(cfg)
    assert cfg.data.task_type == "video2world"
    assert cfg.data.manifest == "data/external/video_world/manifests/robot_train.jsonl"
    assert cfg.data.eval_manifest == "data/external/video_world/manifests/robot_eval.jsonl"
    assert cfg.data.source_report == "data/external/video_world/robot_report.json"
    assert cfg.cosmos.reference_mode == "per_sample"
    assert cfg.model.reference_image == ""
    assert cfg.reward.kwargs.kling_video_reward.inference_runtime == "ray"
    assert cfg.reward.kwargs.kling_video_reward.artifact_format == "mp4"
    assert cfg.reward.kwargs.kling_video_reward.reward_name == "KlingTeam/VideoReward@main"
    assert "model_factory" not in cfg.reward.kwargs.kling_video_reward.worker_config
    assert cfg.reward.kwargs.kling_video_reward.worker_config.local_files_only is True
    assert cfg.distributed.resources.reward.share_with_rollout is True


def test_cosmos_v2w_production_validation_accepts_source_backed_data(
    tmp_path: Path,
) -> None:
    """Checks Cosmos V2W production validation accepts source-backed data."""
    data_root = tmp_path / "external"
    reference = data_root / "video_world" / "references" / "ref.ppm"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    metadata = {
        "source": "droid",
        "source_repo": "lerobot/droid_100",
        "source_split": "main",
        "source_episode": "episode_train",
        "source_video": "videos/camera/chunk-000/file-000.mp4",
        "source_frame_index": 0,
        "decode_method": "pyav_http_first_frame",
        "conditioning": "first_frame",
    }
    train = tmp_path / "robot_train.jsonl"
    eval_manifest = tmp_path / "robot_eval.jsonl"
    train.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves toward the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "metadata": metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    eval_metadata = dict(metadata, source_episode="episode_eval")
    eval_manifest.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves away from the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "metadata": eval_metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "robot_report.json"
    report.write_text(
        json.dumps(
            {
                "dataset": "video_world_bridge",
                "source": "droid",
                "repo_id": "lerobot/droid_100",
                "source_split": "main",
                "decode_method": "pyav_http_first_frame",
                "train_rows": 1,
                "eval_rows": 1,
                "train_manifest": train.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
                "reference_dir": reference.parent.as_posix(),
                "validation_summary": {"row_count": 1},
            },
        ),
        encoding="utf-8",
    )
    cfg = load_config(
        "experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference",
        overrides=[
            "production.kling_video_reward.enabled=true",
            f"data.manifest={train.as_posix()}",
            f"data.eval_manifest={eval_manifest.as_posix()}",
            f"data.source_report={report.as_posix()}",
            f"data.artifact_data_root={data_root.as_posix()}",
        ],
    )

    validate_training_config(cfg)


def test_wan_video_reward_production_config() -> None:
    """Checks Wan video reward production config."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_kling_video_reward")

    validate_training_config(cfg)
    assert cfg.model.family == "wan"
    assert cfg.reward.components.kling_video_reward == 1.0
    assert cfg.reward.kwargs.kling_video_reward.inference_runtime == "ray"
    assert cfg.reward.kwargs.kling_video_reward.media_type == "video"
    assert cfg.reward.kwargs.kling_video_reward.artifact_format == "mp4"
    assert cfg.reward.kwargs.kling_video_reward.artifact_dir == (
        f"{cfg.trainer.output_dir}/reward_artifacts"
    )
    assert cfg.reward.kwargs.kling_video_reward.debug_dir == (
        f"{cfg.trainer.output_dir}/reward_debug"
    )
    assert cfg.reward.kwargs.kling_video_reward.reward_name == "KlingTeam/VideoReward@main"
    assert "model_factory" not in cfg.reward.kwargs.kling_video_reward.worker_config
    assert "backend" not in cfg.reward.kwargs.kling_video_reward.worker_config
    assert "score_key_map" not in cfg.reward.kwargs.kling_video_reward.worker_config
    assert "reward_model_name" not in cfg.reward.kwargs.kling_video_reward.worker_config
    assert cfg.reward.kwargs.kling_video_reward.worker_config.local_files_only is True
    assert cfg.rollout.n == 4
    assert cfg.rollout.rollout_batch_size == 1
    assert cfg.rollout.sample_batch_size == 1
    assert cfg.rollout.noise_level == pytest.approx(0.7)
    assert cfg.rollout.sde.type == "cps"
    assert cfg.data.task_type == "text_to_video"
    assert cfg.data.manifest == "datasets/videophy/train.txt"
    assert cfg.data.eval_manifest == "datasets/videophy/eval.txt"
    assert cfg.data.source_report == "datasets/videophy/report.json"
    assert cfg.distributed.resources.reward.share_with_rollout is True


def test_wan_i2v_physics_config() -> None:
    """Checks Wan I2V physics config."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_physics_i2v")

    validate_training_config(cfg)
    assert cfg.model.family == "wan_2_1_i2v"
    assert cfg.model.task_variant == "i2v"
    assert cfg.data.loader == "prompt_image_manifest"
    assert cfg.data.task_type == "image_to_video"
    assert cfg.data.manifest == "data/external/videophy_i2v/manifests/train.jsonl"
    assert cfg.data.eval_manifest == "data/external/videophy_i2v/manifests/eval.jsonl"
    assert cfg.data.artifact_data_root == "data/external/videophy_i2v"
    assert cfg.data.source_report == "data/external/videophy_i2v/report.json"
    assert cfg.sampling.height == 480
    assert cfg.sampling.width == 832
    assert cfg.sampling.num_frames == 81
    assert cfg.sampling.guidance_scale == pytest.approx(5.0)
    assert cfg.rollout.n == 2
    assert cfg.rollout.rollout_batch_size == 1
    assert cfg.trainer.entrypoint == (
        "vrl.scripts.diffusion.wan_2_1.train:train_wan_2_1_i2v_grpo"
    )
    assert cfg.production.kling_video_reward.enabled is False


def test_wan_i2v_production_validation_accepts_source_backed_data(tmp_path: Path) -> None:
    """Checks Wan I2V production validation accepts source-backed data."""
    data_root = tmp_path / "videophy_i2v"
    train_image = data_root / "images" / "train" / "000.ppm"
    eval_image = data_root / "images" / "eval" / "000.ppm"
    train_image.parent.mkdir(parents=True)
    eval_image.parent.mkdir(parents=True)
    train_image.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    eval_image.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    metadata = {
        "source": "videophy",
        "source_repo": "videophysics/videophy_test_public",
        "source_split": "test",
        "source_csv_row": 0,
        "source_video_url": "https://videophysics.example/train.mp4",
        "source_frame_index": 0,
        "decode_method": "imageio_ffmpeg_first_frame",
        "conditioning": "first_frame",
    }
    train_manifest = data_root / "manifests" / "train.jsonl"
    eval_manifest = data_root / "manifests" / "eval.jsonl"
    train_manifest.parent.mkdir(parents=True)
    train_manifest.write_text(
        json.dumps(
            {
                "image": "images/train/000.ppm",
                "caption": "A wheel rolls.",
                "task_type": "image_to_video",
                "metadata": metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    eval_metadata = dict(metadata, source_video_url="https://videophysics.example/eval.mp4")
    eval_manifest.write_text(
        json.dumps(
            {
                "image": "images/eval/000.ppm",
                "caption": "Honey diffuses.",
                "task_type": "image_to_video",
                "metadata": eval_metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    report = data_root / "report.json"
    report.write_text(
        json.dumps(
            {
                "dataset": "videophy_i2v",
                "source_repo": "videophysics/videophy_test_public",
                "source_csv": "videophy_test_public.csv",
                "source_split": "test",
                "decode_method": "imageio_ffmpeg_first_frame",
                "train_rows": 1,
                "eval_rows": 1,
                "train_manifest": train_manifest.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
                "reference_dir": (data_root / "images").as_posix(),
            },
        ),
        encoding="utf-8",
    )

    cfg = load_config(
        "experiment/diffusion/wan_2_1/online_grpo_physics_i2v",
        overrides=[
            "production.kling_video_reward.enabled=true",
            f"data.manifest={train_manifest.as_posix()}",
            f"data.eval_manifest={eval_manifest.as_posix()}",
            f"data.source_report={report.as_posix()}",
            f"data.artifact_data_root={data_root.as_posix()}",
        ],
    )

    validate_training_config(cfg)


def test_wan_video_reward_production_config_requires_reward_name() -> None:
    """Checks Wan video reward production config requires reward name."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.reward_name = ""

    with pytest.raises(ValueError, match="reward_name"):
        validate_training_config(cfg)


def test_wan_video_reward_production_rejects_extra_loader_fields() -> None:
    """Checks Wan video reward production rejects extra loader fields."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.worker_config.import_path = "fake:thing"

    with pytest.raises(ValueError, match="remove extra loader fields"):
        validate_training_config(cfg)

    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.worker_config.model_factory = "fake:factory"

    with pytest.raises(ValueError, match="remove extra loader fields"):
        validate_training_config(cfg)


def test_anima_safe_reward_config_uses_cpu_nsfw_penalty() -> None:
    """Checks Anima safe reward config uses CPU NSFW penalty."""
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



def test_anima_config_keeps_artifact_names_without_local_paths() -> None:
    """Checks Anima config keeps artifact names without local paths."""
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


def test_unified_train_entrypoint_reads_yaml_entrypoint() -> None:
    """Checks unified train entrypoint reads YAML entrypoint."""
    from vrl.scripts.train import _import_callable, resolve_train_target

    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    target = resolve_train_target(cfg)

    assert target.import_path == cfg.trainer.entrypoint
    assert callable(_import_callable(target.import_path))


def test_cli_overrides_reach_typed_trainer_config() -> None:
    """Checks CLI overrides reach typed trainer config."""
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=[
            "trainer.resume_from=/tmp/checkpoint-10",
            "trainer.torch_profiler.enabled=true",
            "trainer.torch_profiler.activities=[cpu]",
            "actor.drop_zero_advantage=false",
        ],
    )
    trainer = build_configs(cfg)["trainer"]

    assert trainer.resume_from == "/tmp/checkpoint-10"
    assert trainer.torch_profiler.enabled is True
    assert trainer.torch_profiler.activities == ("cpu",)
    assert trainer.drop_zero_advantage is False


def test_invalid_algorithm_kind_fails_fast() -> None:
    """Checks invalid algorithm kind fails fast."""
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo", "adv_estimator": "dpo"}})
    with pytest.raises(ValueError, match="adv_estimator"):
        build_algorithm_config(cfg)

    cfg = OmegaConf.create({"algorithm": {"kind": "qpo"}})
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
        build_algorithm_config(cfg)


def test_unknown_algorithm_config_fields_fail_fast() -> None:
    """Checks unknown algorithm config fields fail fast."""
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo", "unknown_knob": 1}})

    with pytest.raises(ValueError, match=r"algorithm\.unknown_knob"):
        build_algorithm_config(cfg)


def test_removed_algorithm_zero_advantage_fields_fail_fast() -> None:
    """Checks removed algorithm zero advantage fields fail fast."""
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo", "per_prompt_stat_tracking": False}})
    with pytest.raises(ValueError, match=r"actor\.drop_zero_advantage"):
        build_algorithm_config(cfg)

    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    del cfg.actor["drop_zero_advantage"]
    with pytest.raises(ValueError, match=r"actor\.drop_zero_advantage"):
        build_configs(cfg)

    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    cfg.algorithm.per_prompt_stat_tracking = False
    with pytest.raises(ValueError, match=r"actor\.drop_zero_advantage"):
        validate_training_config(cfg)


def test_reward_backbone_kwargs_are_required() -> None:
    """Checks reward backbone kwargs are required."""
    cfg = load_config("experiment/diffusion/cosmos_predict2/online_grpo_kling_video_reward")
    del cfg.reward.kwargs.kling_video_reward["score_key"]

    with pytest.raises(ValueError, match="kling_video_reward"):
        validate_reward_config(cfg)


def test_negative_reward_component_weights_are_rejected() -> None:
    """Checks negative reward component weights are rejected."""
    cfg = load_config("experiment/diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety")
    cfg.reward.components.nsfw_safety = -0.5

    with pytest.raises(ValueError, match=r"reward\.components\.nsfw_safety must be >= 0"):
        validate_reward_config(cfg)


def test_required_training_fields_fail_fast() -> None:
    """Checks required training fields fail fast."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_ocr")
    cfg.trainer.output_dir = "???"
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        validate_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    """Checks DPO allows explicit null max train samples."""
    cfg = load_config("experiment/diffusion/wan_2_1/offline_dpo_pickapic")
    cfg.data.max_train_samples = None

    assert optional_none(cfg, "data.max_train_samples") is None
    validate_training_config(cfg)


def test_dpo_recipe_declares_trainer_only_resource_plan() -> None:
    """Checks DPO recipe declares trainer only resource plan."""
    from vrl.ray.resources import resolve_distributed_resources

    cfg = load_config(
        "experiment/diffusion/wan_2_1/offline_dpo_pickapic",
        overrides=["distributed.resources.visible_devices=[0]"],
    )
    resolved = resolve_distributed_resources(cfg)

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == ()
    assert resolved.rollout_num_workers == 0
