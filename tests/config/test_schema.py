"""Tests for the typed config schema boundary (vrl/config/schema.py)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import (
    AlgorithmConfig,
    DataConfig,
    RewardConfig,
    RootConfig,
    VideoRewardKwargs,
    parse_config,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _minimal_grpo_cfg(**overrides):
    base = {
        "algorithm": {"kind": "grpo"},
        "data": {
            "loader": "prompt_manifest",
            "manifest": "datasets/ocr/train.txt",
            "preprocessing": {"format": "text"},
            "sampler": {"type": "random_without_replacement"},
        },
        "rollout": {"sde": {"type": "sde"}},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _video_reward_kwargs(**overrides) -> dict:
    base = {
        "inference_runtime": "ray",
        "reward_name": "org/model@main",
        "score_key": "overall",
        "worker_config": {"model_path": "/tmp/model"},
    }
    base.update(overrides)
    return base


# ── Algorithm kind discriminator ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "kind",
    ["grpo", "token_grpo", "token_grpo_multisegment", "diffusion_dpo", "diffusion_nft"],
)
def test_valid_algorithm_kinds_are_accepted(kind: str) -> None:
    algo = AlgorithmConfig(kind=kind)
    assert algo.kind == kind


def test_unknown_algorithm_kind_raises() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.algorithm.kind = "qpo"
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
        parse_config(cfg)


def test_adv_estimator_raises_with_migration_message() -> None:
    with pytest.raises(ValueError, match="adv_estimator"):
        AlgorithmConfig(kind="grpo", adv_estimator="dpo")


def test_extra_algorithm_fields_are_ignored() -> None:
    # extra="ignore" — unknown keys must not raise
    algo = AlgorithmConfig.model_validate({"kind": "grpo", "init_kl_coef": 0.04, "future_field": True})
    assert algo.kind == "grpo"


# ── Data loader discriminator ─────────────────────────────────────────────────


@pytest.mark.parametrize("loader", ["prompt_manifest", "pickapic_preference"])
def test_valid_data_loaders_are_accepted(loader: str) -> None:
    if loader == "prompt_manifest":
        data = DataConfig(
            loader=loader,
            manifest="datasets/ocr/train.txt",
            preprocessing={"format": "text"},
            sampler={"type": "random_without_replacement"},
        )
    else:
        data = DataConfig(
            loader=loader,
            dataset_name="org/dataset",
            split="train",
            cache_dir="/tmp/cache",
            preprocessing={"resolution": 512, "random_crop": False, "horizontal_flip": True},
            sampler={"shuffle": True, "drop_last": True, "dataloader_num_workers": 4},
        )
    assert data.loader == loader


def test_unknown_data_loader_raises() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.data.loader = "s3_loader"
    with pytest.raises(ValueError, match=r"unknown data\.loader"):
        parse_config(cfg)


# ── Sampler type literal ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sampler_type",
    ["random_without_replacement", "sequential_window"],
)
def test_valid_sampler_types_are_accepted(sampler_type: str) -> None:
    data = DataConfig(
        loader="prompt_manifest",
        manifest="x",
        preprocessing={"format": "text"},
        sampler={"type": sampler_type},
    )
    assert data.sampler["type"] == sampler_type


def test_unknown_sampler_type_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown data\.sampler\.type"):
        DataConfig(
            loader="prompt_manifest",
            manifest="x",
            preprocessing={},
            sampler={"type": "round_robin"},
        )


# ── Reward weight validation ──────────────────────────────────────────────────


def test_zero_weight_component_skips_video_reward_check() -> None:
    cfg = RewardConfig.model_validate(
        {"components": {"video_reward": 0.0}, "kwargs": {}}
    )
    assert cfg.components["video_reward"] == 0.0


def test_negative_reward_weight_raises() -> None:
    with pytest.raises(ValueError, match=r"must be >= 0"):
        RewardConfig.model_validate(
            {"components": {"aesthetic": -1.0}, "kwargs": {"aesthetic": {"model_name": "x"}}}
        )


def test_non_numeric_reward_weight_raises() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        RewardConfig.model_validate(
            {"components": {"aesthetic": "heavy"}, "kwargs": {}}
        )



# ── VideoRewardKwargs: removed field rejection ────────────────────────────────


def test_video_reward_backend_field_raises_specific_message() -> None:
    with pytest.raises(ValueError, match="backend is no longer supported"):
        VideoRewardKwargs.model_validate({**_video_reward_kwargs(), "backend": "http"})


@pytest.mark.parametrize(
    "removed_field",
    ["enqueue_url", "fetch_url", "token", "poll_interval_s", "max_wait_s", "stub_scale", "device"],
)
def test_video_reward_removed_endpoint_fields_raise(removed_field: str) -> None:
    with pytest.raises(ValueError, match="no longer supports external reward endpoint fields"):
        VideoRewardKwargs.model_validate({**_video_reward_kwargs(), removed_field: "value"})


def test_video_reward_non_ray_inference_runtime_raises() -> None:
    with pytest.raises(ValueError, match="inference_runtime must be 'ray'"):
        VideoRewardKwargs.model_validate(
            {**_video_reward_kwargs(), "inference_runtime": "local"}
        )


def test_video_reward_non_sync_scheduling_raises() -> None:
    with pytest.raises(ValueError, match="scheduling currently supports only 'sync'"):
        VideoRewardKwargs.model_validate(
            {**_video_reward_kwargs(), "scheduling": "async"}
        )


def test_video_reward_valid_kwargs_accepted() -> None:
    vr = VideoRewardKwargs.model_validate(_video_reward_kwargs())
    assert vr.inference_runtime == "ray"
    assert vr.scheduling == "sync"


def test_video_reward_extra_fields_are_ignored() -> None:
    vr = VideoRewardKwargs.model_validate(
        {**_video_reward_kwargs(), "artifact_dir": "/tmp/out", "timeout_s": 60.0}
    )
    assert vr.reward_name == "org/model@main"


# ── Cross-field validators ────────────────────────────────────────────────────


def test_grpo_requires_valid_sde_type() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "euler"
    with pytest.raises(ValueError, match=r"rollout\.sde\.type must be"):
        parse_config(cfg)


def test_grpo_accepts_cps_sde_type() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "cps"
    parsed = parse_config(cfg)
    assert parsed.algorithm.kind == "grpo"


def test_diffusion_nft_requires_valid_sde_type() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "diffusion_nft"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "rollout": {"sde": {"type": "invalid"}},
        }
    )
    with pytest.raises(ValueError, match=r"rollout\.sde\.type must be"):
        parse_config(cfg)


def test_token_grpo_multisegment_requires_janus_pro_family() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo_multisegment"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "other_model"},
            "rollout": {"final_image_policy": "always_generate"},
            "sampling": {"r1": {"final_image_policy": "always_generate"}},
        }
    )
    with pytest.raises(ValueError, match="janus_pro"):
        parse_config(cfg)


def test_token_grpo_multisegment_policy_mismatch_raises() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo_multisegment"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "janus_pro"},
            "rollout": {"final_image_policy": "always_generate"},
            "sampling": {"r1": {"final_image_policy": "use_selfcheck"}},  # mismatch
        }
    )
    with pytest.raises(ValueError, match=r"sampling\.r1\.final_image_policy must match"):
        parse_config(cfg)


def test_production_video_reward_structural_rules() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
                "task_type": "text_to_video",
            },
            "rollout": {"sde": {"type": "cps"}},
            "reward": {
                "components": {"video_reward": 1.0},
                "kwargs": {
                    "video_reward": {
                        "inference_runtime": "ray",
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"video_reward": {"enabled": True}},
        }
    )
    parsed = parse_config(cfg)
    assert parsed.production.video_reward.enabled is True


def test_production_video_reward_missing_reward_name_raises() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
                "task_type": "text_to_video",
            },
            "rollout": {"sde": {"type": "cps"}},
            "reward": {
                "components": {"video_reward": 1.0},
                "kwargs": {
                    "video_reward": {
                        "inference_runtime": "ray",
                        "reward_name": "",  # empty
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"video_reward": {"enabled": True}},
        }
    )
    with pytest.raises(ValueError, match="reward_name"):
        parse_config(cfg)


def test_production_video_reward_forbidden_worker_key_raises() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
                "task_type": "text_to_video",
            },
            "rollout": {"sde": {"type": "cps"}},
            "reward": {
                "components": {"video_reward": 1.0},
                "kwargs": {
                    "video_reward": {
                        "inference_runtime": "ray",
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {"import_path": "bad:factory"},
                    }
                },
            },
            "production": {"video_reward": {"enabled": True}},
        }
    )
    with pytest.raises(ValueError, match="remove extra loader fields"):
        parse_config(cfg)


# ── Missing field mapping (??? → ValueError) ──────────────────────────────────


def test_missing_mandatory_value_produces_repo_standard_message() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "sde"
    # Inject an OmegaConf mandatory-missing marker
    OmegaConf.update(cfg, "algorithm.kind", "???")
    with pytest.raises(ValueError, match="config missing required field"):
        parse_config(cfg)


# ── extra="ignore" migration policy ──────────────────────────────────────────


def test_unknown_top_level_sections_are_silently_ignored() -> None:
    cfg = _minimal_grpo_cfg()
    OmegaConf.update(cfg, "some_future_section.foo", "bar")
    # Must not raise even though some_future_section is not in RootConfig
    parsed = parse_config(cfg)
    assert parsed.algorithm.kind == "grpo"


def test_unknown_reward_component_with_unknown_kwargs_is_accepted() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "rollout": {"sde": {"type": "sde"}},
            "reward": {
                "components": {"custom_reward": 1.0},
                "kwargs": {"custom_reward": {"model_repo": "org/model"}},
            },
        }
    )
    parsed = parse_config(cfg)
    assert parsed.reward.components["custom_reward"] == 1.0
