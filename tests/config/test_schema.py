"""Tests for the typed config schema boundary (vrl/config/schema.py)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import (
    AlgorithmConfig,
    DataConfig,
    RewardConfig,
    TrainingSection,
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


def _kling_video_reward_kwargs(**overrides) -> dict:
    base = {
        "execution": "pool",
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
    """Checks valid algorithm kinds are accepted."""
    algo = AlgorithmConfig(kind=kind)
    assert algo.kind == kind


def test_unknown_algorithm_kind_raises() -> None:
    """Checks unknown algorithm kind raises."""
    cfg = _minimal_grpo_cfg()
    cfg.algorithm.kind = "qpo"
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
        parse_config(cfg)


def test_unknown_algorithm_keys_warn_and_load() -> None:
    """Removed keys, typos, and never-seen keys all warn — none of them raise."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create(
        {"algorithm": {"kind": "grpo", "adv_estimator": "dpo", "future_field": True}}
    )
    algo = AlgorithmConfig.model_validate(
        OmegaConf.to_container(cfg.algorithm, resolve=True)
    )
    assert algo.kind == "grpo"  # loads fine
    unknown = find_unknown_keys(cfg)
    assert "algorithm.adv_estimator" in unknown
    assert "algorithm.future_field" in unknown


# ── rollout / sampling string-setting Literals ────────────────────────────────


@pytest.mark.parametrize("mode", ["native", "sde"])
def test_valid_denoise_modes_accepted(mode: str) -> None:
    """Both denoise modes validate; the Literal is the user-facing allow-list."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.denoise_mode = mode
    assert parse_config(cfg).rollout.denoise_mode == mode


def test_unknown_denoise_mode_raises() -> None:
    """An out-of-set denoise_mode is rejected at parse with the dotted path."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.denoise_mode = "bogus"
    with pytest.raises(ValueError, match=r"unknown rollout\.denoise_mode"):
        parse_config(cfg)


def test_attention_backend_defaults_to_vllm_paged() -> None:
    """attention_backend is a registered, typed key (no false unknown-key warning)."""
    cfg = _minimal_grpo_cfg()
    cfg.sampling = {}
    assert parse_config(cfg).sampling.attention_backend == "vllm_paged"


def test_unknown_attention_backend_raises() -> None:
    """An out-of-set attention_backend is rejected at parse with the dotted path."""
    cfg = _minimal_grpo_cfg()
    cfg.sampling = {"attention_backend": "bogus"}
    with pytest.raises(ValueError, match=r"unknown sampling\.attention_backend"):
        parse_config(cfg)


def test_unknown_final_image_policy_raises() -> None:
    """final_image_policy is Literal-typed regardless of algorithm kind."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.final_image_policy = "bogus"
    with pytest.raises(ValueError, match=r"unknown rollout\.final_image_policy"):
        parse_config(cfg)


# ── distributed.training strategy ─────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["single_process", "fsdp"])
def test_valid_training_strategies_are_accepted(strategy: str) -> None:
    """Both readiness strategies validate; the Literal is the only allow-list."""
    section = TrainingSection(strategy=strategy)
    assert section.strategy == strategy


def test_unknown_training_strategy_raises() -> None:
    """An unimplemented/typo strategy is rejected at parse time, not silently run."""
    cfg = _minimal_grpo_cfg(distributed={"training": {"strategy": "deepspeed"}})
    with pytest.raises(ValueError, match=r"unknown distributed\.training\.strategy"):
        parse_config(cfg)


def test_training_keys_are_registered_not_unknown() -> None:
    """distributed.training.* keys are known to the unknown-key walker."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "training": {
                "strategy": "fsdp",
                "num_nodes": 1,
                "gpus_per_node": 2,
            }
        }
    )
    unknown = find_unknown_keys(cfg)
    assert not [k for k in unknown if k.startswith("distributed.training")]


# ── distributed.rollout knobs ─────────────────────────────────────────────────


def test_unknown_chunk_placement_strategy_raises() -> None:
    """A typo chunk placement strategy is rejected at parse time, not at launch."""
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"chunk_placement_strategy": "work_stealing"}},
    )
    with pytest.raises(
        ValueError, match=r"unknown distributed\.rollout\.chunk_placement_strategy",
    ):
        parse_config(cfg)


def test_legacy_sync_trainable_state_string_rejected() -> None:
    """sync_trainable_state is a plain bool now; the legacy "lora_only" string is
    rejected at parse, not silently coerced (no backward-compat)."""
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"sync_trainable_state": "lora_only"}},
    )
    with pytest.raises(ValueError, match=r"valid boolean"):
        parse_config(cfg)


def test_rollout_keys_are_registered_not_unknown() -> None:
    """distributed.rollout.* keys (incl. nested colocate) are known to the walker."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "cpus_per_worker": 2.0,
                "max_inflight_chunks_per_worker": 2,
                "chunk_placement_strategy": "dynamic",
                "sync_trainable_state": False,
                "colocate": {"memory_fraction": 0.5},
            }
        }
    )
    unknown = find_unknown_keys(cfg)
    assert not [k for k in unknown if k.startswith("distributed.rollout")]


# ── Data loader discriminator ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "loader",
    ["prompt_manifest", "prompt_image_manifest", "pickapic_preference"],
)
def test_valid_data_loaders_are_accepted(loader: str) -> None:
    """Checks valid data loaders are accepted."""
    if loader == "prompt_manifest":
        data = DataConfig(
            loader=loader,
            manifest="datasets/ocr/train.txt",
            preprocessing={"format": "text"},
            sampler={"type": "random_without_replacement"},
        )
    elif loader == "prompt_image_manifest":
        data = DataConfig(
            loader=loader,
            manifest="data/external/videophy_i2v/manifests/train.jsonl",
            eval_manifest="data/external/videophy_i2v/manifests/eval.jsonl",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "caption_field": "caption",
                "media_type": "video",
                "conditioning": "reference_image",
            },
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
    """Checks unknown data loader raises."""
    cfg = _minimal_grpo_cfg()
    cfg.data.loader = "s3_loader"
    with pytest.raises(ValueError, match=r"unknown data\.loader"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("image_caption_jsonl", "prompt_image_manifest"),
        ("jsonl", "prompt_manifest"),
        ("text", "prompt_manifest"),
    ],
)
def test_omitted_loader_derives_from_preprocessing_format(fmt: str, expected: str) -> None:
    """An omitted data.loader is derived from preprocessing.format for the prompt-* family."""
    if expected == "prompt_image_manifest":
        data = DataConfig(
            manifest="data/external/videophy_i2v/manifests/train.jsonl",
            eval_manifest="data/external/videophy_i2v/manifests/eval.jsonl",
            preprocessing={
                "format": fmt,
                "image_field": "image",
                "caption_field": "caption",
                "media_type": "video",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )
    else:
        data = DataConfig(
            manifest="datasets/ocr/train.txt",
            preprocessing={"format": fmt},
            sampler={"type": "random_without_replacement"},
        )
    assert data.loader == expected


def test_prompt_image_manifest_requires_image_caption_fields() -> None:
    """Checks prompt image manifest requires image caption fields."""
    with pytest.raises(ValueError, match=r"data\.preprocessing\.caption_field"):
        DataConfig(
            loader="prompt_image_manifest",
            manifest="x",
            eval_manifest="y",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "media_type": "video",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


# ── Sampler type literal ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sampler_type",
    ["random_without_replacement", "sequential_window"],
)
def test_valid_sampler_types_are_accepted(sampler_type: str) -> None:
    """Checks valid sampler types are accepted."""
    data = DataConfig(
        loader="prompt_manifest",
        manifest="x",
        preprocessing={"format": "text"},
        sampler={"type": sampler_type},
    )
    assert data.sampler["type"] == sampler_type


def test_unknown_sampler_type_raises() -> None:
    """Checks unknown sampler type raises."""
    with pytest.raises(ValueError, match=r"unknown data\.sampler\.type"):
        DataConfig(
            loader="prompt_manifest",
            manifest="x",
            preprocessing={},
            sampler={"type": "round_robin"},
        )


# ── Reward weight validation ──────────────────────────────────────────────────


def test_zero_weight_component_skips_video_reward_check() -> None:
    """Checks zero weight component skips video reward check."""
    cfg = RewardConfig.model_validate(
        {"components": {"kling_video_reward": 0.0}, "kwargs": {}}
    )
    assert cfg.components["kling_video_reward"] == 0.0


def test_negative_reward_weight_raises() -> None:
    """Checks negative reward weight raises."""
    with pytest.raises(ValueError, match=r"must be >= 0"):
        RewardConfig.model_validate(
            {"components": {"aesthetic": -1.0}, "kwargs": {"aesthetic": {"model_name": "x"}}}
        )


def test_non_numeric_reward_weight_raises() -> None:
    """Checks non numeric reward weight raises."""
    with pytest.raises(ValueError, match="must be numeric"):
        RewardConfig.model_validate(
            {"components": {"aesthetic": "heavy"}, "kwargs": {}}
        )



def test_grpo_requires_valid_sde_type() -> None:
    """Checks GRPO requires valid SDE type."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "euler"
    with pytest.raises(ValueError, match=r"unknown rollout\.sde\.type"):
        parse_config(cfg)


def test_grpo_accepts_cps_sde_type() -> None:
    """Checks GRPO accepts cps SDE type."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "cps"
    parsed = parse_config(cfg)
    assert parsed.algorithm.kind == "grpo"


def test_diffusion_nft_requires_valid_sde_type() -> None:
    """Checks diffusion NFT requires valid SDE type."""
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
    with pytest.raises(ValueError, match=r"unknown rollout\.sde\.type"):
        parse_config(cfg)


def test_token_grpo_multisegment_requires_janus_pro_family() -> None:
    """Checks token GRPO multisegment requires Janus pro family."""
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
    """Checks token GRPO multisegment policy mismatch raises."""
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


def test_token_grpo_multisegment_final_image_policy_single_source() -> None:
    """final_image_policy may be set in rollout alone; the sampling.r1 duplicate is
    no longer required (the collector resolves it rollout-first)."""
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
            "sampling": {"r1": {"train_segments": {"initial_image": True}}},
        }
    )
    assert parse_config(cfg).rollout.final_image_policy == "always_generate"


def test_production_video_reward_structural_rules() -> None:
    """Checks production video reward structural rules."""
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
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "execution": "pool",
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        }
    )
    from vrl.config.validation import validate_production_reward_contract

    parse_config(cfg)  # schema parse stays clean
    validate_production_reward_contract(cfg)


def test_production_video_reward_accepts_image_to_video_task_type() -> None:
    """Checks production video reward accepts image to video task type."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_image_manifest",
                "manifest": "x",
                "eval_manifest": "y",
                "preprocessing": {
                    "format": "image_caption_jsonl",
                    "image_field": "image",
                    "caption_field": "caption",
                    "media_type": "video",
                    "conditioning": "reference_image",
                },
                "sampler": {"type": "random_without_replacement"},
                "task_type": "image_to_video",
            },
            "rollout": {"sde": {"type": "cps"}},
            "reward": {
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "execution": "pool",
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        },
    )

    parsed = parse_config(cfg)

    assert parsed.data.task_type == "image_to_video"


def test_production_video_reward_missing_reward_name_raises() -> None:
    """Checks production video reward missing reward name raises."""
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
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "execution": "pool",
                        "reward_name": "",  # empty
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        }
    )
    from vrl.config.validation import validate_production_reward_contract

    with pytest.raises(ValueError, match="reward_name"):
        validate_production_reward_contract(cfg)


def test_production_video_reward_forbidden_worker_key_raises() -> None:
    """Checks production video reward forbidden worker key raises."""
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
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "execution": "pool",
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {"import_path": "bad:factory"},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        }
    )
    from vrl.config.validation import validate_production_reward_contract

    with pytest.raises(ValueError, match="remove extra loader fields"):
        validate_production_reward_contract(cfg)


# ── Missing field mapping (??? → ValueError) ──────────────────────────────────


def test_missing_mandatory_value_produces_repo_standard_message() -> None:
    """Checks missing mandatory value produces repo standard message."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "sde"
    # Inject an OmegaConf mandatory-missing marker
    OmegaConf.update(cfg, "algorithm.kind", "???")
    with pytest.raises(ValueError, match="config missing required field"):
        parse_config(cfg)


# ── extra="ignore" migration policy ──────────────────────────────────────────


def test_unknown_top_level_sections_are_silently_ignored() -> None:
    """Checks unknown top level sections are silently ignored."""
    cfg = _minimal_grpo_cfg()
    OmegaConf.update(cfg, "some_future_section.foo", "bar")
    # Must not raise even though some_future_section is not in RootConfig
    parsed = parse_config(cfg)
    assert parsed.algorithm.kind == "grpo"


def test_unknown_reward_component_with_unknown_kwargs_is_accepted() -> None:
    """Checks unknown reward component with unknown kwargs is accepted."""
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
