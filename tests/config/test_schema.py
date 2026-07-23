"""Tests for the typed config schema boundary (vrl/config/schema.py)."""

from __future__ import annotations

import typing

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import (
    AlgorithmConfig,
    DataConfig,
    RewardConfig,
    parse_config,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _literal_args(annotation) -> tuple[str, ...]:
    """Flatten a ``Literal[...]`` or ``Literal[...] | None`` annotation into its
    members, so the allow-list tests derive their cases from the schema's single
    source of truth instead of hand-copying the Literal members (a copy never
    sees a newly added member, leaving it silently untested)."""
    members = typing.get_args(annotation)
    # Optional[Literal[...]]: unwrap each non-None union member's Literal args.
    if any(m is type(None) for m in members):
        return tuple(a for m in members if m is not type(None) for a in typing.get_args(m))
    return members


def _minimal_grpo_cfg(**overrides):
    base = {
        "algorithm": {"kind": "grpo"},
        "data": {
            "loader": "prompt_manifest",
            "manifest": "datasets/ocr/train.txt",
            "preprocessing": {"format": "text"},
            "sampler": {"type": "random_without_replacement"},
        },
        "rollout": {"sde": {"type": "flow_grpo"}},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _kling_video_reward_kwargs(**overrides) -> dict:
    base = {
        "sleep_offload": True,
        "reward_name": "org/model@main",
        "score_key": "overall",
        "worker_config": {"model_path": "/tmp/model"},
    }
    base.update(overrides)
    return base


# ── Algorithm kind discriminator ──────────────────────────────────────────────


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
    algo = AlgorithmConfig.model_validate(OmegaConf.to_container(cfg.algorithm, resolve=True))
    assert algo.kind == "grpo"  # loads fine
    unknown = find_unknown_keys(cfg)
    assert "algorithm.adv_estimator" in unknown
    assert "algorithm.future_field" in unknown


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("grpo", "flow_kl_use_dt", True),
        ("dance_grpo", "sft_weight", 0.1),
        ("flow_dppo", "add_kl_coefficient", 0.2),
        ("grpo_guard", "clip_ratio", 0.2),
        ("token_grpo", "kl_estimator", "k1"),
        ("token_grpo_multisegment", "segment_weights", [1.0]),
        ("diffusion_dpo", "beta", 5000.0),
        ("diffusion_nft", "nft_beta", 0.1),
    ],
)
def test_algorithm_keys_derive_from_selected_runtime_config(
    kind: str,
    field: str,
    value: object,
) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"algorithm": {"kind": kind, field: value}})
    assert find_unknown_keys(cfg) == []


@pytest.mark.parametrize(
    ("kind", "foreign_field"),
    [("grpo", "beta"), ("diffusion_dpo", "clip_ratio")],
)
def test_algorithm_keys_are_scoped_to_selected_kind(kind: str, foreign_field: str) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"algorithm": {"kind": kind, foreign_field: 1}})
    assert find_unknown_keys(cfg) == [f"algorithm.{foreign_field}"]


@pytest.mark.parametrize("family", ["wan_2_1", "echo"])
def test_removed_task_variant_is_an_unknown_model_key(family: str) -> None:
    """A deleted no-op knob must not remain accepted by family schemas."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"model": {"family": family, "task_variant": "unused"}})
    assert find_unknown_keys(cfg) == ["model.task_variant"]


def test_algorithm_unknown_key_selector_defers_invalid_kind_to_schema() -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"algorithm": {"kind": "qpo", "future_field": 1}})
    assert find_unknown_keys(cfg) == ["algorithm.future_field"]


def test_algorithm_dispatch_covers_schema_kind_vocabulary() -> None:
    from vrl.config.algorithm import algorithm_config_class

    kinds = _literal_args(AlgorithmConfig.model_fields["kind"].annotation)
    assert kinds
    assert all(algorithm_config_class(kind) for kind in kinds)


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


# ── model family scoped keys ──────────────────────────────────────────────────


def test_wan_model_keys_are_scoped_to_wan_family() -> None:
    """Wan dual-stage/offload keys are accepted only for Wan model families."""
    from vrl.config.unknown_keys import find_unknown_keys

    wan_cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert find_unknown_keys(wan_cfg) == []

    alias_cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert find_unknown_keys(alias_cfg) == []

    sd3_cfg = OmegaConf.create(
        {
            "model": {
                "family": "sd3_5",
                "path": "stabilityai/stable-diffusion-3.5-medium",
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert find_unknown_keys(sd3_cfg) == [
        "model.boundary_ratio",
        "model.offload_mode",
        "model.trainable_transformers",
    ]


def test_nextstep_freeze_vae_is_scoped_to_nextstep_family() -> None:
    """freeze_vae is a NextStep model key, not a global model knob."""
    from vrl.config.unknown_keys import find_unknown_keys

    nextstep_cfg = OmegaConf.create(
        {
            "model": {
                "family": "nextstep_1",
                "path": "stepfun-ai/NextStep-1.1",
                "vae_path": "stepfun-ai/NextStep-1-f8ch16-Tokenizer",
                "freeze_vae": True,
            },
        },
    )
    assert find_unknown_keys(nextstep_cfg) == []

    sd3_cfg = OmegaConf.create(
        {
            "model": {
                "family": "sd3_5",
                "path": "stabilityai/stable-diffusion-3.5-medium",
                "freeze_vae": True,
            },
        },
    )
    assert find_unknown_keys(sd3_cfg) == ["model.freeze_vae"]


def test_unknown_wan_offload_mode_raises() -> None:
    """Wan offload mode is a typed three-state enum, not two independent bools."""
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                "offload_mode": "stream",
            },
        },
    )
    with pytest.raises(ValueError, match=r"unknown model\.offload_mode"):
        parse_config(cfg)


# ── distributed.rollout knobs ─────────────────────────────────────────────────


def test_unknown_chunk_placement_strategy_raises() -> None:
    """A typo chunk placement strategy is rejected at parse time, not at launch."""
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"chunk_placement_strategy": "work_stealing"}},
    )
    with pytest.raises(
        ValueError,
        match=r"unknown distributed\.rollout\.chunk_placement_strategy",
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
    """distributed.rollout.* per-worker runtime keys are known to the walker.

    Colocation lives at distributed.resources.rollout.gpu_pool=trainer, so the
    removed distributed.rollout.colocate block is correctly an unknown key."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "cpus_per_worker": 2.0,
                "max_inflight_chunks_per_worker": 2,
                "health_check_interval_s": 45.0,
                "health_check_timeout_s": 30.0,
                "health_check_first_wait_s": 5.0,
                "chunk_placement_strategy": "dynamic",
                "sync_trainable_state": False,
                "pipelined": True,
            }
        }
    )
    unknown = find_unknown_keys(cfg)
    assert not [k for k in unknown if k.startswith("distributed.rollout")]


def test_rollout_health_check_defaults_and_accepts_override() -> None:
    default = parse_config(_minimal_grpo_cfg(distributed={"rollout": {}}))
    assert default.distributed.rollout.health_check_interval_s == 30.0
    assert default.distributed.rollout.health_check_timeout_s == 30.0
    assert default.distributed.rollout.health_check_first_wait_s == 0.0

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": 12.5,
                "health_check_timeout_s": 7.5,
                "health_check_first_wait_s": 2.5,
            }
        },
    )
    rollout = parse_config(cfg).distributed.rollout
    assert rollout.health_check_interval_s == 12.5
    assert rollout.health_check_timeout_s == 7.5
    assert rollout.health_check_first_wait_s == 2.5


@pytest.mark.parametrize("interval_s", [0.0, -1.0])
def test_rollout_health_check_interval_le_zero_disables_probe(interval_s: float) -> None:
    """A non-positive interval turns the probe off; the timeout is then unchecked."""

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": interval_s,
                "health_check_timeout_s": 0.0,
            }
        },
    )

    assert parse_config(cfg).distributed.rollout.health_check_interval_s == interval_s


@pytest.mark.parametrize("interval_s", [float("inf"), float("-inf"), float("nan")])
def test_rollout_health_check_interval_must_be_finite(interval_s: float) -> None:
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"health_check_interval_s": interval_s}},
    )

    with pytest.raises(ValueError, match=r"health_check_interval_s must be finite"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "timeout_s",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_rollout_health_check_timeout_must_be_finite_and_positive_when_enabled(
    timeout_s: float,
) -> None:
    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": 30.0,
                "health_check_timeout_s": timeout_s,
            }
        },
    )

    with pytest.raises(ValueError, match=r"health_check_timeout_s must be finite and > 0"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "first_wait_s",
    [-1.0, float("inf"), float("-inf"), float("nan")],
)
def test_rollout_health_check_first_wait_must_be_finite_and_non_negative(
    first_wait_s: float,
) -> None:
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"health_check_first_wait_s": first_wait_s}},
    )

    with pytest.raises(ValueError, match=r"health_check_first_wait_s must be finite and >= 0"):
        parse_config(cfg)


def test_removed_rollout_memory_fraction_is_unknown() -> None:
    """The removed bounded-resident input is absent from the key registry."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "resources": {
                "rollout": {
                    "gpu_pool": "trainer",
                    "memory_fraction": 0.4,
                },
            },
        },
    )

    assert "distributed.resources.rollout.memory_fraction" in find_unknown_keys(cfg)


@pytest.mark.parametrize("removed_key", ["placement_strategy", "max_inflight_batches"])
def test_removed_reward_pool_runtime_keys_are_unknown(removed_key: str) -> None:
    """Ray reward actor-pool knobs are no longer registered config keys."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={"reward": {"cpus_per_worker": 2.0, removed_key: "legacy"}},
    )
    unknown = find_unknown_keys(cfg)
    assert f"distributed.reward.{removed_key}" in unknown
    assert "distributed.reward.cpus_per_worker" not in unknown


def test_reward_resident_overlap_is_not_a_resource_key() -> None:
    """Same-GPU reward/rollout residency is not a public resource topology knob."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "resources": {
                "reward": {
                    "resident_overlap": True,
                },
            },
        },
    )

    assert "distributed.resources.reward.resident_overlap" in find_unknown_keys(cfg)


# ── Data loader discriminator ─────────────────────────────────────────────────


@pytest.mark.parametrize("loader", _literal_args(DataConfig.model_fields["loader"].annotation))
def test_valid_data_loaders_are_accepted(loader: str) -> None:
    """Every loader in the DataConfig.loader Literal allow-list is accepted; the
    per-loader construction branches below stay as real behavior coverage."""
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


@pytest.mark.parametrize(
    ("loader", "fmt", "message"),
    [
        (
            "prompt_manifest",
            "image_caption_jsonl",
            r"requires.*prompt_image_manifest",
        ),
        (
            "prompt_image_manifest",
            "text",
            r"requires.*image_caption_jsonl",
        ),
    ],
)
def test_explicit_data_loader_rejects_preprocessing_format_conflict(
    loader: str,
    fmt: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DataConfig(
            loader=loader,
            manifest="train.jsonl",
            eval_manifest="eval.jsonl",
            preprocessing={
                "format": fmt,
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


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


def test_zero_weight_observation_component_is_valid() -> None:
    """Checks zero weight keeps a component valid for observation-only scoring."""
    cfg = RewardConfig.model_validate({"components": {"kling_video_reward": 0.0}, "kwargs": {}})
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
        RewardConfig.model_validate({"components": {"aesthetic": "heavy"}, "kwargs": {}})


def test_reward_http_inference_config_is_validated_inside_open_component_kwargs() -> None:
    """Transport config stays typed even though reward-specific kwargs are open."""

    cfg = RewardConfig.model_validate(
        {
            "components": {"videoscore2": 1.0},
            "kwargs": {
                "videoscore2": {
                    "inference": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                    },
                },
            },
        },
    )

    assert cfg.kwargs["videoscore2"]["inference"]["kind"] == "http"


def test_reward_http_inference_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match=r"unsupported .* keys"):
        RewardConfig.model_validate(
            {
                "components": {"videoscore2": 1.0},
                "kwargs": {
                    "videoscore2": {
                        "inference": {
                            "kind": "http",
                            "endpoint": "http://reward:8300",
                            "expected_model": "videoscore2-v1",
                            "service_url": "http://legacy",
                        },
                    },
                },
            },
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


def test_token_grpo_multisegment_requires_explicit_janus_r1_family() -> None:
    """The algorithm cannot silently turn base Janus into the R1 protocol."""
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
        }
    )
    with pytest.raises(ValueError, match="janus_pro_r1"):
        parse_config(cfg)


def test_token_grpo_multisegment_final_image_policy_single_source() -> None:
    """final_image_policy is owned by the rollout section."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo_multisegment"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "janus_pro_r1"},
            "rollout": {"final_image_policy": "always_generate"},
        }
    )
    assert parse_config(cfg).rollout.final_image_policy == "always_generate"


def test_janus_r1_family_requires_multisegment_algorithm() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "janus_r1"},
            "rollout": {},
        }
    )

    with pytest.raises(ValueError, match="token_grpo_multisegment"):
        parse_config(cfg)


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
                        "sleep_offload": True,
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
                        "sleep_offload": True,
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


def test_production_schema_defaults_known_gate_to_disabled() -> None:
    cfg = OmegaConf.create({"production": {}})

    parsed = parse_config(cfg)

    assert parsed.production is not None
    assert parsed.production.kling_video_reward.enabled is False


def test_production_schema_accepts_enabled_gate() -> None:
    cfg = OmegaConf.create(
        {"production": {"kling_video_reward": {"enabled": True}}},
    )

    parsed = parse_config(cfg)

    assert parsed.production is not None
    assert parsed.production.kling_video_reward.enabled is True


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
                        "sleep_offload": True,
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
                        "sleep_offload": True,
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
    cfg.rollout.sde.type = "flow_grpo"
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
            "rollout": {"sde": {"type": "flow_grpo"}},
            "reward": {
                "components": {"custom_reward": 1.0},
                "kwargs": {"custom_reward": {"model_repo": "org/model"}},
            },
        }
    )
    parsed = parse_config(cfg)
    assert parsed.reward.components["custom_reward"] == 1.0


# ── algorithm.sft_weight x data.sft_latents (regularizer data channel) ───────


def test_sft_weight_without_latents_shard_raises() -> None:
    """A weight without its data channel would be a silent no-op knob."""
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": 0.1})
    with pytest.raises(ValueError, match=r"data\.sft_latents"):
        parse_config(cfg)


def test_sft_weight_with_latents_shard_parses() -> None:
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": 0.1})
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    parse_config(cfg)


def test_diffusion_dpo_sft_weight_does_not_require_online_latents_shard() -> None:
    cfg = _minimal_grpo_cfg(
        algorithm={"kind": "diffusion_dpo", "sft_weight": 0.1},
    )
    del cfg.rollout
    parsed = parse_config(cfg)
    assert parsed.algorithm.sft_weight == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("section", "payload", "field"),
    [
        ("actor", {"ema": {"enable": True}}, "actor.ema"),
        ("actor", {"ppo_epochs": 2}, "actor.ppo_epochs"),
        ("trainer", {"total_epochs": 10}, "trainer.total_epochs"),
        ("trainer", {"precision_drift_guard": {"mode": "warn"}}, "trainer.precision_drift_guard"),
        ("rollout", {"prompts_per_batch": 1}, "rollout.prompts_per_batch"),
    ],
)
def test_diffusion_dpo_rejects_online_only_config_fields(
    section: str,
    payload: dict,
    field: str,
) -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "diffusion_dpo"},
            section: payload,
        },
    )

    with pytest.raises(ValueError, match=rf"{field}"):
        parse_config(cfg)


def test_diffusion_dpo_accepts_its_resume_and_optimizer_surface() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "diffusion_dpo"},
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "bf16"},
            },
            "actor": {
                "optim": {"lr": 1e-8},
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "max_norm": 1.0,
                "prediction_type": "flow_matching",
                "scale_lr": False,
                "train_batch_size": 1,
                "use_adafactor": False,
            },
            "trainer": {
                "checkpointing_steps": 10,
                "entrypoint": "pkg.module:train",
                "log_interval": 1,
                "max_train_steps": 20,
                "output_dir": "outputs/dpo",
                "resume_from": "",
                "resume_strict": True,
            },
        },
    )

    parsed = parse_config(cfg)

    assert parsed.algorithm.kind == "diffusion_dpo"


def test_latents_shard_without_weight_is_inert_and_allowed() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    parse_config(cfg)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_sft_weight_must_be_finite_and_nonnegative(value: float) -> None:
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": value})
    with pytest.raises(ValueError, match="finite number >= 0"):
        parse_config(cfg)


def test_sft_weight_rejects_non_diffusion_grpo_kind() -> None:
    cfg = _minimal_grpo_cfg(
        algorithm={"kind": "token_grpo", "sft_weight": 0.1},
    )
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    with pytest.raises(ValueError, match="only for diffusion"):
        parse_config(cfg)
