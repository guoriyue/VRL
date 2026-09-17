"""The algorithm section: kind-scoped keys, KL reward coefficients, SDE types,
and the sft_weight x sft_latents channel."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tests.config.helpers import literal_args, minimal_grpo_cfg, unknown_keys
from vrl.config.schema import (
    AlgorithmConfig,
    parse_config,
)

# ── Algorithm kind discriminator ──────────────────────────────────────────────


def test_unknown_algorithm_keys_are_rejected_together() -> None:
    """Removed keys, typos, and never-seen keys: one error naming all of them."""

    cfg = OmegaConf.create(
        {"algorithm": {"kind": "grpo", "adv_estimator": "dpo", "future_field": True}}
    )
    assert unknown_keys(cfg) == ["algorithm.adv_estimator", "algorithm.future_field"]


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("grpo", "flow_kl_use_dt", True),
        ("dance_grpo", "sft_weight", 0.1),
        ("flow_dppo", "add_kl_coefficient", False),
        ("grpo_guard", "clip_ratio", 0.2),
        ("diffusion_dpo", "beta", 5000.0),
        ("diffusion_nft", "nft_beta", 0.1),
        ("v_grpo", "adv_soft_clip", 2.0),
    ],
)
def test_algorithm_keys_derive_from_selected_runtime_config(
    kind: str,
    field: str,
    value: object,
) -> None:
    """The section accepts exactly the selected dataclass's fields, typed, and
    hands back the built dataclass (cross-field rules are the root's job)."""

    section = AlgorithmConfig.model_validate({"kind": kind, field: value})
    assert section.kind == kind
    assert getattr(section.hyperparameters, field) == value


@pytest.mark.parametrize(
    ("kind", "foreign_field"),
    [("grpo", "beta"), ("diffusion_dpo", "clip_ratio")],
)
def test_algorithm_keys_are_scoped_to_selected_kind(kind: str, foreign_field: str) -> None:
    with pytest.raises(ValueError, match=rf"unknown algorithm\.{foreign_field}"):
        AlgorithmConfig.model_validate({"kind": kind, foreign_field: 1})


def test_algorithm_dispatch_covers_schema_kind_vocabulary() -> None:
    from vrl.config.algorithm import algorithm_config_class

    kinds = literal_args(AlgorithmConfig.model_fields["kind"].annotation)
    assert kinds
    assert all(algorithm_config_class(kind) for kind in kinds)


def test_positive_kl_reward_coef_is_accepted_for_diffusion_rollouts() -> None:
    cfg = minimal_grpo_cfg()
    cfg.algorithm.kl_reward_coef = 0.25

    assert parse_config(cfg).algorithm.kl_reward_coef == 0.25


def test_positive_kl_reward_coef_rejects_trajectories_without_step_kl() -> None:
    cfg = OmegaConf.create(
        {"algorithm": {"kind": "diffusion_dpo", "kl_reward_coef": 0.25}},
    )

    with pytest.raises(
        ValueError,
        match=r"algorithm\.kl_reward_coef > 0 requires a diffusion rollout trajectory",
    ):
        parse_config(cfg)


@pytest.mark.parametrize("value", [-0.1, float("nan")])
def test_kl_reward_coef_rejects_invalid_public_values(value: object) -> None:
    cfg = minimal_grpo_cfg()
    cfg.algorithm.kl_reward_coef = value

    with pytest.raises(
        ValueError,
        match=r"algorithm\.kl_reward_coef must be a finite number >= 0",
    ):
        parse_config(cfg)


def test_grpo_requires_valid_sde_type() -> None:
    cfg = minimal_grpo_cfg()
    cfg.rollout.sde.type = "euler"
    with pytest.raises(ValueError, match=r"unknown rollout\.sde\.type"):
        parse_config(cfg)


def test_grpo_accepts_cps_sde_type() -> None:
    """``cps`` is a valid ``rollout.sde.type`` for GRPO."""
    cfg = minimal_grpo_cfg()
    cfg.rollout.sde.type = "cps"
    parsed = parse_config(cfg)
    assert parsed.algorithm.kind == "grpo"


# ── algorithm.sft_weight x data.sft_latents (regularizer data channel) ───────


def test_sft_weight_without_latents_shard_raises() -> None:
    """A weight without its data channel would be a silent no-op knob."""
    cfg = minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": 0.1})
    with pytest.raises(ValueError, match=r"data\.sft_latents"):
        parse_config(cfg)


def test_sft_weight_with_latents_shard_parses() -> None:
    cfg = minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": 0.1})
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    parse_config(cfg)


def test_diffusion_dpo_sft_weight_does_not_require_online_latents_shard() -> None:
    cfg = minimal_grpo_cfg(
        algorithm={"kind": "diffusion_dpo", "sft_weight": 0.1},
    )
    del cfg.rollout
    parsed = parse_config(cfg)
    assert parsed.algorithm.hyperparameters.sft_weight == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("section", "payload", "field"),
    [
        ("actor", {"ema": {"enable": True}}, "actor.ema"),
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
    cfg = minimal_grpo_cfg()
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    parse_config(cfg)


@pytest.mark.parametrize("value", [-0.1, float("nan")])
def test_sft_weight_must_be_finite_and_nonnegative(value: float) -> None:
    cfg = minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": value})
    with pytest.raises(ValueError, match="finite number >= 0"):
        parse_config(cfg)


def test_sft_weight_rejects_non_diffusion_grpo_kind() -> None:
    cfg = minimal_grpo_cfg(
        algorithm={"kind": "flash_grpo", "sft_weight": 0.1},
    )
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    with pytest.raises(ValueError, match=r"not supported by algorithm\.kind='flash_grpo'"):
        parse_config(cfg)
