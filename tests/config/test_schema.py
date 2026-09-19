"""Tests for the typed config schema boundary (vrl/config/schema.py)."""

from __future__ import annotations

import re

import pytest
from omegaconf import OmegaConf

from tests.config.helpers import minimal_grpo_cfg
from vrl.config.schema import (
    RewardConfig,
    RolloutRuntimeSection,
    parse_config,
)


@pytest.mark.parametrize("family", ["cosmos-predict2", "cosmos-predict2.5"])
def test_cosmos_video_accepts_frame_shared_adaln(family: str) -> None:
    cfg = parse_config(
        minimal_grpo_cfg(model={"family": family, "frame_shared_adaln": True, "use_lora": True}),
    )
    assert cfg.model.model_dump()["frame_shared_adaln"] is True


def test_frame_shared_adaln_rejects_models_without_frame_conditioning() -> None:
    cfg = minimal_grpo_cfg(model={"family": "sd3_5", "frame_shared_adaln": True})
    with pytest.raises(ValueError, match=r"unknown model\.frame_shared_adaln"):
        parse_config(cfg)


def test_fused_lora_branch_requires_enabled_adapters() -> None:
    cfg = minimal_grpo_cfg(model={"family": "sd3_5", "fused_lora_branch": True})
    with pytest.raises(ValueError, match=r"model\.fused_lora_branch requires model\.use_lora"):
        parse_config(cfg)

    cfg.model.use_lora = True
    assert parse_config(cfg).model.fused_lora_branch is True


@pytest.mark.parametrize("kind", ["diffusion_nft", "v_grpo"])
def test_previous_policy_requirements_belong_to_algorithm(kind: str) -> None:
    # Whether the policy is a LoRA adapter or the whole transformer is the
    # model's business: both admit the previous-policy objectives.
    for use_lora in (False, True):
        parse_config(
            minimal_grpo_cfg(
                model={"family": "cosmos-predict2.5", "use_lora": use_lora},
                algorithm={"kind": kind},
            ),
        )
    # Any full-sequence family with a replay recipe qualifies; a chunk-autoregressive
    # policy has no full-sequence replay forward to evaluate the clean latent through.
    parse_config(
        minimal_grpo_cfg(
            model={"family": "sana", "use_lora": True},
            algorithm={"kind": kind},
        ),
    )
    with pytest.raises(ValueError, match="full-sequence denoise replay forward"):
        parse_config(
            minimal_grpo_cfg(
                model={"family": "causvid", "use_lora": True},
                algorithm={"kind": kind},
            ),
        )


def test_previous_adapter_is_not_a_user_model_setting() -> None:
    with pytest.raises(ValueError, match=r"unknown model.lora.previous_adapter"):
        parse_config(
            minimal_grpo_cfg(
                model={"family": "flux", "use_lora": True, "lora": {"previous_adapter": True}},
            ),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("algorithm.kind", "qpo"),
        ("rollout.denoise_mode", "bogus"),
        ("distributed.training.strategy", "deepspeed"),
        ("data.loader", "s3_loader"),
    ],
)
def test_unknown_enum_value_is_rejected_at_parse_by_its_dotted_path(path: str, value: str) -> None:
    """A typo in any Literal-typed field fails at parse time naming the field, not at launch."""
    cfg = minimal_grpo_cfg()
    OmegaConf.update(cfg, path, value, force_add=True)
    with pytest.raises(ValueError, match=rf"unknown {re.escape(path)}"):
        parse_config(cfg)


# ── distributed.rollout knobs ─────────────────────────────────────────────────


def test_rollout_health_check_defaults_and_accepts_override() -> None:
    default = parse_config(minimal_grpo_cfg(distributed={"rollout": {}}))
    assert default.distributed.rollout.health_check_interval_s == 30.0
    assert default.distributed.rollout.health_check_timeout_s == 30.0
    assert default.distributed.rollout.health_check_first_wait_s == 0.0
    assert default.distributed.rollout.worker_rpc_timeout_s == 600.0
    assert default.distributed.rollout.generation_stall_timeout_s == 3600.0

    cfg = minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": 12.5,
                "health_check_timeout_s": 7.5,
                "health_check_first_wait_s": 2.5,
                "worker_rpc_timeout_s": 3600.0,
                "generation_stall_timeout_s": 1200.0,
            }
        },
    )
    rollout = parse_config(cfg).distributed.rollout
    assert rollout.health_check_interval_s == 12.5
    assert rollout.health_check_timeout_s == 7.5
    assert rollout.health_check_first_wait_s == 2.5
    assert rollout.worker_rpc_timeout_s == 3600.0
    assert rollout.generation_stall_timeout_s == 1200.0


def test_rollout_worker_section_mirrors_worker_runtime_config() -> None:
    """RolloutRuntimeSection (pydantic lint boundary) and RolloutWorkerConfig (the
    frozen runtime projection composed into RayGenerationConfig) must stay
    field-for-field identical. ``from_public_section`` builds the dataclass via
    ``cls(**section.model_dump())``, so a field on one but not the other silently
    breaks at runtime (TypeError / unfilled required field) instead of at parse.

    The two types are deliberately NOT merged (the pydantic schema is a lint-only
    boundary; the dataclass is the real runtime consumer), so this parity test is
    the guard against drift. Public defaults remain only on the section; the
    runtime dataclass has no fallback literals.
    """
    import dataclasses

    from vrl.generation.ray.config import RolloutWorkerConfig

    section_fields = set(RolloutRuntimeSection.model_fields)
    config_fields = {f.name for f in dataclasses.fields(RolloutWorkerConfig)}
    assert section_fields == config_fields
    assert "health_check_first_wait_s" in section_fields
    assert "worker_rpc_timeout_s" in section_fields
    assert "generation_stall_timeout_s" in section_fields

    # Per-field default parity: the section's declared defaults must survive the
    # projection unchanged (from_public_section adds no fallbacks or overrides), so
    # the section stays the single home of the default literals.
    projected = RolloutWorkerConfig.from_public_section(RolloutRuntimeSection())
    for name in section_fields:
        assert getattr(projected, name) == RolloutRuntimeSection.model_fields[name].default


def test_rollout_health_check_interval_le_zero_disables_probe() -> None:
    """A non-positive interval turns the probe off; the timeout is then unchecked."""

    cfg = minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": 0.0,
                "health_check_timeout_s": 0.0,
            }
        },
    )

    assert parse_config(cfg).distributed.rollout.health_check_interval_s == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("health_check_interval_s", float("nan"), r"health_check_interval_s must be finite"),
        ("health_check_timeout_s", 0.0, r"health_check_timeout_s must be finite and > 0"),
        ("health_check_first_wait_s", -1.0, r"health_check_first_wait_s must be finite and >= 0"),
        ("worker_rpc_timeout_s", float("inf"), r"worker_rpc_timeout_s must be finite and > 0"),
        ("generation_stall_timeout_s", 0.0, r"generation_stall_timeout_s must be finite and > 0"),
    ],
)
def test_rollout_worker_timeouts_must_be_finite_and_in_range(
    field: str,
    value: float,
    message: str,
) -> None:
    cfg = minimal_grpo_cfg(distributed={"rollout": {field: value}})

    with pytest.raises(ValueError, match=message):
        parse_config(cfg)


# ── Reward weight validation ──────────────────────────────────────────────────


def test_zero_weight_observation_component_is_valid() -> None:
    """Checks zero weight keeps a component valid for observation-only scoring."""
    cfg = RewardConfig.model_validate({"components": {"kling_video_reward": 0.0}, "kwargs": {}})
    assert cfg.components["kling_video_reward"] == 0.0


def test_non_numeric_reward_weight_raises() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        RewardConfig.model_validate({"components": {"aesthetic": "heavy"}, "kwargs": {}})


def test_reward_http_inference_config_is_typed_beside_open_component_kwargs() -> None:
    """Transport config is typed even though reward-specific kwargs are open."""

    cfg = RewardConfig.model_validate(
        {
            "components": {"videoscore2": 1.0},
            "kwargs": {"videoscore2": {"artifact_dir": "/shared/artifacts"}},
            "inference": {
                "videoscore2": {
                    "kind": "http",
                    "endpoint": "http://reward:8300",
                    "expected_model": "videoscore2-v1",
                },
            },
        },
    )

    assert cfg.inference["videoscore2"].kind == "http"


def test_reward_inference_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match=r"reward\.inference\..* fields: .*unknown="):
        RewardConfig.model_validate(
            {
                "components": {"videoscore2": 1.0},
                "inference": {
                    "videoscore2": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                        "service_url": "http://legacy",
                    },
                },
            },
        )


def test_reward_inference_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="unknown component"):
        RewardConfig.model_validate(
            {
                "components": {"videoscore2": 1.0},
                "inference": {
                    "typo_component": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                    },
                },
            },
        )


def test_production_video_reward_structural_rules() -> None:
    """Production validates the named reward and task, not transport encoding."""
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
                        "worker_config": {},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        }
    )
    from vrl.rewards.functions.registry import get_reward

    root = parse_config(cfg)
    get_reward("kling_video_reward").production.require(
        "kling_video_reward",
        root.reward.kwargs["kling_video_reward"],
        task_type=str(root.data.task_type),
    )


def test_production_gate_defaults_to_disabled_and_accepts_enabled() -> None:
    disabled = parse_config(OmegaConf.create({"production": {}}))
    assert disabled.production.kling_video_reward.enabled is False

    enabled = parse_config(
        OmegaConf.create({"production": {"kling_video_reward": {"enabled": True}}}),
    )
    assert enabled.production.kling_video_reward.enabled is True


# ── Missing field mapping (??? → ValueError) ──────────────────────────────────


def test_missing_mandatory_value_produces_repo_standard_message() -> None:
    """An OmegaConf ``???`` marker surfaces as the repo-standard 'config missing required field'
    error.
    """
    cfg = minimal_grpo_cfg()
    cfg.rollout.sde.type = "flow_grpo"
    # Inject an OmegaConf mandatory-missing marker
    OmegaConf.update(cfg, "algorithm.kind", "???")
    with pytest.raises(ValueError, match="config missing required field"):
        parse_config(cfg)


# ── extra="ignore" migration policy ──────────────────────────────────────────


def test_unknown_top_level_sections_are_rejected() -> None:
    """parse_config is the one gate: a section no consumer reads fails loud."""
    cfg = minimal_grpo_cfg()
    OmegaConf.update(cfg, "some_future_section.foo", "bar")
    with pytest.raises(ValueError, match=r"unknown some_future_section"):
        parse_config(cfg)
