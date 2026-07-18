"""Offline Wan DPO config must expose only knobs the trainer consumes."""

from __future__ import annotations

import pytest

from vrl.algorithms.dpo import DiffusionDPOConfig
from vrl.config.loading import load_config
from vrl.scripts.diffusion.wan_2_1.train_dpo import (
    _build_offline_dpo_trainer_config,
    train_wan_2_1_dpo,
)


def _resolved_trainer_config(overrides: list[str] | None = None):
    cfg = load_config(
        "experiment/diffusion/wan_2_1/offline_dpo_pickapic",
        overrides=overrides,
    )
    return _build_offline_dpo_trainer_config(
        cfg,
        DiffusionDPOConfig(beta=123.0, sft_weight=0.25),
        train_batch_size=int(cfg.actor.train_batch_size),
        gradient_accumulation_steps=int(cfg.actor.gradient_accumulation_steps),
    )


def test_offline_dpo_bridges_every_supported_adam_knob() -> None:
    resolved = _resolved_trainer_config(
        [
            "actor.optim.lr=0.01",
            "actor.optim.adam_beta1=0.7",
            "actor.optim.adam_beta2=0.8",
            "actor.optim.weight_decay=0.03",
            "actor.optim.eps=1e-6",
            "actor.scale_lr=true",
            "actor.train_batch_size=2",
            "actor.gradient_accumulation_steps=3",
        ],
    )

    assert resolved.beta == pytest.approx(123.0)
    assert resolved.sft_weight == pytest.approx(0.25)
    assert resolved.lr == pytest.approx(0.06)
    assert resolved.adam_beta1 == pytest.approx(0.7)
    assert resolved.adam_beta2 == pytest.approx(0.8)
    assert resolved.adam_weight_decay == pytest.approx(0.03)
    assert resolved.adam_epsilon == pytest.approx(1e-6)


def test_offline_dpo_uses_typed_optimizer_defaults_when_keys_are_absent() -> None:
    resolved = _resolved_trainer_config(["actor.scale_lr=false"])

    assert resolved.adam_beta1 == pytest.approx(0.9)
    assert resolved.adam_beta2 == pytest.approx(0.999)
    assert resolved.adam_weight_decay == pytest.approx(1e-4)
    assert resolved.adam_epsilon == pytest.approx(1e-8)


def test_offline_dpo_rejects_unsupported_8bit_optimizer() -> None:
    with pytest.raises(ValueError, match=r"optim_8bit=true is not supported"):
        _resolved_trainer_config(["actor.optim.optim_8bit=true"])


@pytest.mark.parametrize(
    ("key", "value"),
    [("adam_beta1", "0.8"), ("adam_beta2", "0.9"), ("eps", "1e-6")],
)
def test_offline_dpo_rejects_explicit_adamw_only_knobs_for_adafactor(
    key: str,
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"use_adafactor=true does not consume AdamW-only key\(s\): actor\.optim\.{key}",
    ):
        _resolved_trainer_config(
            ["actor.use_adafactor=true", f"actor.optim.{key}={value}"],
        )


def test_offline_dpo_adafactor_keeps_shared_optimizer_knobs() -> None:
    resolved = _resolved_trainer_config(
        [
            "actor.use_adafactor=true",
            "actor.scale_lr=false",
            "actor.optim.lr=2e-7",
            "actor.optim.weight_decay=0.03",
        ],
    )

    assert resolved.use_adafactor is True
    assert resolved.lr == pytest.approx(2e-7)
    assert resolved.adam_weight_decay == pytest.approx(0.03)


def test_offline_dpo_recipe_does_not_inherit_online_only_state() -> None:
    cfg = load_config("experiment/diffusion/wan_2_1/offline_dpo_pickapic")

    assert "ema" not in cfg.actor
    assert "drop_zero_advantage" not in cfg.actor
    assert "timestep_fraction" not in cfg.actor
    assert "total_epochs" not in cfg.trainer
    assert "rollout_orchestration" not in cfg.trainer
    assert "rollout" not in cfg


def test_offline_dpo_builds_its_full_model_through_the_family_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config("experiment/diffusion/wan_2_1/offline_dpo_pickapic")
    captured: dict[str, object] = {}

    class _ReachedRegistryBoundary(RuntimeError):
        pass

    class _Entry:
        def resolve_model_build(
            self,
            received_cfg: object,
            device: object,
            *,
            for_rollout: bool,
            precision_role: str,
            parameter_dtype_override: object,
        ) -> object:
            captured.update(
                cfg=received_cfg,
                device=device,
                for_rollout=for_rollout,
                precision_role=precision_role,
                parameter_dtype_override=parameter_dtype_override,
            )
            raise _ReachedRegistryBoundary

    import vrl.families.registry as registry
    from vrl.scripts.diffusion.wan_2_1 import train_dpo

    def _entry_for(family: str) -> _Entry:
        captured["family"] = family
        return _Entry()

    monkeypatch.setattr(
        registry,
        "get_model_family_entry",
        _entry_for,
    )
    monkeypatch.setattr(train_dpo, "resolve_distributed_resources", lambda _cfg: object())
    monkeypatch.setattr(train_dpo, "format_distributed_resource_plan", lambda _plan: "")
    monkeypatch.setattr(train_dpo, "trainer_torch_device", lambda _plan: "cpu")

    with pytest.raises(_ReachedRegistryBoundary):
        train_wan_2_1_dpo(cfg)

    assert captured["family"] == "wan"
    assert captured["cfg"] is cfg
    assert captured["for_rollout"] is True
    assert captured["precision_role"] == "training"
