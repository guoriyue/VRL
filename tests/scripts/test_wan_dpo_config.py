"""Offline Wan DPO config must expose only knobs the trainer consumes."""

from __future__ import annotations

import pytest

from vrl.algorithms.dpo import DiffusionDPOConfig
from vrl.config.builders import build_offline_dpo_trainer_config
from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import RootConfig
from vrl.scripts.families.wan_2_1.train_dpo import train_wan_2_1_dpo


def _resolved_trainer_config(overrides: list[str] | None = None):
    cfg = load_config(
        "experiment/wan_2_1/offline_dpo_pickapic",
        overrides=overrides,
    )
    return build_offline_dpo_trainer_config(
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


def test_offline_dpo_max_grad_norm_default_belongs_to_offline_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.trainers.offline import OfflineDPOTrainerConfig
    from vrl.trainers.online.config import TrainerConfig

    monkeypatch.setattr(
        TrainerConfig.__dataclass_fields__["max_norm"],
        "default",
        9.0,
    )

    resolved = _resolved_trainer_config()

    assert resolved.max_grad_norm == OfflineDPOTrainerConfig().max_grad_norm


def test_offline_dpo_bridges_explicit_max_grad_norm() -> None:
    resolved = _resolved_trainer_config(["actor.max_norm=0.25"])

    assert resolved.max_grad_norm == pytest.approx(0.25)


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
    cfg = load_config("experiment/wan_2_1/offline_dpo_pickapic")

    assert "ema" not in cfg.actor
    assert "drop_zero_advantage" not in cfg.actor
    assert "timestep_fraction" not in cfg.actor
    assert "total_epochs" not in cfg.trainer
    assert "rollout_orchestration" not in cfg.trainer
    assert "rollout" not in cfg


@pytest.mark.parametrize("family", ["wan_2_1", "wan"])
def test_offline_dpo_builds_its_full_model_through_the_family_registry(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    cfg = load_config("experiment/wan_2_1/offline_dpo_pickapic")
    cfg.model.family = family
    captured: dict[str, object] = {}

    class _ReachedRegistryBoundary(RuntimeError):
        pass

    class _Entry:
        def resolve_model_build(
            self,
            root: object,
            device: object,
            *,
            precision: object,
            for_rollout: bool,
            precision_role: str,
        ) -> object:
            captured.update(
                root=root,
                device=device,
                precision=precision,
                for_rollout=for_rollout,
                precision_role=precision_role,
            )
            raise _ReachedRegistryBoundary

    import vrl.families.registry as registry
    import vrl.ray.resources as ray_resources

    def _entry_for(family: str) -> _Entry:
        captured["family"] = family
        return _Entry()

    monkeypatch.setattr(
        registry,
        "get_model_family_entry",
        _entry_for,
    )
    monkeypatch.setattr(ray_resources, "resolve_distributed_resources", lambda _cfg, **_kwargs: object())
    monkeypatch.setattr(ray_resources, "format_distributed_resource_plan", lambda _plan: "")
    monkeypatch.setattr(ray_resources, "trainer_torch_device", lambda _plan: "cpu")

    with pytest.raises(_ReachedRegistryBoundary):
        train_wan_2_1_dpo(cfg)

    assert captured["family"] == "wan_2_1"
    assert isinstance(captured["root"], RootConfig)
    assert isinstance(captured["precision"], PrecisionPolicy)
    assert captured["for_rollout"] is True
    assert captured["precision_role"] == "training"


@pytest.mark.parametrize("family", ["wan_2_1_i2v", "wan_i2v", "sd3_5"])
def test_offline_dpo_rejects_non_t2v_wan_family_before_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    cfg = load_config("experiment/wan_2_1/offline_dpo_pickapic")
    cfg.model.family = family
    if family == "sd3_5":
        del cfg.sampling.num_frames
    calls: list[str] = []

    import vrl.families.registry as registry
    import vrl.ray.resources as ray_resources
    import vrl.trainers.checkpointing as checkpointing

    def _unexpected(name: str):
        def fail(*_args: object, **_kwargs: object) -> object:
            calls.append(name)
            raise AssertionError(f"{name} must not run before the Wan DPO family guard")

        return fail

    monkeypatch.setattr(
        checkpointing,
        "load_training_checkpoint_for_resume",
        _unexpected("checkpoint"),
    )
    monkeypatch.setattr(
        ray_resources,
        "resolve_distributed_resources",
        _unexpected("resources"),
    )
    monkeypatch.setattr(
        registry,
        "get_model_family_entry",
        _unexpected("registry"),
    )

    with pytest.raises(
        ValueError,
        match=rf"Wan Diffusion-DPO requires model\.family='wan_2_1'.*got '{family}'",
    ):
        train_wan_2_1_dpo(cfg)

    assert calls == []


@pytest.mark.parametrize(
    ("checkpointing", "expected_mode"),
    [
        ('"off"', "off"),
        ("false", "off"),
        ('"full"', "full"),
        ("true", "full"),
        ('"selective"', "selective"),
    ],
)
def test_offline_dpo_uses_shared_gradient_checkpointing_policy(
    monkeypatch: pytest.MonkeyPatch,
    checkpointing: str,
    expected_mode: str,
) -> None:
    cfg = load_config(
        "experiment/wan_2_1/offline_dpo_pickapic",
        overrides=[f"actor.gradient_checkpointing={checkpointing}"],
    )
    calls: list[dict[str, object]] = []

    class _ReachedEncoderBoundary(RuntimeError):
        pass

    class _Transformer:
        def enable_gradient_checkpointing(self, **kwargs: object) -> None:
            calls.append(kwargs)

    transformer = _Transformer()

    class _Model:
        pass

    model = _Model()
    model.transformer = transformer

    class _Bundle:
        def __init__(self) -> None:
            self.raw_handle = object()
            self.trainable_modules = {"transformer": transformer}

    bundle = _Bundle()
    bundle.model = model

    class _Entry:
        def resolve_model_build(self, *args: object, **kwargs: object) -> object:
            return object()

        def build_rollout(self, build: object) -> _Bundle:
            return bundle

    import vrl.families.registry as registry
    import vrl.models.checkpoint_identity as checkpoint_identity
    import vrl.ray.resources as ray_resources

    monkeypatch.setattr(registry, "get_model_family_entry", lambda _family: _Entry())
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda _build: {"schema": "test"},
    )
    monkeypatch.setattr(ray_resources, "resolve_distributed_resources", lambda _cfg, **_kwargs: object())
    monkeypatch.setattr(ray_resources, "format_distributed_resource_plan", lambda _plan: "")
    monkeypatch.setattr(ray_resources, "trainer_torch_device", lambda _plan: "cpu")

    def _stop_at_encoder(*args: object, **kwargs: object) -> None:
        raise _ReachedEncoderBoundary

    monkeypatch.setattr(
        "vrl.scripts.families.wan_2_1.train_dpo._build_encoders",
        _stop_at_encoder,
    )

    with pytest.raises(_ReachedEncoderBoundary):
        train_wan_2_1_dpo(cfg)

    if expected_mode == "off":
        assert calls == []
    elif expected_mode == "full":
        assert calls == [{}]
    else:
        assert len(calls) == 1
        assert callable(calls[0]["gradient_checkpointing_func"])
