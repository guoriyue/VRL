"""Wan DPO checkpoint identity must gate and follow the whole training run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.utils.data

import vrl.families.registry as registry
import vrl.models.checkpoint_identity as checkpoint_identity
import vrl.ray.resources as ray_resources
import vrl.trainers.activation_checkpointing as activation_checkpointing
import vrl.trainers.checkpointing as checkpointing
import vrl.trainers.data as trainer_data
import vrl.trainers.offline as offline
from vrl.config.loading import load_config
from vrl.scripts.families.wan_2_1.train_dpo import train_wan_2_1_dpo


def _install_pre_model_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entry: object,
    checkpoint: object,
) -> None:
    monkeypatch.setattr(registry, "get_model_family_entry", lambda _family: entry)
    monkeypatch.setattr(ray_resources, "resolve_distributed_resources", lambda _cfg, **_kwargs: object())
    monkeypatch.setattr(ray_resources, "format_distributed_resource_plan", lambda _plan: "")
    monkeypatch.setattr(ray_resources, "trainer_torch_device", lambda _plan: "cpu")
    monkeypatch.setattr(
        checkpointing,
        "load_training_checkpoint_for_resume",
        lambda _resume: checkpoint,
    )


def test_matching_identity_gates_model_and_threads_restore_and_saves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = load_config(
        "experiment/wan_2_1/offline_dpo_pickapic",
        overrides=[
            "model.use_lora=false",
            "trainer.max_train_steps=1",
            "trainer.checkpointing_steps=1",
            "trainer.log_interval=1",
            f"trainer.output_dir={tmp_path}",
        ],
    )
    identity = {"schema": "test"}
    checkpoint = SimpleNamespace(
        checkpoint_dir=tmp_path / "resume",
        next_step=0,
        rng_state={},
    )
    build = object()
    transformer = object()
    events: list[str] = []

    class _Scheduler:
        config = SimpleNamespace(num_train_timesteps=1000)

        def set_timesteps(self, count: int, *, device: object) -> None:
            assert count == 1000
            assert str(device) == "cpu"

    bundle = SimpleNamespace(
        model=SimpleNamespace(transformer=transformer),
        raw_handle=SimpleNamespace(scheduler=_Scheduler()),
        trainable_modules={"transformer": transformer},
    )

    class _Entry:
        def resolve_model_build(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            events.append("resolve_build")
            return build

        def build_rollout(self, actual_build: object) -> object:
            assert actual_build is build
            events.append("build_model")
            return bundle

    class _Trainer:
        global_step = 1

        def step(self, batch: object) -> SimpleNamespace:
            assert batch == "batch"
            return SimpleNamespace(
                loss=0.0,
                raw_model_loss=0.0,
                raw_ref_loss=0.0,
                model_diff=0.0,
                ref_diff=0.0,
                implicit_acc=0.0,
                sft_loss=0.0,
                grad_norm=0.0,
            )

    _install_pre_model_fakes(
        monkeypatch,
        entry=_Entry(),
        checkpoint=checkpoint,
    )

    def _resolve_identity(actual_build: object) -> dict[str, str]:
        assert actual_build is build
        events.append("resolve_identity")
        return identity

    def _validate(
        actual_checkpoint: object,
        *,
        family: str,
        expected_model_identity: dict[str, object],
        strict: bool,
    ) -> None:
        assert actual_checkpoint is checkpoint
        assert family == "wan_2_1"
        assert expected_model_identity is identity
        assert strict is True
        events.append("validate")

    def _restore(actual_checkpoint: object, **kwargs: object) -> None:
        assert actual_checkpoint is checkpoint
        assert kwargs["expected_model_identity"] is identity
        events.append("restore")

    def _save(path: object, **kwargs: object) -> None:
        assert kwargs["model_identity"] is identity
        events.append(f"save:{path.name}")

    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        _resolve_identity,
    )
    monkeypatch.setattr(checkpointing, "validate_checkpoint_compatibility", _validate)
    monkeypatch.setattr(checkpointing, "restore_training_checkpoint", _restore)
    monkeypatch.setattr(checkpointing, "restore_rng_state", lambda _state: None)
    monkeypatch.setattr(checkpointing, "save_resolved_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(checkpointing, "prepare_metrics_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(checkpointing, "capture_rng_state", lambda: {})
    monkeypatch.setattr(checkpointing, "save_training_checkpoint", _save)
    monkeypatch.setattr(
        activation_checkpointing,
        "enable_transformer_gradient_checkpointing",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        trainer_data,
        "load_pickapic",
        lambda **_kwargs: events.append("load_dataset") or ["sample"],
    )
    monkeypatch.setattr(torch.utils.data, "DataLoader", lambda *_args, **_kwargs: ["batch"])
    monkeypatch.setattr(offline, "OfflineDPOTrainer", lambda **_kwargs: _Trainer())
    monkeypatch.setattr(
        "vrl.scripts.families.wan_2_1.train_dpo._build_encoders",
        lambda *_args, **_kwargs: (object(), object()),
    )

    train_wan_2_1_dpo(cfg)

    assert events == [
        "resolve_build",
        "resolve_identity",
        "validate",
        "build_model",
        "resolve_identity",
        "load_dataset",
        "restore",
        "save:checkpoint-1",
        "save:checkpoint-final",
    ]


def test_identity_mismatch_stops_before_model_and_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config("experiment/wan_2_1/offline_dpo_pickapic")
    checkpoint = object()
    build = object()
    identity = {"schema": "test"}
    events: list[str] = []

    class _Entry:
        def resolve_model_build(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            events.append("resolve_build")
            return build

        def build_rollout(self, actual_build: object) -> object:
            del actual_build
            events.append("build_model")
            raise AssertionError("model construction must not run after identity mismatch")

    _install_pre_model_fakes(
        monkeypatch,
        entry=_Entry(),
        checkpoint=checkpoint,
    )

    def _resolve_identity(actual_build: object) -> dict[str, str]:
        assert actual_build is build
        events.append("resolve_identity")
        return identity

    def _reject(
        actual_checkpoint: object,
        *,
        family: str,
        expected_model_identity: dict[str, object],
        strict: bool,
    ) -> None:
        assert actual_checkpoint is checkpoint
        assert family == "wan_2_1"
        assert expected_model_identity is identity
        assert strict is True
        events.append("validate")
        raise ValueError("checkpoint model identity mismatch")

    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        _resolve_identity,
    )
    monkeypatch.setattr(checkpointing, "validate_checkpoint_compatibility", _reject)
    monkeypatch.setattr(
        trainer_data,
        "load_pickapic",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dataset loading must not run after identity mismatch"),
        ),
    )

    with pytest.raises(ValueError, match="checkpoint model identity mismatch"):
        train_wan_2_1_dpo(cfg)

    assert events == ["resolve_build", "resolve_identity", "validate"]


def test_source_change_during_model_load_stops_before_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config("experiment/wan_2_1/offline_dpo_pickapic")
    checkpoint = object()
    build = object()
    identities = iter(({"source": "before"}, {"source": "after"}))
    events: list[str] = []

    class _Entry:
        def resolve_model_build(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            events.append("resolve_build")
            return build

        def build_rollout(self, actual_build: object) -> object:
            assert actual_build is build
            events.append("build_model")
            return object()

    _install_pre_model_fakes(
        monkeypatch,
        entry=_Entry(),
        checkpoint=checkpoint,
    )

    def _resolve_identity(actual_build: object) -> dict[str, str]:
        assert actual_build is build
        events.append("resolve_identity")
        return next(identities)

    def _validate(*args: object, **kwargs: object) -> None:
        del args, kwargs
        events.append("validate")

    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        _resolve_identity,
    )
    monkeypatch.setattr(checkpointing, "validate_checkpoint_compatibility", _validate)
    monkeypatch.setattr(
        trainer_data,
        "load_pickapic",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dataset loading must not run after checkpoint source changes"),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="model checkpoint source changed during Wan DPO bundle construction",
    ):
        train_wan_2_1_dpo(cfg)

    assert events == [
        "resolve_build",
        "resolve_identity",
        "validate",
        "build_model",
        "resolve_identity",
    ]
