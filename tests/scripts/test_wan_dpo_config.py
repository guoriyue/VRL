"""The Wan DPO entrypoint resolves its model through the family registry and
refuses non-T2V families before any runtime side effect."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.config.loading import load_config
from vrl.scripts.families.wan_2_1.train_dpo import train_wan_2_1_dpo


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
        family = "wan_2_1"

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

    import vrl.models.families.registry as registry
    import vrl.ray.resources as ray_resources

    def _entry_for(family: str) -> _Entry:
        captured["family"] = family
        return _Entry()

    monkeypatch.setattr(
        registry,
        "get_model_family_entry",
        _entry_for,
    )
    monkeypatch.setattr(
        ray_resources.ResolvedDistributedResources,
        "from_root",
        classmethod(lambda _cls, _cfg, **_kwargs: SimpleNamespace(trainer_torch_device="cpu")),
    )
    monkeypatch.setattr(ray_resources, "format_distributed_resource_plan", lambda _plan: "")

    with pytest.raises(_ReachedRegistryBoundary):
        train_wan_2_1_dpo(cfg)

    assert captured["family"] == "wan_2_1"
    assert captured["for_rollout"] is True
    assert captured["precision_role"] == "training"


@pytest.mark.parametrize("family", ["wan_2_1_i2v", "sd3_5"])
def test_offline_dpo_rejects_non_t2v_wan_family_before_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    cfg = load_config("experiment/wan_2_1/offline_dpo_pickapic")
    cfg.model.family = family
    if family == "sd3_5":
        del cfg.sampling.num_frames
    calls: list[str] = []

    import vrl.trainers.checkpointing as checkpointing

    def unexpected_checkpoint(*_args: object, **_kwargs: object) -> object:
        calls.append("checkpoint")
        raise AssertionError("checkpoint loading must not run before the Wan DPO family guard")

    monkeypatch.setattr(
        checkpointing,
        "load_training_checkpoint_for_resume",
        unexpected_checkpoint,
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
        family = "wan_2_1"

        def resolve_model_build(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(rollout=object())

        def build_rollout(self, build: object) -> _Bundle:
            return bundle

    import vrl.models.checkpoint_identity as checkpoint_identity
    import vrl.models.families.registry as registry
    import vrl.ray.resources as ray_resources

    monkeypatch.setattr(registry, "get_model_family_entry", lambda _family: _Entry())
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda _build: {"schema": "test"},
    )
    monkeypatch.setattr(
        ray_resources.ResolvedDistributedResources,
        "from_root",
        classmethod(lambda _cls, _cfg, **_kwargs: SimpleNamespace(trainer_torch_device="cpu")),
    )
    monkeypatch.setattr(ray_resources, "format_distributed_resource_plan", lambda _plan: "")

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
