from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise.base import diffusers_pipeline_dtypes


def test_full_pipeline_propagates_revision_like_component_loader() -> None:
    build = ModelBuild(
        model_name_or_path="org/model",
        revision="immutable-revision",
        device="cpu",
        parameter_dtype=torch.float16,
        family="sd3_5",
        precision=RolePrecision("fp16", "tf32"),
        model_config={},
    )

    _, kwargs = diffusers_pipeline_dtypes(build, torch.float16)

    assert kwargs["revision"] == "immutable-revision"


def test_full_pipeline_propagates_local_files_only_like_component_loader() -> None:
    build = ModelBuild(
        model_name_or_path="org/model",
        device="cpu",
        parameter_dtype=torch.float16,
        family="sd3_5",
        revision="immutable-revision",
        precision=RolePrecision("fp16", "tf32"),
        model_config={"local_files_only": True},
    )

    _, kwargs = diffusers_pipeline_dtypes(build, torch.float16)

    assert kwargs["revision"] == "immutable-revision"
    assert kwargs["local_files_only"] is True


def test_full_pipeline_omits_absent_revision() -> None:
    build = ModelBuild(
        model_name_or_path="org/model",
        revision=None,
        device="cpu",
        parameter_dtype=torch.float16,
        family="sd3_5",
        precision=RolePrecision("fp16", "tf32"),
        model_config={},
    )

    _, kwargs = diffusers_pipeline_dtypes(build, torch.float16)

    assert "revision" not in kwargs
    assert "local_files_only" not in kwargs


def test_flow_match_replay_maps_sana_flow_shift(monkeypatch) -> None:
    from vrl.models import loader

    calls: list[dict] = []

    class Config(dict):
        def __getattr__(self, name):
            return self[name]

    class Scheduler:
        def __init__(self, config):
            self.config = Config(config)
            self.timesteps = None

        @classmethod
        def from_config(cls, config, **kwargs):
            calls.append(dict(kwargs))
            return cls({**dict(config), **kwargs})

        def set_timesteps(self, num_steps, device=None):
            self.timesteps = (num_steps, device)

    original = Scheduler({"flow_shift": 3.0, "shift": 1.0})
    monkeypatch.setattr(loader, "load_diffusers_scheduler", lambda *args, **kwargs: original)
    build = SimpleNamespace(num_steps=10, device="cpu")

    scheduler = loader.load_flow_match_scheduler(build)

    assert calls == [{"shift": 3.0}]
    assert scheduler.config.shift == 3.0
    assert scheduler.timesteps == (10, "cpu")


def test_flow_match_replay_keeps_native_shift_config(monkeypatch) -> None:
    from vrl.models import loader

    scheduler = SimpleNamespace(config=SimpleNamespace(shift=3.0))
    monkeypatch.setattr(loader, "load_diffusers_scheduler", lambda *args, **kwargs: scheduler)

    assert loader.load_flow_match_scheduler(SimpleNamespace()) is scheduler


def test_config_revision_kwargs_omit_absent_dependency_revision() -> None:
    build = ModelBuild(
        model_name_or_path="org/model",
        revision="immutable-revision",
        device="cpu",
        parameter_dtype=torch.float16,
        family="sd3_5",
        precision=RolePrecision("fp16", "tf32"),
        model_config={},
    )

    assert build.config_revision_kwargs("tokenizer_revision") == {}
