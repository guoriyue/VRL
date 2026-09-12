from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise.base import DiffusersPipelineModelBase


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

    _, kwargs = DiffusersPipelineModelBase._pipeline_load_dtypes(build, torch.float16)

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

    _, kwargs = DiffusersPipelineModelBase._pipeline_load_dtypes(build, torch.float16)

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

    _, kwargs = DiffusersPipelineModelBase._pipeline_load_dtypes(build, torch.float16)

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


def test_pipeline_load_preserves_source_vae_precision(tmp_path) -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    from tests.models.steps.denoise.fixtures import (
        build_tiny_autoencoder_kl,
        build_tiny_pipeline_shell,
        build_tiny_sd3_transformer,
    )

    pipeline = build_tiny_pipeline_shell(
        transformer=build_tiny_sd3_transformer(),
        vae=build_tiny_autoencoder_kl(),
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    with torch.no_grad():
        next(pipeline.vae.parameters()).fill_(1.0001)
    expected = next(pipeline.vae.parameters()).detach().clone()
    assert not torch.equal(expected, expected.bfloat16().float())
    pipeline.save_pretrained(tmp_path)
    build = ModelBuild(
        model_name_or_path=str(tmp_path),
        revision=None,
        device="cpu",
        parameter_dtype=torch.bfloat16,
        family="sd3_5",
        precision=RolePrecision("bf16", "tf32"),
        model_config={},
    )
    _, kwargs = DiffusersPipelineModelBase._pipeline_load_dtypes(build, torch.bfloat16)
    loaded = type(pipeline).from_pretrained(tmp_path, text_encoder=None, **kwargs)
    assert next(loaded.transformer.parameters()).dtype == torch.bfloat16
    assert next(loaded.vae.parameters()).dtype == torch.float32
    assert torch.equal(next(loaded.vae.parameters()), expected)


def test_pipeline_dtype_projection_keeps_encoder_override_separate_from_model() -> None:
    from vrl.models.interfaces.runtime import RolloutBuildOptions

    build = ModelBuild(
        model_name_or_path="org/model",
        revision="snapshot",
        device="cpu",
        parameter_dtype=torch.bfloat16,
        family="sd3_5",
        precision=RolePrecision("bf16", "tf32"),
        model_config={},
        rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.float32),
    )

    class DualEncoderPipelineModel(DiffusersPipelineModelBase):
        _frozen_encoder_names = ("text_encoder", "text_encoder_2")

    encoder_dtype, kwargs = DualEncoderPipelineModel._pipeline_load_dtypes(
        build,
        torch.bfloat16,
    )
    assert encoder_dtype == torch.float32
    assert kwargs["torch_dtype"] == {
        "default": torch.bfloat16,
        "vae": torch.float32,
        "text_encoder": torch.float32,
        "text_encoder_2": torch.float32,
    }
    assert kwargs["revision"] == "snapshot"


@pytest.mark.parametrize("num_steps", [0, -1, True, 2.5, "3"])
def test_model_build_rejects_coerced_scheduler_step_count(num_steps):
    build = ModelBuild(
        model_name_or_path="org/model",
        device="cpu",
        parameter_dtype=torch.float16,
        family="sd3_5",
        precision=RolePrecision("fp16", "tf32"),
        sampling_config={"num_steps": num_steps},
        revision=None,
    )
    with pytest.raises(ValueError, match=r"sampling.num_steps"):
        _ = build.num_steps
