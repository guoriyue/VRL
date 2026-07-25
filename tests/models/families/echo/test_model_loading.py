"""Download-free revision propagation tests for JoyAI Echo artifacts."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch

from vrl.config.loading import load_config
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild


class _FakeEcho(torch.nn.Module):
    def __init__(self, *, video_height: int, video_width: int) -> None:
        super().__init__()
        self.model = torch.nn.Linear(2, 2)
        self.video_height = video_height
        self.video_width = video_width


def _patch_echo_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_calls: list[dict[str, Any]] | None = None,
) -> None:
    from ltx_distillation.models import ltx_wrapper, text_encoder_wrapper

    def create_wrapper(**kwargs: Any) -> _FakeEcho:
        if wrapper_calls is not None:
            wrapper_calls.append(kwargs)
        return _FakeEcho(
            video_height=int(kwargs["video_height"]),
            video_width=int(kwargs["video_width"]),
        )

    monkeypatch.setattr(
        ltx_wrapper,
        "create_ltx2_wrapper",
        create_wrapper,
    )
    monkeypatch.setattr(
        text_encoder_wrapper,
        "create_text_encoder_wrapper",
        lambda **_kwargs: object(),
    )
    vae_wrapper = types.ModuleType("ltx_distillation.models.vae_wrapper")
    vae_wrapper.create_vae_wrappers = lambda **_kwargs: (object(), object())
    monkeypatch.setitem(
        sys.modules,
        "ltx_distillation.models.vae_wrapper",
        vae_wrapper,
    )


def _build(gemma_path: str, revision: str | None) -> ModelBuild:
    return ModelBuild(
        model_name_or_path="jdopensource/JoyAI-Echo",
        revision=revision,
        device=torch.device("cpu"),
        parameter_dtype=torch.float32,
        family="echo",
        precision=RolePrecision("fp32", "tf32", outer_autocast=False),
        model_config={
            "gemma_path": gemma_path,
            "use_lora": False,
            "video_height": 64,
            "video_width": 64,
        },
        sampling_config={"height": 64, "width": 64},
    )


@pytest.mark.parametrize("revision", [None, "echo-immutable-revision"])
def test_echo_rollout_checkpoint_uses_optional_model_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    revision: str | None,
) -> None:
    """Rollout resolves the merged checkpoint at the configured revision."""
    from vrl.models.families.echo.model import EchoModel

    wrapper_calls: list[dict[str, Any]] = []
    _patch_echo_dependencies(monkeypatch, wrapper_calls)
    gemma_path = tmp_path / "gemma"
    gemma_path.mkdir()
    calls: list[dict[str, Any]] = []

    def fake_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        return "/cache/echo.safetensors"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)

    EchoModel.from_build(_build(str(gemma_path), revision))

    expected = {
        "repo_id": "jdopensource/JoyAI-Echo",
        "filename": "JoyAI-Echo-release.safetensors",
    }
    if revision is not None:
        expected["revision"] = revision
    assert calls == [expected]
    assert wrapper_calls[0]["video_height"] == 64
    assert wrapper_calls[0]["video_width"] == 64


@pytest.mark.parametrize("revision", [None, "echo-immutable-revision"])
def test_echo_replay_checkpoint_uses_optional_model_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    revision: str | None,
) -> None:
    """Replay resolves the same merged checkpoint revision as rollout."""
    from vrl.models.families.echo.runtime import build_echo_replay_runtime_bundle

    wrapper_calls: list[dict[str, Any]] = []
    _patch_echo_dependencies(monkeypatch, wrapper_calls)
    gemma_path = tmp_path / "gemma"
    gemma_path.mkdir()
    calls: list[dict[str, Any]] = []

    def fake_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        return "/cache/echo.safetensors"

    import huggingface_hub

    import vrl.models.steps.denoise.build as diffusion_build

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)
    monkeypatch.setattr(
        diffusion_build,
        "assemble_replay_bundle",
        lambda model, _build, **_kwargs: model,
    )

    build_echo_replay_runtime_bundle(_build(str(gemma_path), revision))

    expected = {
        "repo_id": "jdopensource/JoyAI-Echo",
        "filename": "JoyAI-Echo-release.safetensors",
    }
    if revision is not None:
        expected["revision"] = revision
    assert calls == [expected]
    assert wrapper_calls[0]["video_height"] == 64
    assert wrapper_calls[0]["video_width"] == 64


def test_echo_build_rejects_sampling_dimensions_before_checkpoint_loading(tmp_path) -> None:
    from vrl.models.families.echo.model import EchoModel

    gemma_path = tmp_path / "gemma"
    gemma_path.mkdir()
    build = _build(str(gemma_path), revision=None)
    build.sampling_config["height"] = 96

    with pytest.raises(
        ValueError,
        match=r"sampling\.height=96.*model\.video_height=64",
    ):
        EchoModel.from_build(build)


def test_echo_direct_build_rejects_invalid_model_dimensions_before_loading(tmp_path) -> None:
    from vrl.models.families.echo.model import EchoModel

    gemma_path = tmp_path / "gemma"
    gemma_path.mkdir()
    build = _build(str(gemma_path), revision=None)
    build.model_config["video_width"] = 65
    build.sampling_config["width"] = 65

    with pytest.raises(
        ValueError,
        match=r"model\.video_width must be a positive multiple of 32, got 65",
    ):
        EchoModel.from_build(build)


def test_echo_preset_derives_request_dimensions_from_model_construction() -> None:
    cfg = load_config("experiment/echo/online_grpo_kling_video_reward")

    assert cfg.model.video_height == 256
    assert cfg.model.video_width == 256
    assert cfg.sampling.height == cfg.model.video_height
    assert cfg.sampling.width == cfg.model.video_width


@pytest.mark.parametrize("revision", [None, "gemma-immutable-revision"])
def test_echo_gemma_uses_its_own_optional_revision(
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
) -> None:
    from vrl.models.families.echo.model import _resolve_gemma_dir

    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "/cache/gemma"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    assert _resolve_gemma_dir("google/gemma-3-12b-it", revision=revision) == "/cache/gemma"
    assert calls[0]["repo_id"] == "google/gemma-3-12b-it"
    if revision is None:
        assert "revision" not in calls[0]
    else:
        assert calls[0]["revision"] == revision
