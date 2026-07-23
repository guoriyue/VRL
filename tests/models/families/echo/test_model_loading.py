"""Download-free revision propagation tests for JoyAI Echo artifacts."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch


class _FakeEcho(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(2, 2)


def _patch_echo_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    from ltx_distillation.models import ltx_wrapper, text_encoder_wrapper

    monkeypatch.setattr(
        ltx_wrapper,
        "create_ltx2_wrapper",
        lambda **_kwargs: _FakeEcho(),
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


def _build(gemma_path: str, revision: str | None) -> SimpleNamespace:
    model_config = {"gemma_path": gemma_path, "use_lora": False}
    return SimpleNamespace(
        model_name_or_path="jdopensource/JoyAI-Echo",
        revision=revision,
        model_config=model_config,
        sampling_config={"height": 64, "width": 64},
        parameter_dtype=torch.float32,
        device=torch.device("cpu"),
        num_steps=None,
    )


@pytest.mark.parametrize("revision", [None, "echo-immutable-revision"])
def test_echo_rollout_checkpoint_uses_optional_model_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    revision: str | None,
) -> None:
    """Rollout resolves the merged checkpoint at the configured revision."""
    from vrl.models.families.echo.model import EchoModel

    _patch_echo_dependencies(monkeypatch)
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


@pytest.mark.parametrize("revision", [None, "echo-immutable-revision"])
def test_echo_replay_checkpoint_uses_optional_model_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    revision: str | None,
) -> None:
    """Replay resolves the same merged checkpoint revision as rollout."""
    from vrl.models.families.echo.runtime import build_echo_replay_runtime_bundle

    _patch_echo_dependencies(monkeypatch)
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
