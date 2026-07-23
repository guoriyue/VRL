"""Revision propagation tests for Anima single-file artifacts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch


def _build(revision: str | None) -> SimpleNamespace:
    model_config: dict[str, Any] = {
        "transformer_file": "weights/transformer.safetensors",
        "text_encoder_file": "weights/text_encoder.safetensors",
        "vae_file": "weights/vae.safetensors",
    }
    return SimpleNamespace(
        model_name_or_path="org/anima",
        revision=revision,
        model_config=model_config,
        parameter_dtype=torch.float32,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("revision", [None, "anima-immutable-revision"])
def test_anima_hub_artifact_uses_optional_revision(
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
) -> None:
    """The Hub resolver omits revision when unset and forwards it when set."""
    from vrl.models.families.cosmos.anima.model import _resolve_artifact

    calls: list[dict[str, Any]] = []

    def fake_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "/cache/anima.safetensors"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)

    result = _resolve_artifact(
        "org/anima",
        explicit_path="",
        relative_file="weights/transformer.safetensors",
        field_name="transformer_path",
        revision=revision,
    )

    expected = {
        "repo_id": "org/anima",
        "filename": "weights/transformer.safetensors",
    }
    if revision is not None:
        expected["revision"] = revision
    assert result == "/cache/anima.safetensors"
    assert calls == [expected]


@pytest.mark.parametrize("revision", [None, "anima-immutable-revision"])
def test_anima_rollout_resolves_every_artifact_at_model_revision(
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
) -> None:
    """Rollout passes one model revision to all three same-repo artifacts."""
    from vrl.models.families.cosmos.anima import model as anima_model

    calls: list[dict[str, Any]] = []

    def fake_resolve(root: str, **kwargs: Any) -> str:
        calls.append({"root": root, **kwargs})
        return f"/cache/{kwargs['field_name']}.safetensors"

    class _StopAfterResolution(RuntimeError):
        pass

    def stop_load(*_args: Any, **_kwargs: Any) -> Any:
        raise _StopAfterResolution

    import safetensors.torch

    monkeypatch.setattr(anima_model, "_resolve_artifact", fake_resolve)
    monkeypatch.setattr(safetensors.torch, "load_file", stop_load)

    with pytest.raises(_StopAfterResolution):
        anima_model.AnimaModel.from_build(_build(revision))

    assert [call["field_name"] for call in calls] == [
        "transformer_path",
        "text_encoder_path",
        "vae_path",
    ]
    if revision is None:
        assert all("revision" not in call for call in calls)
    else:
        assert {call["revision"] for call in calls} == {revision}


@pytest.mark.parametrize("revision", [None, "anima-immutable-revision"])
def test_anima_replay_resolves_transformer_at_model_revision(
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
) -> None:
    """Replay forwards the same model revision to its transformer resolver."""
    from vrl.models.families.cosmos.anima import runtime as anima_runtime

    calls: list[dict[str, Any]] = []

    def fake_resolve(root: str, **kwargs: Any) -> str:
        calls.append({"root": root, **kwargs})
        return "/cache/transformer.safetensors"

    class _StopAfterResolution(RuntimeError):
        pass

    def stop_load(*_args: Any, **_kwargs: Any) -> Any:
        raise _StopAfterResolution

    import safetensors.torch

    monkeypatch.setattr(anima_runtime, "_resolve_artifact", fake_resolve)
    monkeypatch.setattr(safetensors.torch, "load_file", stop_load)

    with pytest.raises(_StopAfterResolution):
        anima_runtime.load_anima_transformer(_build(revision))

    assert len(calls) == 1
    assert calls[0]["field_name"] == "transformer_path"
    if revision is None:
        assert "revision" not in calls[0]
    else:
        assert calls[0]["revision"] == revision
