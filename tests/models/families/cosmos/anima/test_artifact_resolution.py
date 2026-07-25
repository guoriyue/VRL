"""Revision propagation tests for Anima single-file artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild


def _anima_build(
    *,
    revision: str | None,
    model_name_or_path: str = "org/anima",
    model_config: dict[str, Any] | None = None,
) -> ModelBuild:
    return ModelBuild(
        model_name_or_path=model_name_or_path,
        revision=revision,
        device=torch.device("cpu"),
        parameter_dtype=torch.float32,
        family="cosmos-predict2-anima",
        precision=RolePrecision("fp32", "tf32", outer_autocast=False),
        model_config=model_config
        if model_config is not None
        else {
            "transformer_file": "weights/transformer.safetensors",
            "text_encoder_file": "weights/text_encoder.safetensors",
            "vae_file": "weights/vae.safetensors",
        },
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


@pytest.mark.parametrize("relative_file", ["../outside.safetensors", "/tmp/outside"])
def test_anima_member_cannot_escape_checkpoint_source(relative_file: str) -> None:
    from vrl.models.families.cosmos.anima.model import _resolve_artifact

    with pytest.raises(ValueError, match="stay within its checkpoint source"):
        _resolve_artifact(
            "org/anima",
            explicit_path="",
            relative_file=relative_file,
            field_name="transformer_path",
            revision="immutable",
        )


@pytest.mark.parametrize("revision", [None, "anima-immutable-revision"])
def test_anima_rollout_resolves_every_artifact_at_model_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
) -> None:
    """Rollout passes one model revision to all three same-repo artifacts."""
    from vrl.models.families.cosmos.anima import model as anima_model

    calls: list[dict[str, Any]] = []
    artifact = tmp_path / "artifact.safetensors"
    from safetensors.torch import save_file

    save_file({"marker": torch.ones(1)}, artifact)

    def fake_resolve(root: str, **kwargs: Any) -> str:
        calls.append({"root": root, **kwargs})
        return str(artifact)

    class _StopAfterResolution(RuntimeError):
        pass

    def stop_after_load(state: dict[str, torch.Tensor], *, dtype: torch.dtype) -> Any:
        assert set(state) == {"marker"}
        assert dtype == torch.float32
        raise _StopAfterResolution

    monkeypatch.setattr(anima_model, "_resolve_artifact", fake_resolve)
    monkeypatch.setattr(anima_model, "_load_anima_transformer", stop_after_load)

    with pytest.raises(_StopAfterResolution):
        anima_model.AnimaModel.from_build(_anima_build(revision=revision))

    assert [call["field_name"] for call in calls] == [
        "transformer_path",
        "text_encoder_path",
        "vae_path",
    ]
    if revision is None:
        assert all("revision" not in call for call in calls)
    else:
        assert {call["revision"] for call in calls} == {revision}


def test_anima_rollout_resolution_preserves_build_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
    from vrl.models.families.cosmos.anima import model as anima_model

    root = tmp_path / "anima"
    artifact_files = {
        "transformer_file": "weights/transformer.safetensors",
        "text_encoder_file": "weights/text_encoder.safetensors",
        "vae_file": "weights/vae.safetensors",
    }
    from safetensors.torch import save_file

    for relative_file in artifact_files.values():
        path = root / relative_file
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({f"{path.stem}_marker": torch.ones(1)}, path)
    qwen_tokenizer = tmp_path / "qwen-tokenizer"
    t5_tokenizer = tmp_path / "t5-tokenizer"
    qwen_tokenizer.mkdir()
    t5_tokenizer.mkdir()
    (qwen_tokenizer / "tokenizer.json").write_text("qwen")
    (t5_tokenizer / "tokenizer.json").write_text("t5")

    model_config: dict[str, Any] = {
        **artifact_files,
        "qwen_tokenizer_path": str(qwen_tokenizer),
        "qwen_tokenizer_revision": None,
        "t5_tokenizer_path": str(t5_tokenizer),
        "t5_tokenizer_revision": None,
    }
    build = _anima_build(
        revision=None,
        model_name_or_path=str(root),
        model_config=model_config,
    )
    configured_model_config = dict(model_config)
    identity_before = resolve_checkpoint_model_identity(build)

    class _StopAfterResolution(RuntimeError):
        pass

    def stop_after_transformer_load(
        state: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> Any:
        assert set(state) == {"transformer_marker"}
        assert dtype == torch.float32
        raise _StopAfterResolution

    monkeypatch.setattr(
        anima_model,
        "_load_anima_transformer",
        stop_after_transformer_load,
    )

    with pytest.raises(_StopAfterResolution):
        anima_model.AnimaModel.from_build(build)
    identity_after = resolve_checkpoint_model_identity(build)

    assert build.model_config == configured_model_config
    assert identity_after == identity_before


@pytest.mark.parametrize("revision", [None, "anima-immutable-revision"])
def test_anima_replay_resolves_transformer_at_model_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
) -> None:
    """Replay forwards the same model revision to its transformer resolver."""
    from vrl.models.families.cosmos.anima import model as anima_model
    from vrl.models.families.cosmos.anima import runtime as anima_runtime

    calls: list[dict[str, Any]] = []
    artifact = tmp_path / "transformer.safetensors"
    from safetensors.torch import save_file

    save_file({"marker": torch.ones(1)}, artifact)

    def fake_resolve(root: str, **kwargs: Any) -> str:
        calls.append({"root": root, **kwargs})
        return str(artifact)

    class _StopAfterResolution(RuntimeError):
        pass

    def stop_after_load(state: dict[str, torch.Tensor], *, dtype: torch.dtype) -> Any:
        assert set(state) == {"marker"}
        assert dtype == torch.float32
        raise _StopAfterResolution

    monkeypatch.setattr(anima_runtime, "_resolve_artifact", fake_resolve)
    monkeypatch.setattr(anima_model, "_load_anima_transformer", stop_after_load)

    with pytest.raises(_StopAfterResolution):
        anima_runtime.load_anima_transformer(_anima_build(revision=revision))

    assert len(calls) == 1
    assert calls[0]["field_name"] == "transformer_path"
    if revision is None:
        assert "revision" not in calls[0]
    else:
        assert calls[0]["revision"] == revision
