from __future__ import annotations

from pathlib import Path

import pytest

from vrl.rewards.models.hub import parse_hf_repo_revision


def test_parse_hf_repo_revision_defaults_bare_repo_to_main() -> None:
    """Checks bare Hugging Face repo ids use the default revision."""

    ref = parse_hf_repo_revision("org/model")

    assert ref.repo_id == "org/model"
    assert ref.revision == "main"


def test_parse_hf_repo_revision_accepts_explicit_revision() -> None:
    """Checks repo@revision model references split at the final at sign."""

    ref = parse_hf_repo_revision("org/model@checkpoint-1")

    assert ref.repo_id == "org/model"
    assert ref.revision == "checkpoint-1"


def test_parse_hf_repo_revision_defaults_empty_revision() -> None:
    """Checks repo@ uses the default revision instead of an empty string."""

    ref = parse_hf_repo_revision("org/model@")

    assert ref.repo_id == "org/model"
    assert ref.revision == "main"


def test_parse_hf_repo_revision_rejects_missing_repo_id() -> None:
    """Checks malformed references fail before Hugging Face download."""

    with pytest.raises(ValueError, match="repo id"):
        parse_hf_repo_revision("@main")


def test_resolve_model_root_uses_shared_repo_revision_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the shared resolver passes parsed repo id, revision, and offline flag."""

    from vrl.rewards.models.hub import resolve_model_root

    captured = {}

    def _fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

    resolved = resolve_model_root(
        {
            "reward_model_name": "videophysics/videocon_physics@paper-rev",
            "local_files_only": True,
        },
        default_model="videophysics/videocon_physics",
        family="VideoCon-Physics",
    )

    assert resolved == tmp_path.resolve()
    assert captured["repo_id"] == "videophysics/videocon_physics"
    assert captured["revision"] == "paper-rev"
    assert captured["local_files_only"] is True
