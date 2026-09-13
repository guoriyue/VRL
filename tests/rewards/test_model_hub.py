from __future__ import annotations

from pathlib import Path

import pytest

from vrl.rewards.models.hub import HuggingFaceRepoRevision


@pytest.mark.parametrize(
    ("reference", "repo_id", "revision"),
    [
        ("org/model", "org/model", "main"),
        ("org/model@checkpoint-1", "org/model", "checkpoint-1"),
        # ``repo@`` means the default revision, never an empty string.
        ("org/model@", "org/model", "main"),
    ],
)
def test_parse_hf_repo_revision_splits_at_the_final_at_sign(
    reference: str, repo_id: str, revision: str
) -> None:
    ref = HuggingFaceRepoRevision.parse(reference)

    assert (ref.repo_id, ref.revision) == (repo_id, revision)


def test_parse_hf_repo_revision_rejects_missing_repo_id() -> None:
    """Checks malformed references fail before Hugging Face download."""

    with pytest.raises(ValueError, match="repo id"):
        HuggingFaceRepoRevision.parse("@main")


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
