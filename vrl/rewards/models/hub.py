"""Shared Hugging Face model-reference helpers for reward loaders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class HuggingFaceRepoRevision:
    """Parsed ``repo_id@revision`` model reference."""

    repo_id: str
    revision: str


def parse_hf_repo_revision(model_reference: str) -> HuggingFaceRepoRevision:
    """Parse ``repo_id@revision`` with a default revision for bare repo ids."""

    text = str(model_reference).strip()
    repo_id, separator, revision = text.rpartition("@")
    if not separator:
        repo_id = text
        revision = "main"
    else:
        repo_id = repo_id.strip()
        revision = revision.strip() or "main"
    if not repo_id:
        raise ValueError("Hugging Face model reference must include a repo id")
    return HuggingFaceRepoRevision(repo_id=repo_id, revision=str(revision))


def resolve_model_root(
    worker_config: Mapping[str, Any],
    *,
    default_model: str,
    family: str,
) -> Path:
    """Return a local checkpoint dir: ``model_path`` if set, else snapshot_download.

    Shared by the reward judges that wrap ready public models (videoscore2,
    unified_reward_video, videocon_physics). Kling keeps its
    own resolver: it pins a revision, wraps download failures in a RuntimeError
    with recovery hints, and validates the checkpoint layout.
    """

    model_path = str(worker_config.get("model_path", "")).strip()
    if model_path:
        root = Path(model_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"{family} model_path missing: {root}")
        return root

    reward_model_name = str(worker_config.get("reward_model_name", "")).strip()
    if not reward_model_name:
        reward_model_name = default_model
    model_ref = parse_hf_repo_revision(reward_model_name)

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_ref.repo_id,
            revision=model_ref.revision,
            local_files_only=bool(worker_config.get("local_files_only", False)),
        ),
    ).resolve()


__all__ = [
    "HuggingFaceRepoRevision",
    "parse_hf_repo_revision",
    "resolve_model_root",
]
