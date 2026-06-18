"""Shared Hugging Face model-reference helpers for reward loaders."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HF_REVISION = "main"


@dataclass(frozen=True, slots=True)
class HuggingFaceRepoRevision:
    """Parsed ``repo_id@revision`` model reference."""

    repo_id: str
    revision: str


def parse_hf_repo_revision(
    model_reference: str,
    *,
    default_revision: str = DEFAULT_HF_REVISION,
) -> HuggingFaceRepoRevision:
    """Parse ``repo_id@revision`` with a default revision for bare repo ids."""

    text = str(model_reference).strip()
    repo_id, separator, revision = text.rpartition("@")
    if not separator:
        repo_id = text
        revision = default_revision
    else:
        repo_id = repo_id.strip()
        revision = revision.strip() or default_revision
    if not repo_id:
        raise ValueError("Hugging Face model reference must include a repo id")
    return HuggingFaceRepoRevision(repo_id=repo_id, revision=str(revision))


__all__ = ["DEFAULT_HF_REVISION", "HuggingFaceRepoRevision", "parse_hf_repo_revision"]
