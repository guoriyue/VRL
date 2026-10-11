"""Hugging Face Hub references: ``repo_id@revision`` and what to fetch from them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vrl.utils.paths import user_path


@dataclass(frozen=True, slots=True)
class HuggingFaceRepoRevision:
    """Parsed ``repo_id@revision`` model reference."""

    repo_id: str
    revision: str

    @classmethod
    def parse(cls, model_reference: str) -> HuggingFaceRepoRevision:
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
        return cls(repo_id=repo_id, revision=str(revision))

    def download(
        self,
        *,
        local_files_only: bool = False,
        allow_patterns: Sequence[str] | None = None,
    ) -> Path:
        """The snapshot's local directory, fetched into the Hugging Face cache."""

        from huggingface_hub import snapshot_download

        options: dict[str, object] = {}
        if local_files_only:
            options["local_files_only"] = True
        if allow_patterns is not None:
            options["allow_patterns"] = list(allow_patterns)
        try:
            snapshot = snapshot_download(repo_id=self.repo_id, revision=self.revision, **options)
        except Exception as exc:
            mode = "local cache" if local_files_only else "Hugging Face download"
            raise RuntimeError(
                f"Failed to fetch {self.repo_id}@{self.revision} from {mode}. Point the "
                "config at a local directory, or pre-download the model cache before "
                "training.",
            ) from exc
        return Path(snapshot).resolve()

    def download_file(
        self,
        filename: str,
        *,
        repo_type: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> Path:
        """One file of the repo, fetched into the Hugging Face cache."""

        from huggingface_hub import hf_hub_download

        options: dict[str, object] = {}
        if repo_type is not None:
            options["repo_type"] = repo_type
        if cache_dir is not None:
            options["cache_dir"] = str(user_path(cache_dir))
        try:
            local = hf_hub_download(
                repo_id=self.repo_id, filename=filename, revision=self.revision, **options
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch {filename} of {self.repo_id}@{self.revision} from the "
                "Hugging Face hub. Point the config at a local file, or pre-download it.",
            ) from exc
        return Path(local).resolve()


__all__ = ["HuggingFaceRepoRevision"]
