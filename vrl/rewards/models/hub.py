"""Shared helpers for locating and loading reward-model checkpoints.

Model references (``repo_id@revision``, local roots) live here, and so does the
Qwen2-VL key relocation: transformers 4.52 nested the Qwen2-VL modules
(``model.*`` under ``model.language_model.*``, ``visual.*`` under
``model.visual.*``), and the published HPSv3 and Kling VideoReward checkpoints
predate that rename, so their keys are moved before a ``strict=True`` load.
The relocation is per key and idempotent, so a state dict that already mixes
both layouts (a live model's keys updated with a legacy LoRA/non-LoRA split)
still lands on the nested layout.
"""

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
    model_ref = HuggingFaceRepoRevision.parse(reward_model_name)

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_ref.repo_id,
            revision=model_ref.revision,
            local_files_only=bool(worker_config.get("local_files_only", False)),
        ),
    ).resolve()


_NESTED_MODULES = ("language_model.", "visual.")


def remap_legacy_qwen2vl_key(key: str, *, prefix: str = "") -> str:
    """Move one pre-4.52 key under the nested layout; nested keys pass through.

    ``prefix`` is whatever wraps the Qwen2-VL model in the checkpoint (for a
    PEFT-wrapped model ``"base_model.model."``); keys outside ``{prefix}model.``
    and ``{prefix}visual.`` (``lm_head``, reward heads) are left untouched.
    """

    visual_prefix = f"{prefix}visual."
    model_prefix = f"{prefix}model."
    if key.startswith(visual_prefix):
        return f"{model_prefix}visual.{key[len(visual_prefix) :]}"
    if key.startswith(model_prefix):
        rest = key[len(model_prefix) :]
        if not rest.startswith(_NESTED_MODULES):
            return f"{model_prefix}language_model.{rest}"
    return key


def remap_legacy_qwen2vl_state_dict(
    state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Mapping[str, Any]:
    """Relocate ``state`` onto the live model's layout when that yields its exact key set.

    The caller loads ``strict=True`` afterwards, so anything short of an exact
    match returns ``state`` unchanged and lets the strict load report the
    real mismatch instead of a half-relocated one.
    """

    remapped = {
        remap_legacy_qwen2vl_key(str(key), prefix=prefix): value for key, value in state.items()
    }
    return remapped if set(remapped) == set(target_state) else state


__all__ = [
    "HuggingFaceRepoRevision",
    "remap_legacy_qwen2vl_key",
    "remap_legacy_qwen2vl_state_dict",
    "resolve_model_root",
]
