"""Shared helpers for locating and loading reward-model checkpoints.

Model references (``repo_id@revision``, local roots) live here, and so does
:func:`relocate_checkpoint_keys`: the reward loaders build a Qwen2-VL model
with ``from_pretrained`` and then overlay a fine-tuned state dict with
``load_state_dict(strict=True)`` (after resizing embeddings or wrapping in
PEFT), which bypasses the key conversions ``from_pretrained`` would apply.
The published HPSv3 and Kling VideoReward checkpoints predate the
transformers 4.52 Qwen2-VL nesting (``model.*`` under
``model.language_model.*``, ``visual.*`` under ``model.visual.*``), so the
overlay runs their keys through the model's own conversion mapping first.
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


def relocate_checkpoint_keys(model: Any, state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Rename ``state``'s keys the way ``from_pretrained`` would for ``model``.

    The conversion rules come from transformers itself (the same ones its
    loader applies), so no Qwen2-VL layout knowledge lives here; a PEFT
    wrapper is peeled to reach the pretrained model and its key prefix is
    kept. The caller loads ``strict=True`` afterwards, so anything short of an
    exact match with the live keys returns ``state`` unchanged and lets the
    strict load report the real mismatch instead of a half-relocated one.
    """

    from transformers import PreTrainedModel
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    prefix = ""
    pretrained = model
    if not isinstance(pretrained, PreTrainedModel):
        from peft import PeftModel

        if not isinstance(pretrained, PeftModel):
            raise TypeError(
                f"relocate_checkpoint_keys expects a PreTrainedModel or PeftModel, got {type(model)!r}",
            )
        pretrained = model.get_base_model()
        prefix = "base_model.model."
    # transformers registers the Qwen2-VL rules under the ForConditionalGeneration
    # class name; the reward heads subclass it, so walk the MRO the way a
    # task-head override would, then fall back to the model type.
    transforms = None
    for cls in type(pretrained).__mro__:
        transforms = get_checkpoint_conversion_mapping(cls.__name__)
        if transforms is not None:
            break
    if transforms is None:
        transforms = get_checkpoint_conversion_mapping(pretrained.config.model_type) or []

    def relocate(key: str) -> str:
        if not key.startswith(prefix):
            return key
        inner = key[len(prefix) :]
        for transform in transforms:
            inner, matched = transform.rename_source_key(inner)
            if matched is not None:
                break
        return prefix + inner

    live_keys = set(model.state_dict())
    if set(state) == live_keys:
        return state
    remapped = {relocate(str(key)): value for key, value in state.items()}
    return remapped if set(remapped) == live_keys else state


__all__ = [
    "HuggingFaceRepoRevision",
    "relocate_checkpoint_keys",
    "resolve_model_root",
]
