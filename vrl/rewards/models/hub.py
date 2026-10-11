"""Shared helpers for locating and loading reward-model checkpoints.

A judge's checkpoint directory (``worker_model_root``) lives here, and so does
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
from pathlib import Path
from typing import Any

from vrl.utils.hf_hub import HuggingFaceRepoRevision
from vrl.utils.paths import user_path


def worker_model_root(
    worker_config: Mapping[str, Any],
    *,
    default_model: str,
    family: str,
) -> Path:
    """A judge's checkpoint directory from its ``worker_config``.

    ``model_path`` names a local directory; otherwise ``reward_model_name``
    (``repo_id@revision``, ``default_model`` when absent) is fetched into the
    Hugging Face cache. ``family`` only names the judge in errors.
    """

    model_path = str(worker_config.get("model_path", "")).strip()
    if model_path:
        root = user_path(model_path)
        if not root.exists():
            raise FileNotFoundError(f"{family} model_path missing: {root}")
        return root
    reward_model_name = str(worker_config.get("reward_model_name", "")).strip()
    return HuggingFaceRepoRevision.parse(reward_model_name or default_model).download(
        local_files_only=bool(worker_config.get("local_files_only", False)),
    )


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
    "relocate_checkpoint_keys",
    "worker_model_root",
]
