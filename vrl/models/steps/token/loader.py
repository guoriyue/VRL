"""Hugging Face checkpoint adapters for AR families.

Local paths pass through untouched; hub ids are snapshot-downloaded once and
reused. Families that assemble their own upstream pipeline (nextstep_1) need a
directory, not a loaded module, and the AR replay cores
(:class:`vrl.models.steps.token.base.ARReplayCore`) strict-load only the subset
of checkpoint weights that their trainer-side module owns.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Protocol


class _StateDictModule(Protocol):
    """Minimal module surface consumed by the checkpoint format adapter."""

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ) -> Any: ...


def resolve_hf_checkpoint_dir(
    model_path: str,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
) -> str:
    """Local dir passthrough, else the HF snapshot dir (downloading if needed)."""
    if os.path.isdir(model_path):
        base = model_path
    else:
        from huggingface_hub import snapshot_download

        base = snapshot_download(model_path, revision=revision)
    return os.path.join(base, subfolder) if subfolder else base


def load_ar_replay_checkpoint(module: _StateDictModule, checkpoint_dir: str) -> None:
    """Strict-load the core-owned subset of an HF checkpoint into ``module``.

    Replay cores deliberately omit rollout-only towers and decoders. For a
    sharded checkpoint, the HF index therefore selects only shards containing
    keys owned by ``module``; irrelevant shards are never opened and relevant
    shards are installed one at a time to keep host-memory use bounded. For
    either layout, unrelated keys are filtered before PyTorch sees the state
    dict. Missing core keys fail loudly because replaying with random
    parameters would silently corrupt the training objective.
    """

    from transformers.modeling_utils import load_state_dict
    from transformers.utils import (
        SAFE_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
        WEIGHTS_INDEX_NAME,
        WEIGHTS_NAME,
    )

    core_keys = set(module.state_dict())
    index_path = next(
        (
            os.path.join(checkpoint_dir, filename)
            for filename in (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME)
            if os.path.exists(os.path.join(checkpoint_dir, filename))
        ),
        None,
    )
    if index_path is not None:
        with open(index_path, encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        missing_keys = sorted(core_keys - weight_map.keys())
        if not missing_keys:
            shard_names = sorted(
                {name for key, name in weight_map.items() if key in core_keys},
            )
            loaded_keys: set[str] = set()
            for shard_name in shard_names:
                shard_state = load_state_dict(os.path.join(checkpoint_dir, shard_name))
                core_state = {key: value for key, value in shard_state.items() if key in core_keys}
                module.load_state_dict(core_state, strict=False)
                loaded_keys.update(core_state)
            missing_keys = sorted(core_keys - loaded_keys)
    else:
        checkpoint_path = next(
            (
                os.path.join(checkpoint_dir, filename)
                for filename in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME)
                if os.path.exists(os.path.join(checkpoint_dir, filename))
            ),
            None,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No supported {type(module).__name__} checkpoint file found in {checkpoint_dir}",
            )
        state = load_state_dict(checkpoint_path)
        checkpoint_state = {key: value for key, value in state.items() if key in core_keys}
        missing_keys = sorted(core_keys - checkpoint_state.keys())
        if not missing_keys:
            module.load_state_dict(checkpoint_state, strict=True)

    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        suffix = " ..." if len(missing_keys) > 5 else ""
        raise RuntimeError(
            f"{type(module).__name__} checkpoint is missing keys: {preview}{suffix}",
        )


__all__ = ["load_ar_replay_checkpoint", "resolve_hf_checkpoint_dir"]
