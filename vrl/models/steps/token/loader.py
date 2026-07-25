"""Checkpoint-directory resolution for AR families.

Local paths pass through untouched; hub ids are snapshot-downloaded once and
reused. Families that assemble their own upstream pipeline (nextstep_1) need a
directory, not a loaded module, and the AR replay cores
(:class:`vrl.models.steps.token.base.ARReplayCore`) point their strict weight
load at the same resolved directory.
"""

from __future__ import annotations


def resolve_hf_checkpoint_dir(
    model_path: str,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
) -> str:
    """Local dir passthrough, else the HF snapshot dir (downloading if needed)."""
    import os

    if os.path.isdir(model_path):
        base = model_path
    else:
        from huggingface_hub import snapshot_download

        base = snapshot_download(model_path, revision=revision)
    return os.path.join(base, subfolder) if subfolder else base


__all__ = ["resolve_hf_checkpoint_dir"]
