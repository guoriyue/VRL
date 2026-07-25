"""Shared AR replay-checkpoint loading.

Every AR family's replay path does the same thing: config-init a minimal
replay core (no VQ / vision tower), then load only the checkpoint weights the
core owns and fail loud on anything missing. The load walks the shard index
directly — transformers 5 removed ``load_sharded_checkpoint`` — and skips
shards that carry no replay-core keys entirely (a GLM-Image-sized checkpoint
keeps its vision/VQ shards on disk, untouched).

This existed as three per-family copies (janus_pro, emu3, glm_image) that
rotted independently: emu3's copy got the transformers-5 fix, janus_pro's
kept the deleted import and broke. One implementation, three importers.
"""

from __future__ import annotations

from typing import Any


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


def load_replay_core_checkpoint(core: Any, checkpoint_dir: str, *, owner: str) -> None:
    """Load ``core``'s weights from a sharded/single HF checkpoint, strictly.

    Only shards carrying keys the core owns are read (per the shard index);
    both safetensors and legacy ``pytorch_model.bin`` layouts are supported.
    Raises when any core key is left unloaded — a replay module silently
    running on random init is the worst possible failure mode.
    """
    import json
    import os

    from transformers.modeling_utils import load_state_dict

    core_keys = set(core.state_dict())
    loaded_keys: set[str] = set()

    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    single_names = ("model.safetensors", "pytorch_model.bin")

    index_path = next(
        (
            os.path.join(checkpoint_dir, name)
            for name in index_names
            if os.path.exists(os.path.join(checkpoint_dir, name))
        ),
        None,
    )
    if index_path is not None:
        with open(index_path) as index_file:
            weight_map = json.load(index_file)["weight_map"]
        shard_names = sorted(
            {name for key, name in weight_map.items() if key in core_keys},
        )
        for shard_name in shard_names:
            state = load_state_dict(os.path.join(checkpoint_dir, shard_name))
            core.load_state_dict(state, strict=False)
            loaded_keys.update(set(state) & core_keys)
    else:
        single_path = next(
            (
                os.path.join(checkpoint_dir, name)
                for name in single_names
                if os.path.exists(os.path.join(checkpoint_dir, name))
            ),
            None,
        )
        if single_path is None:
            raise FileNotFoundError(
                f"No supported {owner} checkpoint file found in {checkpoint_dir}",
            )
        state = load_state_dict(single_path)
        core.load_state_dict(state, strict=False)
        loaded_keys.update(set(state) & core_keys)

    missing_replay = sorted(core_keys - loaded_keys)
    if missing_replay:
        preview = ", ".join(missing_replay[:5])
        suffix = " ..." if len(missing_replay) > 5 else ""
        raise RuntimeError(
            f"{owner} replay checkpoint is missing keys: {preview}{suffix}",
        )


def load_replay_core(
    core_cls: type[Any],
    model_path: str,
    *,
    device: Any,
    dtype: Any,
    owner: str,
    revision: str | None = None,
    subfolder: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    """Config-init a replay core and strict-load its checkpoint weights.

    The shared replay-loader shape across every AR family: ``AutoConfig`` ->
    ``core_cls(config)`` -> strict ``load_replay_core_checkpoint`` -> place on
    ``device``/``dtype`` and ``eval()``. ``subfolder`` threads through both the
    config load and the checkpoint dir (GLM-Image's ``vision_language_encoder``);
    ``trust_remote_code`` covers Janus' custom config class. Families keep only
    their own pre-flight (e.g. Janus' upstream-package import check).
    """

    from transformers import AutoConfig

    config_kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    }
    # transformers defaults ``subfolder`` to "", so only pass a real one.
    if subfolder is not None:
        config_kwargs["subfolder"] = subfolder
    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    core = core_cls(config)
    load_replay_core_checkpoint(
        core,
        resolve_hf_checkpoint_dir(model_path, subfolder=subfolder, revision=revision),
        owner=owner,
    )
    return core.to(device=device, dtype=dtype).eval()


__all__ = [
    "load_replay_core",
    "load_replay_core_checkpoint",
    "resolve_hf_checkpoint_dir",
]
