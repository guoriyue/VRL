"""Clean-latents shard I/O for the GRPO diffusion-loss regularizer.

One on-disk contract: ``{target_video -> [C, T, H, W] VAE latents}`` plus the
family/model provenance needed to reject a shard encoded with a different model.
The producer is ``vrl/scripts/denoise/encode_targets.py``; the consumer is
``run_online_recipe`` (via ``data.sft_latents``) when ``algorithm.sft_weight > 0``.

Split out of ``vrl/trainers/data/artifacts.py`` (which owns prompt-manifest path
resolution + validation) because this tensor-persistence contract shares no
symbol, constant, or helper with the manifest code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SFT_LATENTS_SCHEMA_VERSION = 2


def save_sft_latents(
    path: str | Path,
    *,
    family: str,
    model_path: str,
    model_revision: str,
    latents_by_target: dict[str, Any],
) -> None:
    """Write the clean-latents shard the GRPO diffusion-loss regularizer reads.

    One file, one contract: ``{target_video -> [C, T, H, W] VAE latents}``
    plus the provenance needed to reject a shard encoded with a different
    family or model. The producer is
    ``vrl/scripts/denoise/encode_targets.py``.
    """

    import torch

    if not latents_by_target:
        raise ValueError("refusing to write an empty sft-latents shard")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SFT_LATENTS_SCHEMA_VERSION,
            "family": str(family),
            "model_path": str(model_path),
            "model_revision": str(model_revision),
            "latents": {
                str(target): value.detach().cpu() for target, value in latents_by_target.items()
            },
        },
        out,
    )


def load_sft_latents(
    path: str | Path,
    *,
    family: str | None = None,
    model_path: str | None = None,
    model_revision: str | None = None,
) -> dict[str, Any]:
    """Load clean target latents pinned to the training family and model."""

    import torch

    shard_path = Path(path).expanduser()
    if not shard_path.exists():
        raise FileNotFoundError(
            f"data.sft_latents shard not found: {shard_path} "
            "(produce it with vrl/scripts/denoise/encode_targets.py)",
        )
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "latents" not in payload:
        raise ValueError(f"{shard_path} is not an sft-latents shard")
    version = int(payload.get("schema_version", 0))
    if version != SFT_LATENTS_SCHEMA_VERSION:
        raise ValueError(
            f"{shard_path}: unsupported sft-latents schema_version={version}; "
            f"expected {SFT_LATENTS_SCHEMA_VERSION}",
        )
    if family is not None and str(payload.get("family")) != str(family):
        raise ValueError(
            f"{shard_path} was encoded for family {payload.get('family')!r}, "
            f"but this run trains {family!r} — latent spaces are not "
            "interchangeable; re-encode with the training model",
        )
    if model_path is not None and str(payload.get("model_path")) != str(model_path):
        raise ValueError(
            f"{shard_path} was encoded with model.path "
            f"{payload.get('model_path')!r}, but this run uses {model_path!r}; "
            "VAE latent normalization may differ even within one family, so "
            "re-encode with the training model",
        )
    if model_revision is not None and str(payload.get("model_revision")) != str(model_revision):
        raise ValueError(
            f"{shard_path} was encoded with model.revision "
            f"{payload.get('model_revision')!r}, but this run uses "
            f"{model_revision!r}; re-encode with the training checkpoint revision",
        )
    latents = payload["latents"]
    if not isinstance(latents, dict) or not latents:
        raise ValueError(f"{shard_path}: empty sft-latents shard")
    return latents


def merge_sft_latent_shards(
    inputs: list[str | Path],
    output: str | Path,
) -> None:
    """Merge disjoint encoder shards while preserving exact model provenance."""

    import torch

    if not inputs:
        raise ValueError("at least one SFT latent shard is required")
    merged: dict[str, Any] = {}
    provenance: dict[str, Any] | None = None
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not an SFT latent shard")
        current = {
            key: payload.get(key)
            for key in ("schema_version", "family", "model_path", "model_revision")
        }
        if provenance is None:
            provenance = current
        elif current != provenance:
            raise ValueError(
                f"{path} provenance does not match the first SFT latent shard",
            )
        latents = payload.get("latents")
        if not isinstance(latents, dict) or not latents:
            raise ValueError(f"{path}: empty SFT latent shard")
        overlap = sorted(set(merged) & set(latents))
        if overlap:
            raise ValueError(f"{path}: duplicate target keys across shards: {overlap[:5]}")
        merged.update(latents)
    assert provenance is not None
    if int(provenance["schema_version"] or 0) != SFT_LATENTS_SCHEMA_VERSION:
        raise ValueError("unsupported SFT latent shard schema version")
    save_sft_latents(
        output,
        family=str(provenance["family"]),
        model_path=str(provenance["model_path"]),
        model_revision=str(provenance["model_revision"]),
        latents_by_target=merged,
    )


__all__ = [
    "SFT_LATENTS_SCHEMA_VERSION",
    "load_sft_latents",
    "merge_sft_latent_shards",
    "save_sft_latents",
]
