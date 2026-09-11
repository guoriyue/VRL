"""Clean-latents shard I/O for the GRPO diffusion-loss regularizer.

One on-disk contract: ``{target artifact -> unbatched VAE latents}`` plus the
family/model provenance needed to reject a shard encoded with a different model.
Video latents typically use ``[C, T, H, W]``; the trainer checks each target
against the rollout geometry before device transfer.
The producer is ``vrl/scripts/denoise/encode_targets.py``; the consumer is
``run_online_recipe`` (via ``data.sft_latents``) when ``algorithm.sft_weight > 0``.

Split out of ``vrl/trainers/data/artifacts.py`` (which owns prompt-manifest path
resolution + validation) because this tensor-persistence contract shares no
symbol, constant, or helper with the manifest code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vrl.trainers.data.prompts import PromptExample

SFT_LATENTS_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class CleanTargetRef:
    """Stable manifest identity and media kind for one clean target."""

    field: Literal["target_image", "target_video"]
    key: str

    @classmethod
    def from_source(cls, source: Mapping[str, object] | PromptExample) -> CleanTargetRef:
        """Build the clean-target identity shared by the shard producer and consumer.

        ``source`` is either a typed ``PromptExample`` or rollout metadata received
        across the collector boundary. Requiring exactly one field prevents a shard
        from being encoded under one identity and looked up under another.
        """

        candidates = (
            (
                ("target_image", source.get("target_image")),
                ("target_video", source.get("target_video")),
            )
            if isinstance(source, Mapping)
            else (("target_image", source.target_image), ("target_video", source.target_video))
        )
        targets: list[CleanTargetRef] = []
        for field, value in candidates:
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"clean target {field} must be a string")
            key = value.strip()
            if key:
                targets.append(cls(field=field, key=key))
        if len(targets) != 1:
            present = [target.field for target in targets]
            raise ValueError(
                "the diffusion regularizer requires exactly one clean target field "
                f"(target_image or target_video); found {present}",
            )
        return targets[0]


def save_sft_latents(
    path: str | Path,
    *,
    family: str,
    model_path: str,
    model_revision: str,
    latents_by_target: dict[str, Any],
) -> None:
    """Write the clean-latents shard the GRPO diffusion-loss regularizer reads.

    One file, one contract: ``{target artifact -> unbatched VAE latents}``
    plus the provenance needed to reject a shard encoded with a different
    family or model. The producer is
    ``vrl/scripts/denoise/encode_targets.py``.
    """

    import torch

    if not latents_by_target:
        raise ValueError("refusing to write an empty sft-latents shard")
    if any(not isinstance(target, str) or not target.strip() for target in latents_by_target):
        raise ValueError("sft-latents target keys must be non-empty strings")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SFT_LATENTS_SCHEMA_VERSION,
            "family": str(family),
            "model_path": str(model_path),
            "model_revision": str(model_revision),
            "latents": {
                target: value.detach().cpu() for target, value in latents_by_target.items()
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
    version = payload.get("schema_version")
    if type(version) is not int or version != SFT_LATENTS_SCHEMA_VERSION:
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
    if any(not isinstance(target, str) or not target.strip() for target in latents):
        raise ValueError(f"{shard_path}: sft-latents target keys must be non-empty strings")
    return latents


__all__ = [
    "SFT_LATENTS_SCHEMA_VERSION",
    "CleanTargetRef",
    "load_sft_latents",
    "save_sft_latents",
]
