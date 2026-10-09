"""Clean-latents shard I/O for the GRPO diffusion-loss regularizer.

One on-disk contract: ``{target artifact -> unbatched VAE latents}`` plus the
family/model provenance needed to reject a shard encoded with a different model.
Video latents typically use ``[C, T, H, W]``; the trainer checks each target
against the rollout geometry before device transfer.

A target may also carry a spatial weight map in latent geometry (schema 3):
the regularizer then averages the squared error under those weights, so a
recipe can concentrate supervision on, say, an occluded region or a protected
edit area. Weights are normalized to mean one at save time, which keeps the
``sft_weight`` scale identical to the unweighted term.
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

from vrl.utils.artifacts import atomic_file

if TYPE_CHECKING:
    from vrl.trainers.data.prompts import PromptExample

SFT_LATENTS_SCHEMA_VERSION = 3
# Schema 2 shards carry latents only; they load with ``weights=None``.
_READABLE_SCHEMA_VERSIONS = (2, SFT_LATENTS_SCHEMA_VERSION)


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


@dataclass(frozen=True, slots=True)
class CleanTarget:
    """One clean target: unbatched VAE latents plus an optional weight map.

    ``weights`` lives in latent geometry and must broadcast to ``latents``
    (for ``[C, T, H, W]`` latents a ``[1, 1, H, W]`` map weights every channel
    and frame alike). It is finite, non-negative and has mean one, so a
    uniform map reproduces the plain MSE term exactly.
    """

    latents: Any
    weights: Any | None = None

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.latents, torch.Tensor):
            raise TypeError("CleanTarget.latents must be a tensor")
        weights = self.weights
        if weights is None:
            return
        if not isinstance(weights, torch.Tensor):
            raise TypeError("CleanTarget.weights must be a tensor or None")
        try:
            broadcast = torch.broadcast_shapes(tuple(weights.shape), tuple(self.latents.shape))
        except RuntimeError as exc:
            raise ValueError(
                f"sft weight map shape {tuple(weights.shape)} does not broadcast to "
                f"the latent shape {tuple(self.latents.shape)}",
            ) from exc
        if broadcast != tuple(self.latents.shape):
            raise ValueError(
                f"sft weight map shape {tuple(weights.shape)} would enlarge the "
                f"latent shape {tuple(self.latents.shape)}",
            )
        if not weights.is_floating_point() or not torch.isfinite(weights).all():
            raise ValueError("sft weight map must be a finite floating-point tensor")
        if bool((weights < 0).any()):
            raise ValueError("sft weight map must be non-negative")
        mean = weights.float().mean()
        if not torch.isclose(mean, torch.ones_like(mean), rtol=0, atol=1e-4):
            raise ValueError(
                f"sft weight map must have mean 1 so sft_weight keeps its scale; "
                f"got mean {float(mean):.6f}",
            )

    @classmethod
    def coerce(cls, value: Any) -> CleanTarget:
        """Accept an already-typed target or bare latents (an unweighted target)."""

        return value if isinstance(value, cls) else cls(latents=value)


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
    ``vrl/scripts/denoise/encode_targets.py``. A value is bare latents or a
    ``CleanTarget``; only targets with a weight map add a ``weights`` entry.
    """

    import torch

    if not latents_by_target:
        raise ValueError("refusing to write an empty sft-latents shard")
    if any(not isinstance(target, str) or not target.strip() for target in latents_by_target):
        raise ValueError("sft-latents target keys must be non-empty strings")
    targets = {target: CleanTarget.coerce(value) for target, value in latents_by_target.items()}
    # Preserve writing through an existing symlink when publishing by replacement.
    out = Path(path).expanduser().resolve()
    payload = {
        "schema_version": SFT_LATENTS_SCHEMA_VERSION,
        "family": str(family),
        "model_path": str(model_path),
        "model_revision": str(model_revision),
        "latents": {target: value.latents.detach().cpu() for target, value in targets.items()},
        "weights": {
            target: value.weights.detach().cpu()
            for target, value in targets.items()
            if value.weights is not None
        },
    }
    with atomic_file(out, binary=True) as handle:
        torch.save(payload, handle)


def load_sft_latents(
    path: str | Path,
    *,
    family: str | None = None,
    model_path: str | None = None,
    model_revision: str | None = None,
) -> dict[str, CleanTarget]:
    """Load clean targets pinned to the training family and model."""

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
    if type(version) is not int or version not in _READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"{shard_path}: unsupported sft-latents schema_version={version}; "
            f"expected one of {_READABLE_SCHEMA_VERSIONS}",
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
    weights = payload.get("weights", {})
    if not isinstance(weights, dict):
        raise ValueError(f"{shard_path}: sft-latents weights must be a mapping")
    unknown = sorted(set(weights) - set(latents))
    if unknown:
        raise ValueError(f"{shard_path}: sft weight maps for unknown targets {unknown}")
    try:
        return {
            target: CleanTarget(latents=value, weights=weights.get(target))
            for target, value in latents.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{shard_path}: {exc}") from exc


__all__ = [
    "SFT_LATENTS_SCHEMA_VERSION",
    "CleanTarget",
    "CleanTargetRef",
    "load_sft_latents",
    "save_sft_latents",
]
