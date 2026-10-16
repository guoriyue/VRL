"""Cached layered scenes: the on-disk contract and the dataset the SFT trainer reads.

A layered scene is a directory with ``source.png`` (the photo or render), one
``layer_<i>.png`` RGBA per layer back to front and, for rendered scenes, an
``inferable_mask.png`` marking hidden content that can be deduced from the
image. ``vrl/scripts/denoise/encode_layered_scenes.py`` turns each scene into
``cache.pt`` with the family model that will train on it:

- ``condition_latents``, ``prompt_embeds``, ``prompt_embeds_mask``: what the
  layered forward conditions on (the packed source tokens, the fixed prompt);
- ``target_latents``: the packed ``layers + 1`` frames (source composite, then
  every layer), the clean SFT target; absent for a truth-free photo;
- ``weights_weighted`` / ``weights_control``: mean-one token weights. The
  weighted map raises the tokens of each layer's inferable hidden pixels so
  they carry ``INFERABLE_SHARE`` of the loss; unknowable pixels are never
  raised, so memorizing them earns nothing. The control map is all ones.

``empty_frame.pt`` beside the splits holds the packed latent of one fully
transparent frame: the target of the spare layers ``LayeredSceneDataset``
appends so unused layers learn to stay empty.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from vrl.trainers.offline.sft import SFTBatch

LAYERED_SCENE_CACHE = "cache.pt"
EMPTY_FRAME_FILE = "empty_frame.pt"
LAYERED_SCENE_SCHEMA_VERSION = 1
# Pixels per packed token along each axis: the 8x VAE and the 2x2 patch.
TOKEN_PIXELS = 16
INFERABLE_SHARE = 0.25
# Upper bound on how much a hidden token can be raised, so a scene with a tiny
# inferable region does not put the whole share on a handful of tokens.
MAX_RAISE = 80.0

# The conditioning keys the layered forward reads from a cached scene.
CONDITIONING_KEYS = (
    "condition_latents",
    "prompt_embeds",
    "prompt_embeds_mask",
    "layers",
    "height",
    "width",
)


def token_weights(
    scene: Path,
    layer_count: int,
    *,
    height: int,
    width: int,
    share: float = INFERABLE_SHARE,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean-one token weights that give inferable hidden pixels ``share`` of the loss.

    A layer's pixel is hidden when a layer above it covers it; it is raised only
    where ``inferable_mask.png`` marks the hidden content as deducible. Returns
    ``[1, (layers + 1) * tokens, 1]`` in the packed frame order (the source
    composite first) and the raise statistics.
    """

    grid = (height // TOKEN_PIXELS, width // TOKEN_PIXELS)
    tokens = grid[0] * grid[1]
    frames = layer_count + 1  # composite frame + every layer
    alphas = [
        np.asarray(Image.open(scene / f"layer_{i}.png").convert("RGBA"))[..., 3] > 127
        for i in range(layer_count)
    ]
    inferable = np.asarray(Image.open(scene / "inferable_mask.png").convert("L")) > 127
    coverage = torch.zeros(frames, tokens)
    for layer in range(layer_count):
        front = np.zeros_like(inferable)
        for above in alphas[layer + 1 :]:
            front |= above
        region = torch.tensor(alphas[layer] & front & inferable, dtype=torch.float32)[None, None]
        coverage[layer + 1] = F.interpolate(region, size=grid, mode="area")[0, 0].reshape(-1)
    total = frames * tokens
    mass = float(coverage.sum())
    k = min(MAX_RAISE, share * total / ((1.0 - share) * mass)) if mass > 0 else 0.0
    weights = 1.0 + k * coverage
    weights = weights / weights.mean()
    return weights.reshape(1, total, 1), {
        "k": k,
        "inferable_token_mass": mass,
        "inferable_share": float((k * coverage).sum() / (total + k * mass)) if mass else 0.0,
    }


def append_spare_layers(
    clean: torch.Tensor, weights: torch.Tensor, empty: torch.Tensor, spare: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``spare`` transparent target frames on top and renormalize to mean one.

    Spare layers above the scene's own layers must stay empty, so their target is
    the encoded transparent frame and every spare token weighs one before the
    renormalization.
    """

    if not spare:
        return clean, weights
    clean = torch.cat([clean] + [empty.to(clean.device, clean.dtype)] * spare, dim=1)
    weights = torch.cat(
        [weights, torch.ones(1, spare * empty.shape[1], 1, device=weights.device)], dim=1
    )
    return clean, weights / weights.mean()


def load_layered_scene(
    path: Path, *, family: str | None = None, model_revision: str | None = None
) -> dict[str, Any]:
    """Read one ``cache.pt`` and pin it to the training family and model revision."""

    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict) or cache.get("schema") != LAYERED_SCENE_SCHEMA_VERSION:
        raise ValueError(
            f"{path} is not a layered scene cache of schema {LAYERED_SCENE_SCHEMA_VERSION}; "
            "re-encode it with vrl/scripts/denoise/encode_layered_scenes.py",
        )
    if family is not None and cache.get("family") != family:
        raise ValueError(f"{path} was encoded for family {cache.get('family')!r}, not {family!r}")
    if model_revision is not None and cache.get("model_revision") != model_revision:
        raise ValueError(
            f"{path} was encoded with model.revision {cache.get('model_revision')!r}, "
            f"but this run uses {model_revision!r}; re-encode with the training checkpoint",
        )
    return cache


def scene_cache_paths(roots: Mapping[str, int | None], split: str) -> list[Path]:
    """Every cached scene of ``split`` under each root, the first ``count`` of a root when set."""

    paths: list[Path] = []
    for root, count in roots.items():
        found = sorted(Path(root).expanduser().glob(f"{split}/*/{LAYERED_SCENE_CACHE}"))
        if not found:
            raise FileNotFoundError(f"no cached scenes under {root}/{split}")
        paths.extend(found if count is None else found[:count])
    return paths


class LayeredSceneDataset(Dataset):
    """Cached layered scenes as ``SFTBatch`` samples of one scene each.

    ``token_weights`` picks ``weighted`` or ``control``. ``spare_layers_max`` >
    0 appends ``0..spare_layers_max`` transparent spare layers per draw, so the
    layer budget the model sees varies above the scene's own count.
    """

    def __init__(
        self,
        roots: Mapping[str, int | None],
        *,
        split: str,
        token_weights: str = "control",
        spare_layers_max: int = 0,
        family: str | None = None,
        model_revision: str | None = None,
    ) -> None:
        if token_weights not in ("weighted", "control"):
            raise ValueError(
                f"token_weights must be 'weighted' or 'control', got {token_weights!r}"
            )
        if spare_layers_max < 0:
            raise ValueError("spare_layers_max must be >= 0")
        self.paths = scene_cache_paths(roots, split)
        self.weight_key = f"weights_{token_weights}"
        self.spare_layers_max = int(spare_layers_max)
        self.family = family
        self.model_revision = model_revision
        self.empty: torch.Tensor | None = None
        if self.spare_layers_max:
            empty_path = Path(next(iter(roots))).expanduser() / EMPTY_FRAME_FILE
            if not empty_path.exists():
                raise FileNotFoundError(
                    f"spare layers need {empty_path} (written by encode_layered_scenes)",
                )
            empty = torch.load(empty_path, map_location="cpu", weights_only=True)
            if model_revision is not None and empty.get("model_revision") != model_revision:
                raise ValueError(f"{empty_path} was encoded with another model revision")
            self.empty = empty["latent"].float()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> SFTBatch:
        cache = load_layered_scene(
            self.paths[index], family=self.family, model_revision=self.model_revision
        )
        if "target_latents" not in cache:
            raise ValueError(f"{self.paths[index]} carries no layer truth; it cannot train")
        clean = cache["target_latents"].float()
        weights = cache[self.weight_key].float()
        conditioning = {key: cache[key] for key in CONDITIONING_KEYS}
        spare = (
            int(torch.randint(0, self.spare_layers_max + 1, (1,))) if self.empty is not None else 0
        )
        clean, weights = append_spare_layers(clean, weights, self.empty, spare)
        conditioning["layers"] = int(cache["layers"]) + spare
        conditioning["scene"] = self.paths[index].parent.name
        return SFTBatch(latents=clean, conditioning=conditioning, weights=weights)


def collate_layered_scenes(items: list[SFTBatch]) -> SFTBatch:
    """Batch scenes of one token count; the layered forward takes one layer budget per call."""

    first = items[0]
    if len(items) == 1:
        return first
    layers = first.conditioning["layers"]
    if any(item.conditioning["layers"] != layers for item in items):
        raise ValueError(
            "a layered scene batch must share one layer count; use actor.train_batch_size=1 "
            "with mixed layer counts",
        )
    conditioning = {
        key: (
            torch.cat([item.conditioning[key] for item in items], dim=0)
            if isinstance(first.conditioning[key], torch.Tensor)
            else first.conditioning[key]
        )
        for key in CONDITIONING_KEYS
    }
    conditioning["scene"] = ",".join(item.conditioning["scene"] for item in items)
    return SFTBatch(
        latents=torch.cat([item.latents for item in items], dim=0),
        conditioning=conditioning,
        weights=torch.cat([item.weights for item in items], dim=0),
    )


__all__ = [
    "CONDITIONING_KEYS",
    "EMPTY_FRAME_FILE",
    "INFERABLE_SHARE",
    "LAYERED_SCENE_CACHE",
    "LAYERED_SCENE_SCHEMA_VERSION",
    "LayeredSceneDataset",
    "append_spare_layers",
    "collate_layered_scenes",
    "load_layered_scene",
    "scene_cache_paths",
    "token_weights",
]
