"""Layer-decomposition reward: one object per layer, completed behind occluders.

A decomposition is ``layers`` RGBA frames back to front (frame 0 is the
background) of one source image. Each manifest row names its reference: the
object masks visible on the source (``metadata["instances"]``, an ``.npz`` of
``masks [N, h, w]`` cut from the same center square: COCO panoptic segments,
RevealLayer masks, an offline SAM 3 pass) and, for rendered or synthetically
occluded scenes, a directory of truth layers (``metadata["truth_dir"]``). A row
with truth layers but no masks takes the truth layers' visible regions as its
objects.

The visible region of a layer is what it shows once the layers in front of it
are composited over it. Object layers (every layer but the background, ignoring
layers that show fewer than ``MIN_LAYER_PIXELS``) are matched one to one with
the objects, greedily by descending IoU of their visible regions. Three terms:

- ``grouping``: soft panoptic quality, ``sum((w_obj + w_layer) * IoU)`` over the
  matched pairs divided by the total weight of all objects and all object
  layers, each weighted by the square root of its pixel area. One object in
  one layer scores 1; a split leaves an unmatched layer, a merge or an object
  left in the background leaves an unmatched object, and an impure layer
  lowers its IoU. Empty layers count nowhere: the layer budget is arbitrary.
  Pixels outside every object are void, as in COCO's panoptic evaluation: a
  layer's IoU ignores its void pixels, and an unmatched layer that is mostly
  void (an unannotated object, a patch of wall) is not counted against the
  sample, so separating what the masks do not name costs nothing. Unlike
  thresholded PQ the term stays continuous, so samples that all split the
  same object still differ by how much of it the best layer holds.
- ``completion`` (rows with truth layers only): for every truth object, the
  agreement (1 - mean L1 of premultiplied RGB and of alpha) of its matched
  layer with the truth on the object's hidden pixels that ``inferable_mask.png``
  marks, weighted by those pixels; an unmatched object scores 0.
- ``fidelity``: 1 - mean absolute RGB error of the alpha-over composite against
  the source, so the layers must still rebuild the image.

The reward is ``grouping + fidelity + completion_weight * completion``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

MIN_LAYER_PIXELS = 64


def visible_regions(alphas: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Boolean ``[L, H, W]`` visible region per layer; layer 0 owns the rest."""

    solid = alphas >= threshold
    front = np.zeros(alphas.shape[1:], dtype=bool)
    regions = np.zeros_like(solid)
    for k in range(alphas.shape[0] - 1, 0, -1):
        regions[k] = solid[k] & ~front
        front |= solid[k]
    regions[0] = ~front
    return regions


def alpha_over(layers: np.ndarray) -> np.ndarray:
    """Composite ``[L, H, W, 4]`` straight-alpha layers back to front into ``[H, W, 3]``."""

    out = np.zeros((*layers.shape[1:3], 3), dtype=np.float32)
    for layer in layers:
        a = layer[..., 3:4]
        out = layer[..., :3] * a + out * (1.0 - a)
    return out


def match_objects(
    regions: np.ndarray, objects: Sequence[np.ndarray]
) -> tuple[dict[int, int], float]:
    """Object index -> matched layer, and the soft panoptic quality of the match."""

    labeled = np.zeros(regions.shape[1:], dtype=bool)
    for mask in objects:
        labeled |= mask
    layers = [k for k in range(1, regions.shape[0]) if regions[k].sum() >= MIN_LAYER_PIXELS]
    object_sizes = [float(mask.sum()) for mask in objects]
    layer_sizes = [float((regions[k] & labeled).sum()) for k in layers]
    pairs = []
    for i, mask in enumerate(objects):
        for j, k in enumerate(layers):
            inter = float((regions[k] & mask).sum())
            if inter > 0:
                pairs.append((inter / (object_sizes[i] + layer_sizes[j] - inter), i, j))
    matches: dict[int, int] = {}
    used: set[int] = set()
    quality = 0.0
    for iou, i, j in sorted(pairs, reverse=True):
        if i in matches or j in used:
            continue
        matches[i] = layers[j]
        used.add(j)
        quality += (math.sqrt(object_sizes[i]) + math.sqrt(layer_sizes[j])) * iou
    counted = [
        size
        for j, (k, size) in enumerate(zip(layers, layer_sizes, strict=True))
        if j in used or size >= 0.5 * float(regions[k].sum())
    ]
    total = sum(map(math.sqrt, object_sizes)) + sum(map(math.sqrt, counted))
    return matches, quality / total if total > 0 else 0.0


def completion(layers: np.ndarray, truth: np.ndarray, inferable: np.ndarray) -> float:
    """Agreement with the truth layers on hidden-but-inferable pixels (see the module doc)."""

    truth_regions = visible_regions(truth[..., 3])
    matches, _ = match_objects(visible_regions(layers[..., 3]), list(truth_regions[1:]))
    total = weight = 0.0
    for t in range(1, truth.shape[0]):
        hidden = (truth[t, ..., 3] >= 0.5) & ~truth_regions[t] & inferable
        count = float(hidden.sum())
        if count == 0:
            continue
        k = matches.get(t - 1)
        if k is not None:
            rgb = np.abs(
                layers[k, ..., :3] * layers[k, ..., 3:] - truth[t, ..., :3] * truth[t, ..., 3:]
            )
            alpha = np.abs(layers[k, ..., 3] - truth[t, ..., 3])
            total += count * (
                1.0 - 0.5 * float(rgb.mean(-1)[hidden].mean()) - 0.5 * float(alpha[hidden].mean())
            )
        weight += count
    return total / weight if weight else 0.0


def layers_from_media(media: Any) -> np.ndarray:
    """``[L, 4, H, W]`` (or a one-sample ``[1, L, 4, H, W]``) tensor to ``[L, H, W, 4]`` floats."""

    import torch

    if isinstance(media, np.ndarray):
        media = torch.from_numpy(media)
    if not isinstance(media, torch.Tensor):
        raise TypeError(f"layered reward expected a layer tensor, got {type(media)}")
    if media.ndim == 5:
        if media.shape[0] != 1:
            raise ValueError("layered reward scores one sample at a time")
        media = media[0]
    if media.ndim != 4 or media.shape[1] != 4:
        raise ValueError(f"layered reward expected [L, 4, H, W] layers, got {tuple(media.shape)}")
    layers = media.detach().cpu().float()
    if layers.dtype == torch.uint8 or float(layers.max()) > 1.0:
        layers = layers / 255.0
    return layers.permute(0, 2, 3, 1).clamp(0.0, 1.0).numpy()


def square_resize(image: Any, height: int, width: int) -> np.ndarray:
    """Center-crop a PIL image to a square and resize to ``[height, width, 3]`` in ``[0, 1]``."""

    from PIL import Image

    rgb = image.convert("RGB")
    w, h = rgb.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    rgb = rgb.crop((left, top, left + side, top + side)).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    return np.asarray(rgb, dtype=np.float32) / 255.0


def load_instances(path: str | Path, height: int, width: int) -> list[np.ndarray]:
    """Non-empty boolean ``[height, width]`` masks from an ``.npz`` holding ``masks [N, h, w]``."""

    from PIL import Image

    with np.load(Path(path)) as data:
        masks = np.asarray(data["masks"]).astype(bool)
    if masks.ndim != 3:
        raise ValueError(f"instances file {path} must hold masks [N, h, w], got {masks.shape}")
    found = []
    for mask in masks:
        if mask.shape != (height, width):
            image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 127
        if mask.any():
            found.append(mask)
    return found


def load_truth(path: str | Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Truth layers ``[L, H, W, 4]`` (``layer_<i>.png`` back to front) and the inferable mask."""

    from PIL import Image

    root = Path(path)
    files = sorted(root.glob("layer_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if len(files) < 2:
        raise ValueError(f"truth directory needs at least two layer_<i>.png files: {root}")
    layers = []
    for file in files:
        with Image.open(file) as image:
            rgba = image.convert("RGBA")
            if rgba.size != (width, height):
                rgba = rgba.resize((width, height), Image.Resampling.BILINEAR)
            layers.append(np.asarray(rgba, dtype=np.float32) / 255.0)
    inferable = np.zeros((height, width), dtype=bool)
    mask_file = root / "inferable_mask.png"
    if mask_file.exists():
        with Image.open(mask_file) as image:
            gray = image.convert("L")
            if gray.size != (width, height):
                gray = gray.resize((width, height), Image.Resampling.NEAREST)
            inferable = np.asarray(gray) > 127
    return np.stack(layers), inferable


class LayeredGroupingRewardModel:
    """Scores a decomposition against the row's object masks and truth layers."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self._completion_weight = float(cfg.get("completion_weight", 0.8))
        # Optional audit trail: every scored sample's layers, source and terms as
        # files, so a training run's rollouts can be inspected by eye.
        self._dump_dir = None if cfg.get("dump_dir") is None else str(cfg["dump_dir"])
        self._dumped = 0

    def score_layers(
        self,
        layers: np.ndarray,
        source: np.ndarray,
        objects: Sequence[np.ndarray] | None,
        *,
        truth: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Score ``[L, H, W, 4]`` layers in ``[0, 1]`` against an ``[H, W, 3]`` source.

        ``objects`` are the source's visible object masks; ``None`` takes them
        from ``truth`` (layers and inferable mask, see ``load_truth``), which
        also scores ``completion``.
        """

        if layers.ndim != 4 or layers.shape[-1] != 4 or layers.shape[0] < 2:
            raise ValueError("layers must be [L >= 2, H, W, 4]")
        if source.shape != (*layers.shape[1:3], 3):
            raise ValueError("source must be [H, W, 3] matching the layers")
        if objects is None:
            if truth is None:
                raise ValueError("layered reward needs object masks or truth layers")
            objects = list(visible_regions(truth[0][..., 3])[1:])
        objects = [mask for mask in objects if mask.any()]
        regions = visible_regions(layers[..., 3])
        _, grouping = match_objects(regions, objects)
        terms = {
            "grouping": grouping,
            "completion": 0.0 if truth is None else completion(layers, *truth),
            "fidelity": max(0.0, 1.0 - float(np.abs(alpha_over(layers) - source).mean())),
            "objects": float(len(objects)),
            "layers": float(
                sum(regions[k].sum() >= MIN_LAYER_PIXELS for k in range(1, len(regions)))
            ),
        }
        total = (
            terms["grouping"] + terms["fidelity"] + self._completion_weight * terms["completion"]
        )
        if not math.isfinite(total):
            raise ValueError("layered_grouping produced a non-finite reward")
        return {"layered_grouping": float(total), **terms}

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        return [self(artifact) for artifact in artifacts]

    def __call__(self, artifact: Any) -> dict[str, float]:
        import torch
        from PIL import Image

        path = str(artifact.path or "")
        media = (
            torch.load(path, map_location="cpu", weights_only=True)
            if path.endswith(".pt")
            else None
        )
        layers = layers_from_media(artifact.as_media() if media is None else media)
        height, width = layers.shape[1:3]
        metadata = dict(artifact.metadata or {})
        references = list(metadata.get("reference_images") or [])
        if len(references) != 1:
            raise ValueError(
                "layered reward needs exactly one reference image in the sample metadata"
            )
        with Image.open(Path(references[0])) as image:
            source = square_resize(image, height, width)
        instances, truth_dir = metadata.get("instances"), metadata.get("truth_dir")
        if instances is None and truth_dir is None:
            raise ValueError(
                "layered reward needs metadata.instances or metadata.truth_dir on the row"
            )
        scores = self.score_layers(
            layers,
            source,
            None if instances is None else load_instances(str(instances), height, width),
            truth=None if truth_dir is None else load_truth(str(truth_dir), height, width),
        )
        if self._dump_dir is not None:
            self._dump(artifact, layers, source, scores)
        return scores

    def _dump(
        self, artifact: Any, layers: np.ndarray, source: np.ndarray, scores: Mapping[str, float]
    ) -> None:
        from PIL import Image

        # A reward parked by reload is rebuilt per scoring phase, so the counter
        # alone would restart; the wall clock keeps the directories in order.
        sample_id = str(getattr(artifact, "sample_id", "sample")).replace("/", "_")
        out = Path(self._dump_dir) / f"{int(time.time())}_{self._dumped:03d}_{sample_id}"
        out.mkdir(parents=True, exist_ok=True)
        self._dumped += 1
        Image.fromarray((source * 255).round().astype(np.uint8)).save(out / "source.png")
        for index, layer in enumerate(layers):
            Image.fromarray((layer * 255).round().astype(np.uint8), "RGBA").save(
                out / f"layer_{index}.png"
            )
        (out / "scores.json").write_text(json.dumps(dict(scores), indent=1) + "\n")


__all__ = [
    "MIN_LAYER_PIXELS",
    "LayeredGroupingRewardModel",
    "alpha_over",
    "completion",
    "layers_from_media",
    "load_instances",
    "load_truth",
    "match_objects",
    "square_resize",
    "visible_regions",
]
