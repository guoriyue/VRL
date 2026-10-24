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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MIN_LAYER_PIXELS = 64


@dataclass(frozen=True, eq=False)
class LayerStack:
    """RGBA layers ``[L, H, W, 4]`` in ``[0, 1]``, back to front, and the region each shows."""

    rgba: np.ndarray
    regions: np.ndarray

    @classmethod
    def from_rgba(cls, rgba: np.ndarray) -> LayerStack:
        """Compute each layer's visible region; layer 0 owns what no layer covers."""

        if rgba.ndim != 4 or rgba.shape[-1] != 4 or rgba.shape[0] < 2:
            raise ValueError("layers must be [L >= 2, H, W, 4]")
        solid = rgba[..., 3] >= 0.5
        front = np.zeros(rgba.shape[1:3], dtype=bool)
        regions = np.zeros_like(solid)
        for k in range(rgba.shape[0] - 1, 0, -1):
            regions[k] = solid[k] & ~front
            front |= solid[k]
        regions[0] = ~front
        return cls(rgba, regions)

    @classmethod
    def from_media(cls, media: Any) -> LayerStack:
        """The layered executor's ``[L, 4, H, W]`` (or one-sample ``[1, L, 4, H, W]``) tensor."""

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
            raise ValueError(
                f"layered reward expected [L, 4, H, W] layers, got {tuple(media.shape)}"
            )
        layers = media.detach().cpu().float()
        if media.dtype == torch.uint8 or float(layers.max()) > 1.0:
            layers = layers / 255.0
        return cls.from_rgba(layers.permute(0, 2, 3, 1).clamp(0.0, 1.0).numpy())

    @classmethod
    def from_directory(cls, root: Path, height: int, width: int) -> LayerStack:
        """``layer_<i>.png`` files back to front, resized to ``height x width``."""

        from PIL import Image

        files = sorted(root.glob("layer_*.png"), key=lambda p: int(p.stem.split("_")[1]))
        layers = []
        for file in files:
            with Image.open(file) as image:
                rgba = image.convert("RGBA")
                if rgba.size != (width, height):
                    rgba = rgba.resize((width, height), Image.Resampling.BILINEAR)
                layers.append(np.asarray(rgba, dtype=np.float32) / 255.0)
        if len(layers) < 2:
            raise ValueError(f"truth directory needs at least two layer_<i>.png files: {root}")
        return cls.from_rgba(np.stack(layers))

    def object_layers(self) -> list[int]:
        """Indices of the layers in front of the background that show anything."""

        return [
            k for k in range(1, len(self.regions)) if self.regions[k].sum() >= MIN_LAYER_PIXELS
        ]

    def composite(self) -> np.ndarray:
        """Alpha-over composite ``[H, W, 3]`` of the straight-alpha layers."""

        out = np.zeros((*self.rgba.shape[1:3], 3), dtype=np.float32)
        for layer in self.rgba:
            alpha = layer[..., 3:4]
            out = layer[..., :3] * alpha + out * (1.0 - alpha)
        return out

    def match(self, objects: Sequence[np.ndarray]) -> tuple[dict[int, int], float]:
        """Object index -> matched layer, and the soft panoptic quality of the match."""

        labeled = np.zeros(self.regions.shape[1:], dtype=bool)
        for mask in objects:
            labeled |= mask
        layers = self.object_layers()
        object_sizes = [float(mask.sum()) for mask in objects]
        layer_sizes = [float((self.regions[k] & labeled).sum()) for k in layers]
        pairs = []
        for i, mask in enumerate(objects):
            for j, k in enumerate(layers):
                inter = float((self.regions[k] & mask).sum())
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
            if j in used or size >= 0.5 * float(self.regions[k].sum())
        ]
        total = sum(map(math.sqrt, object_sizes)) + sum(map(math.sqrt, counted))
        return matches, quality / total if total > 0 else 0.0


@dataclass(frozen=True, eq=False)
class LayeredReference:
    """What a row's decomposition is scored against: its source, objects and truth layers."""

    source: np.ndarray
    objects: list[np.ndarray]
    truth: LayerStack | None = None
    inferable: np.ndarray | None = None

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, Any], height: int, width: int
    ) -> LayeredReference:
        """Read the row's reference image, ``instances`` masks and ``truth_dir`` layers.

        The source is center-cropped to a square and resized as the model's
        condition encoder does; the masks were cut from that same square.
        """

        from PIL import Image

        references = list(metadata.get("reference_images") or [])
        if len(references) != 1:
            raise ValueError(
                "layered reward needs exactly one reference image in the sample metadata"
            )
        with Image.open(Path(references[0])) as image:
            rgb = image.convert("RGB")
        side = min(rgb.size)
        left, top = (rgb.size[0] - side) // 2, (rgb.size[1] - side) // 2
        rgb = rgb.crop((left, top, left + side, top + side))
        source = np.asarray(
            rgb.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32
        )
        instances, truth_dir = metadata.get("instances"), metadata.get("truth_dir")
        if instances is None and truth_dir is None:
            raise ValueError(
                "layered reward needs metadata.instances or metadata.truth_dir on the row"
            )
        truth = inferable = None
        if truth_dir is not None:
            root = Path(str(truth_dir))
            truth = LayerStack.from_directory(root, height, width)
            inferable = np.zeros((height, width), dtype=bool)
            if (root / "inferable_mask.png").exists():
                with Image.open(root / "inferable_mask.png") as image:
                    gray = image.convert("L").resize((width, height), Image.Resampling.NEAREST)
                inferable = np.asarray(gray) > 127
        if instances is None:
            objects = list(truth.regions[1:])
        else:
            with np.load(Path(str(instances))) as data:
                masks = np.asarray(data["masks"]).astype(bool)
            if masks.ndim != 3:
                raise ValueError(f"instances file {instances} must hold masks [N, h, w]")
            objects = [
                np.asarray(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (width, height), Image.Resampling.NEAREST
                    )
                )
                > 127
                for mask in masks
            ]
        return cls(source / 255.0, [mask for mask in objects if mask.any()], truth, inferable)

    def completion(self, layers: LayerStack) -> float:
        """Agreement with the truth layers on hidden-but-inferable pixels (see the module doc)."""

        truth = self.truth
        if truth is None:
            return 0.0
        matches, _ = layers.match(list(truth.regions[1:]))
        total = weight = 0.0
        for t in range(1, len(truth.rgba)):
            hidden = (truth.rgba[t, ..., 3] >= 0.5) & ~truth.regions[t] & self.inferable
            count = float(hidden.sum())
            if count == 0:
                continue
            k = matches.get(t - 1)
            if k is not None:
                ours, theirs = layers.rgba[k], truth.rgba[t]
                rgb = np.abs(ours[..., :3] * ours[..., 3:] - theirs[..., :3] * theirs[..., 3:])
                alpha = np.abs(ours[..., 3] - theirs[..., 3])
                total += count * (
                    1.0
                    - 0.5 * float(rgb.mean(-1)[hidden].mean())
                    - 0.5 * float(alpha[hidden].mean())
                )
            weight += count
        return total / weight if weight else 0.0


class LayeredGroupingRewardModel:
    """Scores a decomposition against the row's object masks and truth layers."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self._completion_weight = float(cfg.get("completion_weight", 0.8))
        # Optional audit trail: every scored sample's layers, source and terms as
        # files, so a training run's rollouts can be inspected by eye.
        self._dump_dir = None if cfg.get("dump_dir") is None else str(cfg["dump_dir"])
        self._dumped = 0

    def score(self, layers: LayerStack, reference: LayeredReference) -> dict[str, float]:
        """The reward and its terms for ``layers`` against ``reference``."""

        if reference.source.shape != (*layers.rgba.shape[1:3], 3):
            raise ValueError("source must be [H, W, 3] matching the layers")
        _, grouping = layers.match(reference.objects)
        terms = {
            "grouping": grouping,
            "completion": reference.completion(layers),
            "fidelity": max(
                0.0, 1.0 - float(np.abs(layers.composite() - reference.source).mean())
            ),
            "objects": float(len(reference.objects)),
            "layers": float(len(layers.object_layers())),
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

        path = str(artifact.path or "")
        media = (
            torch.load(path, map_location="cpu", weights_only=True)
            if path.endswith(".pt")
            else artifact.as_media()
        )
        layers = LayerStack.from_media(media)
        height, width = layers.rgba.shape[1:3]
        reference = LayeredReference.from_metadata(dict(artifact.metadata or {}), height, width)
        scores = self.score(layers, reference)
        if self._dump_dir is not None:
            self._dump(artifact, layers, reference, scores)
        return scores

    def _dump(
        self,
        artifact: Any,
        layers: LayerStack,
        reference: LayeredReference,
        scores: Mapping[str, float],
    ) -> None:
        from PIL import Image

        # A reward parked by reload is rebuilt per scoring phase, so the counter
        # alone would restart; the wall clock keeps the directories in order.
        sample_id = str(getattr(artifact, "sample_id", "sample")).replace("/", "_")
        out = Path(self._dump_dir) / f"{int(time.time())}_{self._dumped:03d}_{sample_id}"
        out.mkdir(parents=True, exist_ok=True)
        self._dumped += 1
        Image.fromarray((reference.source * 255).round().astype(np.uint8)).save(out / "source.png")
        for index, layer in enumerate(layers.rgba):
            Image.fromarray((layer * 255).round().astype(np.uint8), "RGBA").save(
                out / f"layer_{index}.png"
            )
        (out / "scores.json").write_text(json.dumps(dict(scores), indent=1) + "\n")


__all__ = ["MIN_LAYER_PIXELS", "LayerStack", "LayeredGroupingRewardModel", "LayeredReference"]
