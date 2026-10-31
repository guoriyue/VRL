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
are composited over it. Foreground layers that overlap annotated objects are
matched one to one to maximize the weighted IoU numerator. Three terms:

- ``grouping``: soft panoptic quality, ``sum((w_obj + w_layer) * IoU)`` over the
  matched pairs divided by the total weight of all objects and all object
  layers, each weighted by the square root of its pixel area. One object in
  one layer scores 1; a split leaves an unmatched layer, a merge or an object
  left in the background leaves an unmatched object, and an impure layer
  lowers its IoU. Empty layers count nowhere: the layer budget is arbitrary.
  Pixels outside every object are void, as in COCO's panoptic evaluation: a
  layer's IoU ignores its void pixels. Every layer containing annotated pixels
  contributes its annotated area to the denominator, even if mostly void or
  smaller than ``MIN_LAYER_PIXELS``. Entirely unannotated layers cost nothing;
  padding an annotated fragment with void cannot remove its penalty. Unlike
  thresholded PQ the term stays continuous, so samples that all split the
  same object still differ by how much of it the best layer holds.
- ``completion`` (rows with truth layers only): for every truth object, the
  agreement (1 - mean L1 of premultiplied RGB and of alpha) of its matched
  layer with the truth on the object's hidden pixels that ``inferable_mask.png``
  marks, weighted by those pixels; an unmatched object scores 0.
- ``fidelity``: 1 - mean absolute RGB error of the alpha-over composite against
  the source, so the layers must still rebuild the image.

An optional ``integrity_weight * integrity`` term compares complete RGBA objects
and the clean background, including invisible and unmatched layers. It is only
valid for rows explicitly marked ``complete_layer_truth: true``; a partial
synthetic occlusion truth must not penalize other correctly separated objects.

The default reward is ``grouping + fidelity + completion_weight * completion``.

Experimental ``grouping_mode: partition`` uses reference-normalized squared
coverage and purity. ``integrity_mode: reference`` uses a fixed truth denominator
and penalizes opacity exceeding the complete truth stack, including hidden
copies. These opt-in modes leave the default recipe unchanged. The controlled
natural-object pilot sets completion_weight to zero: full truth already scores
hidden content, avoiding a second matching-dependent completion bonus.
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
from scipy.optimize import linear_sum_assignment

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
        # Eligibility and weights depend only on annotated pixels. In particular,
        # adding void pixels must never hide an unmatched fragment's penalty.
        sizes = [float((region & labeled).sum()) for region in self.regions[1:]]
        layers = [k for k, size in enumerate(sizes, start=1) if size > 0]
        if not objects or not layers:
            return {}, 0.0
        object_sizes = np.array([float(mask.sum()) for mask in objects])
        layer_sizes = np.array([sizes[k - 1] for k in layers])
        overlap = np.array(
            [[float((mask & self.regions[k]).sum()) for k in layers] for mask in objects]
        )
        iou = overlap / (object_sizes[:, None] + layer_sizes[None, :] - overlap)
        weights = np.sqrt(object_sizes[:, None]) + np.sqrt(layer_sizes[None, :])
        utility = weights * iou
        rows, columns = linear_sum_assignment(utility, maximize=True)
        matches = {
            int(i): layers[j] for i, j in zip(rows, columns, strict=True) if overlap[i, j] > 0
        }
        total = np.sqrt(object_sizes).sum() + np.sqrt(layer_sizes).sum()
        return matches, float(utility[rows, columns].sum() / total)

    def partition_quality(self, objects: Sequence[np.ndarray]) -> float:
        """Visible object coverage squared, weighted by annotated-layer purity.

        Normalize only by reference object area. Moving an isolated pure fragment
        into background cannot improve this score by shrinking a prediction
        denominator. Unknown objects and invisible copies remain unobservable.
        """
        if not objects:
            return 0.0
        labeled = np.logical_or.reduce(objects)
        areas = np.asarray([mask.sum() for mask in objects], dtype=np.float64)
        sizes = np.asarray([(mask & labeled).sum() for mask in self.regions[1:]])
        overlap = np.asarray(
            [[(mask & layer).sum() for layer in self.regions[1:]] for mask in objects],
            dtype=np.float64,
        )
        coverage = overlap / areas[:, None]
        purity = np.divide(overlap, sizes[None], out=np.zeros_like(overlap), where=sizes[None] > 0)
        return float(np.average((coverage**2 * purity).sum(axis=1), weights=np.sqrt(areas)))


@dataclass(frozen=True, eq=False)
class LayeredReference:
    """What a row's decomposition is scored against: its source, objects and truth layers."""

    source: np.ndarray
    objects: list[np.ndarray]
    truth: LayerStack | None = None
    inferable: np.ndarray | None = None
    complete_layer_truth: bool = False

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
        complete = metadata.get("complete_layer_truth") is True
        if complete and truth is None:
            raise ValueError("complete_layer_truth requires metadata.truth_dir")
        return cls(
            source / 255.0, [mask for mask in objects if mask.any()], truth, inferable, complete
        )

    def excess_foreground_opacity(self, layers: LayerStack) -> float:
        """Excess alpha mass relative to exhaustive full-layer references.

        Sum full alpha, including hidden layers. Legitimate object overlap is
        included in the target mass. Visible-only masks cannot supply this term.
        """
        if not self.complete_layer_truth:
            return 0.0
        if self.truth is None:
            raise ValueError("complete_layer_truth requires truth layers")
        target = self.truth.rgba[1:, ..., 3].sum(axis=0, dtype=np.float64)
        predicted = layers.rgba[1:, ..., 3].sum(axis=0, dtype=np.float64)
        return float(np.maximum(predicted - target, 0).sum() / max(target.sum(), 1.0))

    def integrity(self, layers: LayerStack, *, mode: str = "pq") -> tuple[float, float]:
        """Full-object quality and clean-background agreement for exhaustive truth.

        Matching uses full alpha and color, anchored to the visible footprint.
        Unmatched opaque layers remain in the denominator even when completely
        hidden. Legitimate overlap is allowed: each layer has its own full truth.
        """
        if not self.complete_layer_truth:
            return 0.0, 0.0
        if self.truth is None:
            raise ValueError("complete_layer_truth requires truth layers")
        ours, targets = layers.rgba[1:], self.truth.rgba[1:]
        our_nonempty = ours[..., 3].sum(axis=(1, 2)) > 0
        target_nonempty = targets[..., 3].sum(axis=(1, 2)) > 0
        ours, targets = ours[our_nonempty], targets[target_nonempty]
        visible_ours = layers.regions[1:][our_nonempty]
        visible_targets = self.truth.regions[1:][target_nonempty]
        object_weights = np.sqrt(targets[..., 3].sum(axis=(1, 2), dtype=np.float64))
        layer_weights = np.sqrt(ours[..., 3].sum(axis=(1, 2), dtype=np.float64))
        utility = np.zeros((len(targets), len(ours)), dtype=np.float64)
        for i, target in enumerate(targets):
            for j, layer in enumerate(ours):
                # Full masks alone can match two identical copies of a merged
                # layer to two different targets. A hidden copy must not earn
                # credit for an object visible in the reference. Soft visible
                # IoU also prevents revealing a token pixel to bypass this rule.
                visible_union = np.count_nonzero(visible_targets[i] | visible_ours[j])
                visible_iou = (
                    np.count_nonzero(visible_targets[i] & visible_ours[j]) / visible_union
                    if visible_union
                    else 1.0
                )
                if visible_iou == 0:
                    continue
                overlap = np.minimum(target[..., 3], layer[..., 3])
                intersection = float(overlap.sum(dtype=np.float64))
                if intersection == 0:
                    continue
                union = float(np.maximum(target[..., 3], layer[..., 3]).sum(dtype=np.float64))
                color_error = float(
                    (np.abs(target[..., :3] - layer[..., :3]).mean(-1) * overlap).sum(
                        dtype=np.float64
                    )
                    / intersection
                )
                utility[i, j] = (
                    (
                        object_weights[i]
                        if mode == "reference"
                        else object_weights[i] + layer_weights[j]
                    )
                    * (intersection / union)
                    * (1.0 - color_error)
                    * visible_iou
                )
        rows, columns = linear_sum_assignment(utility, maximize=True)
        denominator = (
            float(object_weights.sum())
            if mode == "reference"
            else float(object_weights.sum() + layer_weights.sum())
        )
        foreground = float(utility[rows, columns].sum() / denominator) if denominator else 1.0
        if mode == "reference":
            # A fixed truth denominator cannot reward deleting an unmatched
            # fragment. Penalize excess foreground opacity relative to the full
            # truth stack, allowing exactly the overlap of legitimate occlusion.
            target_alpha = targets[..., 3].sum(axis=0)
            predicted_alpha = ours[..., 3].sum(axis=0)
            excess = float(np.maximum(predicted_alpha - target_alpha, 0).sum(dtype=np.float64))
            mass = float(target_alpha.sum(dtype=np.float64))
            foreground *= max(0.0, 1.0 - excess / max(mass, 1.0))
        # Inspect the background behind truth foregrounds instead of diluting
        # duplicated-object residuals with the usually large unchanged room.
        covered = self.truth.rgba[1:, ..., 3].max(axis=0)
        background_error = np.abs(layers.rgba[0, ..., :3] - self.truth.rgba[0, ..., :3]).mean(-1)
        alpha_error = np.abs(layers.rgba[0, ..., 3] - self.truth.rgba[0, ..., 3])
        area = float(covered.sum(dtype=np.float64))
        background = (
            1.0
            - float(
                ((background_error + alpha_error) * 0.5 * covered).sum(dtype=np.float64) / area
            )
            if area
            else 1.0
        )
        return foreground, background

    def completion(self, layers: LayerStack) -> float:
        """Agreement with the truth layers on hidden-but-inferable pixels (see the module doc)."""

        truth = self.truth
        if truth is None or self.inferable is None:
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
        self._integrity_weight = float(cfg.get("integrity_weight", 0.0))
        self._excess_weight = float(cfg.get("excess_weight", 0.0))
        if not math.isfinite(self._excess_weight) or self._excess_weight < 0:
            raise ValueError("excess_weight must be finite and nonnegative")
        self._grouping_mode = cfg.get("grouping_mode", "pq")
        if self._grouping_mode not in {"pq", "partition"}:
            raise ValueError("grouping_mode must be pq or partition")
        self._integrity_mode = cfg.get("integrity_mode", "pq")
        if self._integrity_mode not in {"pq", "reference"}:
            raise ValueError("integrity_mode must be pq or reference")
        if not math.isfinite(self._integrity_weight) or self._integrity_weight < 0:
            raise ValueError("integrity_weight must be finite and nonnegative")
        # Optional audit trail: every scored sample's layers, source and terms as
        # files, so a training run's rollouts can be inspected by eye.
        self._dump_dir = None if cfg.get("dump_dir") is None else str(cfg["dump_dir"])
        self._dumped = 0

    def score(self, layers: LayerStack, reference: LayeredReference) -> dict[str, float]:
        """The reward and its terms for ``layers`` against ``reference``."""

        if reference.source.shape != (*layers.rgba.shape[1:3], 3):
            raise ValueError("source must be [H, W, 3] matching the layers")
        _, grouping = layers.match(reference.objects)
        if self._grouping_mode == "partition":
            grouping = layers.partition_quality(reference.objects)
        terms = {
            "grouping": grouping,
            "completion": reference.completion(layers),
            "fidelity": max(
                0.0, 1.0 - float(np.abs(layers.composite() - reference.source).mean())
            ),
            "objects": float(len(reference.objects)),
            "layers": float(len(layers.object_layers())),
        }
        foreground, background = (
            reference.integrity(layers, mode=self._integrity_mode)
            if self._integrity_weight
            else (0.0, 0.0)
        )
        terms.update(
            full_object_quality=foreground,
            clean_background=background,
            integrity=0.5 * (foreground + background),
            integrity_supervised=float(reference.complete_layer_truth),
        )
        total = (
            terms["grouping"]
            + terms["fidelity"]
            + self._completion_weight * terms["completion"]
            + self._integrity_weight * terms["integrity"]
        )
        if self._excess_weight:
            terms["excess_foreground_opacity"] = reference.excess_foreground_opacity(layers)
            total -= self._excess_weight * terms["excess_foreground_opacity"]
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
