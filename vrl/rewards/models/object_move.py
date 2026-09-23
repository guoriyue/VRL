"""Object-move edit reward: did the named object move as instructed, and only it?

An instruction-based edit ("move the armchair to the left") is judged from the
source photo and the edited image with two released models; no region, color or
position rule is hand-written:

* **Geometry** (OWLv2 open-vocabulary detection of ``object`` in both images).
  Instances are counted only when their DINOv2 crop embedding matches the
  moved object, so other members of the category cancel out; that count must
  be unchanged -- a second copy or a lost object scores zero, which is the base
  model's dominant failure. The moved instance's box
  centre must travel along ``direction`` (left/right/up/down), or its area must
  grow/shrink for ``closer``/``farther``; lateral moves keep the area, so a crop
  or zoom does not count as a move.
* **Consistency**. Object identity is the cosine of the DINOv2 global
  embeddings of the two object crops. Background preservation is geometric:
  EfficientLoFTR matches source to edit, and the score is the share of
  background matches (outside both boxes) that stayed within a pixel-level
  tolerance, times how many background matches survived relative to matching
  the source with itself. Shifting or cropping the whole frame -- which also
  "moves" the object -- displaces every background match; a redrawn scene
  loses them. An appearance similarity is not enough: shifted patches still
  look alike (measured, GATE A round 2).

The score is ``sqrt(geometry * identity * background)``: a weighted sum would let an
unchanged source earn its full consistency term, the geometric mean gives it
zero. Every intermediate quantity is returned so evaluation can classify each
outcome (duplicated / lost / unchanged / moved).

Per-artifact metadata (from the prompt manifest)::

    reference_images: [source path]           # the edit source
    object_move: {object: "armchair", direction: "left",
                  source_box: [x0, y0, x1, y1]}   # optional, normalized
    object_move: {object: "the helicopter",       # or: a drawn target
                  target_box: [x0, y0, x1, y1]}   # normalized

``source_box`` selects which instance moves when the source holds several;
without it the most confident detection is the moved object. With
``target_box`` (a red box drawn on the source, SpatialEdit style) geometry is
the moved box's IoU with the target, which fixes both place and size; the
target region is excluded from the background check because the mark is meant
to be erased.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.base import LazyTorchModule
from vrl.rewards.models.geneval_owl import Box, Detection, _iou, nms
from vrl.rewards.models.media import artifact_middle_frame_image
from vrl.utils.artifacts import default_data_root, resolve_artifact_path

DIRECTIONS = ("left", "right", "up", "down", "closer", "farther")
# Unit image-plane vector each lateral direction asks the box centre to travel.
_LATERAL = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}


class ObjectMoveRewardModel(LazyTorchModule):
    """Score one edited image against its source and ``metadata.object_move``."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(worker_config)
        self.device = str(cfg.get("device", "cuda"))
        self.data_root = str(cfg.get("data_root") or default_data_root())
        self._detector_model = str(cfg.get("detector_model", "IDEA-Research/grounding-dino-base"))
        self._matcher_model = str(cfg.get("matcher_model", "zju-community/efficientloftr"))
        self._dino_model = str(cfg.get("dino_model", "facebook/dinov2-large"))
        self._query_template = str(cfg.get("query_template", "{obj}."))
        # Detection floor. Low on purpose: an object moved out of its usual context
        # (a bag no longer on a shoulder) loses confidence, and the DINOv2 identity
        # match, not this floor, decides what counts as the moved object.
        self._count_threshold = float(cfg.get("count_threshold", 0.15))
        self._nms_iou = float(cfg.get("nms_iou", 0.5))
        # A detection counts as the moved object when its DINOv2 crop embedding
        # is at least this similar to the source instance; other members of
        # the category (different look) are not counted in either image.
        self._identity_match = float(cfg.get("identity_match", 0.7))
        # Full displacement credit: the centre travels this fraction of the image
        # along the requested axis (a clearly visible move, not a nudge).
        self._full_shift = float(cfg.get("full_shift", 0.25))
        # A background match "stayed" when it moved less than this fraction of
        # the image diagonal (sub-pixel noise and resampling, not a shift).
        self._stay_tolerance = float(cfg.get("stay_tolerance", 0.01))
        # Full size credit for closer/farther: the area changes by this factor.
        self._full_scale = float(cfg.get("full_scale", 2.0))
        for name, value, low, high in (
            ("count_threshold", self._count_threshold, 0.0, 1.0),
            ("nms_iou", self._nms_iou, 0.0, 1.0),
            ("identity_match", self._identity_match, 0.0, 1.0),
            ("full_shift", self._full_shift, 1e-6, 1.0),
            ("stay_tolerance", self._stay_tolerance, 1e-6, 1.0),
        ):
            if not low <= value <= high:
                raise ValueError(f"object_move {name} must lie in [{low}, {high}]")
        if self._full_scale <= 1.0:
            raise ValueError("object_move full_scale must exceed 1")
        self._processor: Any | None = None

    def _load_module(self) -> Any:
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            EfficientLoFTRForKeypointMatching,
        )

        self._processor = (
            AutoProcessor.from_pretrained(self._detector_model),
            AutoImageProcessor.from_pretrained(self._dino_model),
            AutoImageProcessor.from_pretrained(self._matcher_model),
        )
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(self._detector_model).eval()
        dino = AutoModel.from_pretrained(self._dino_model).eval()
        matcher = EfficientLoFTRForKeypointMatching.from_pretrained(self._matcher_model).eval()
        return detector.to(self.device), dino.to(self.device), matcher.to(self.device)

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        spec = artifact.metadata.get("object_move")
        if not isinstance(spec, Mapping):
            raise ValueError("object_move reward needs metadata.object_move")
        references = artifact.metadata.get("reference_images") or []
        if len(references) != 1:
            raise ValueError("object_move reward needs exactly one reference image")
        from PIL import Image

        source_path = resolve_artifact_path(
            str(references[0]), data_root=self.data_root, allow_absolute=True
        )
        edited = artifact_middle_frame_image(artifact)
        with Image.open(source_path) as image:
            source = image.convert("RGB").resize(edited.size, Image.Resampling.LANCZOS)
        return self.score(source, edited, spec)

    def score(self, source: Any, edited: Any, spec: Mapping[str, Any]) -> dict[str, float]:
        """All object-move quantities for one (source, edited) pair of PIL images."""

        obj = str(spec.get("object") or "").strip()
        direction = str(spec.get("direction") or "")
        target_box = spec.get("target_box")
        if not obj or (target_box is None and direction not in DIRECTIONS):
            raise ValueError(
                f"object_move spec needs object and a direction in {DIRECTIONS} or a target_box"
            )
        if source.size != edited.size:
            raise ValueError("object_move compares images of equal size")
        width, height = edited.size
        goal = None
        if target_box is not None:
            x0, y0, x1, y1 = (float(v) for v in target_box)
            goal = (x0 * width, y0 * height, x1 * width, y1 * height)
        src_dets, edit_dets = self._detect([source, edited], obj)
        moved = _pick_source_instance(src_dets, spec.get("source_box"), (width, height))
        out = {
            "object_move": 0.0,
            "object_move_geometry": 0.0,
            "object_move_consistency": 0.0,
            "object_move_source_count": 0.0,
            "object_move_edit_count": 0.0,
            "object_move_count_ok": 0.0,
            "object_move_displacement": 0.0,
            "object_move_area_ratio": 0.0,
            "object_move_identity": 0.0,
            "object_move_background": 0.0,
        }
        if moved is None:
            return out
        # Count only instances that look like the moved object. Other members of
        # the category (parked airplanes, a second chair) appear in both images
        # and cancel; a copy of the moved object is a second match in the edit.
        crops = [_crop(source, moved[0])]
        crops += [_crop(source, det[0]) for det in src_dets]
        crops += [_crop(edited, det[0]) for det in edit_dets]
        embeds = self._embed(crops)
        sims = (embeds[1:] @ embeds[0]).tolist()
        src_sims, edit_sims = sims[: len(src_dets)], sims[len(src_dets) :]
        src_match = [
            d for d, sim in zip(src_dets, src_sims, strict=True) if sim >= self._identity_match
        ]
        edit_match = [
            (d, sim)
            for d, sim in zip(edit_dets, edit_sims, strict=True)
            if sim >= self._identity_match
        ]
        out["object_move_source_count"] = float(len(src_match))
        out["object_move_edit_count"] = float(len(edit_match))
        if not edit_match or len(edit_match) != len(src_match):
            return out
        out["object_move_count_ok"] = 1.0
        target, identity = _match_moved_instance(moved, src_match, edit_match)
        (sx0, sy0, sx1, sy1), (ex0, ey0, ex1, ey1) = moved[0], target[0]
        dx = ((ex0 + ex1) - (sx0 + sx1)) / 2.0 / width
        dy = ((ey0 + ey1) - (sy0 + sy1)) / 2.0 / height
        area_ratio = ((ex1 - ex0) * (ey1 - ey0)) / max((sx1 - sx0) * (sy1 - sy0), 1e-6)
        log_ratio = math.log(max(area_ratio, 1e-6))
        if goal is not None:
            # A drawn target ("into the red box"): overlap with it carries both
            # position and size, as in SpatialEdit-Bench's move score.
            displacement = _iou(target[0], goal) - _iou(moved[0], goal)
            progress = _iou(target[0], goal)
            size_term = 1.0
        elif direction in _LATERAL:
            ux, uy = _LATERAL[direction]
            displacement = dx * ux + dy * uy
            progress = min(max(displacement / self._full_shift, 0.0), 1.0)
            # Lateral moves keep the object's size: area halved or doubled = 0.
            size_term = max(0.0, 1.0 - abs(log_ratio) / math.log(2.0))
        else:
            sign = 1.0 if direction == "closer" else -1.0
            displacement = sign * log_ratio
            progress = min(max(displacement / math.log(self._full_scale), 0.0), 1.0)
            size_term = 1.0
        kp0, kp1 = self._match(source, edited)
        self_kp0, _ = self._match(source, source)
        # The target mark itself (red box lines) is meant to disappear.
        edited_regions = (moved[0], target[0]) if goal is None else (moved[0], target[0], goal)
        background = self._background(kp0, kp1, self_kp0, edited_regions, width, height)
        geometry = progress * size_term
        # Product, not a mean: a frame that shifted or was redrawn has no
        # background left in place, and that alone must sink the score.
        consistency = max(identity, 0.0) * background
        out.update(
            {
                "object_move": math.sqrt(geometry * consistency),
                "object_move_geometry": geometry,
                "object_move_consistency": consistency,
                "object_move_displacement": displacement,
                "object_move_area_ratio": area_ratio,
                "object_move_identity": identity,
                "object_move_background": background,
            }
        )
        return out

    def _detect(self, images: Sequence[Any], obj: str) -> list[list[Detection]]:
        import torch

        detector, _, _ = self._module_for_inference()
        processor, _, _ = self._processor
        text = self._query_template.format(obj=obj)
        found: list[list[Detection]] = []
        with torch.no_grad():
            for image in images:
                inputs = processor(images=image, text=text, return_tensors="pt").to(self.device)
                result = processor.post_process_grounded_object_detection(
                    detector(**inputs),
                    inputs.input_ids,
                    threshold=self._count_threshold,
                    text_threshold=self._count_threshold,
                    target_sizes=[image.size[::-1]],
                )[0]
                dets = [
                    (tuple(float(v) for v in box), float(score))
                    for box, score in zip(
                        result["boxes"].tolist(), result["scores"].tolist(), strict=True
                    )
                ]
                found.append(nms(dets, self._nms_iou))
        return found

    def _embed(self, images: Sequence[Any]) -> Any:
        """Unit-norm DINOv2 global (CLS) embeddings, one row per image."""

        import torch

        _, dino, _ = self._module_for_inference()
        _, dino_processor, _ = self._processor
        with torch.no_grad():
            inputs = dino_processor(images=list(images), return_tensors="pt").to(self.device)
            cls = dino(**inputs).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls.float(), dim=-1)

    def _tolerance_px(self, width: int, height: int) -> float:
        """Largest move, in pixels, that still counts as staying in place."""

        return self._stay_tolerance * math.hypot(width, height)

    def _background(
        self, kp0: Any, kp1: Any, self_kp0: Any, boxes: Sequence[Box], width: int, height: int
    ) -> float:
        """Share of the source background that is still in place in the edit."""

        import torch

        outside = _outside_boxes(kp0, boxes) & _outside_boxes(kp1, boxes)
        if not bool(outside.any()):
            return 0.0
        stayed = torch.linalg.norm(kp1 - kp0, dim=-1) <= self._tolerance_px(width, height)
        stay_share = float((stayed & outside).sum() / outside.sum())
        # Matches a redrawn scene cannot produce: compare with the source's own count.
        reference = int(_outside_boxes(self_kp0, boxes).sum())
        coverage = min(1.0, int(outside.sum()) / max(reference, 1))
        return stay_share * coverage

    def _match(self, first: Any, second: Any) -> tuple[Any, Any]:
        """EfficientLoFTR correspondences as ``[N, 2]`` pixel coordinates in each image."""

        import torch

        _, _, matcher = self._module_for_inference()
        _, _, matcher_processor = self._processor
        with torch.no_grad():
            inputs = matcher_processor(images=[[first, second]], return_tensors="pt").to(
                self.device
            )
            result = matcher_processor.post_process_keypoint_matching(
                matcher(**inputs),
                target_sizes=[[first.size[::-1], second.size[::-1]]],
                threshold=0.2,
            )[0]
        return result["keypoints0"].float().cpu(), result["keypoints1"].float().cpu()


def _pick_source_instance(
    dets: Sequence[Detection], source_box: Any, size: tuple[int, int]
) -> Detection | None:
    """The instance the instruction moves: the one ``source_box`` names, else the most confident."""

    if not dets:
        return None
    if source_box is None:
        return dets[0]
    width, height = size
    x0, y0, x1, y1 = (float(v) for v in source_box)
    named = (x0 * width, y0 * height, x1 * width, y1 * height)
    return max(dets, key=lambda det: _iou(det[0], named))


def _match_moved_instance(
    moved: Detection,
    src_match: Sequence[Detection],
    edit_match: Sequence[tuple[Detection, float]],
) -> tuple[Detection, float]:
    """The edited instance that corresponds to ``moved``, with its identity cosine.

    Look-alike instances the edit should leave alone stay put, so each claims
    its best-overlapping edited match; the moved object is the most similar
    match left over.
    """

    remaining = list(edit_match)
    for det in src_match:
        if det is moved or len(remaining) == 1:
            continue
        claimed = max(remaining, key=lambda item: _iou(item[0][0], det[0]))
        remaining.remove(claimed)
    return max(remaining, key=lambda item: item[1])


def _crop(image: Any, box: Box) -> Any:
    x0, y0, x1, y1 = (round(v) for v in box)
    return image.crop((max(0, x0), max(0, y0), max(x0 + 1, x1), max(y0 + 1, y1)))


def _outside_boxes(points: Any, boxes: Sequence[Box]) -> Any:
    """Boolean mask of ``[N, 2]`` points that lie outside every box."""

    import torch

    keep = torch.ones(points.shape[0], dtype=torch.bool)
    for x0, y0, x1, y1 in boxes:
        inside = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )
        keep = keep & ~inside
    return keep


__all__ = ["DIRECTIONS", "ObjectMoveRewardModel"]
