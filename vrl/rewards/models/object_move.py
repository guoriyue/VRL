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

``object_move_shaped`` is the training signal: ``(object_move + w * background)
/ (1 + w)``, with background measured for every outcome (a copy or a lost
object included). ``object_move`` alone is zero for almost every base sample,
so GRPO pushed down every faithful-but-unmoved edit and the policy stopped
following the reference photo (run1: held-out 0.107 -> 0.001, the scene
redrawn). The shaped key ranks a correct move above a faithful no-op or copy,
and those above a redrawn scene; with a small ``w`` a no-op still earns far
less than any credited move.

Per-artifact metadata (from the prompt manifest)::

    reference_images: [source path]           # the edit source
    object_move: {object: "armchair", direction: "left",
                  source_box: [x0, y0, x1, y1]}   # optional, normalized
    object_move: {object: "the helicopter",       # or: a drawn target
                  target_box: [x0, y0, x1, y1]}   # normalized
    object_move: {object: "laptop", support: "chair",  # or: onto a support
                  support_box: [x0, y0, x1, y1],       # normalized
                  source_box: [x0, y0, x1, y1]}

``source_box`` selects which instance moves when the source holds several;
without it the most confident detection is the moved object. With
``target_box`` (a red box drawn on the source, SpatialEdit style) geometry is
the moved box's IoU with the target, which fixes both place and size; the
target region is excluded from the background check because the mark is meant
to be erased. With ``support_box`` ("put the laptop on the chair") the object
has to come to rest on a real surface in the photo: geometry is full when the
bottom centre of its edited box lies inside the support's box, and partial
for the share of the starting distance to it that was closed; its size may
change with depth, by at most about 2x.
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
        # Floor for assigning an edited detection to the moved object: below it
        # a box is detector noise, not the object (it still has to be more like
        # the moved object than like any other source instance).
        self._identity_match = float(cfg.get("identity_match", 0.5))
        # Full displacement credit: the centre travels this fraction of the image
        # along the requested axis (a clearly visible move, not a nudge).
        self._full_shift = float(cfg.get("full_shift", 0.25))
        # A background match "stayed" when it moved less than this fraction of
        # the image diagonal (sub-pixel noise and resampling, not a shift).
        self._stay_tolerance = float(cfg.get("stay_tolerance", 0.01))
        # Full size credit for closer/farther: the area changes by this factor.
        self._full_scale = float(cfg.get("full_scale", 2.0))
        # Weight of background preservation in ``object_move_shaped``.
        self._background_weight = float(cfg.get("background_weight", 0.2))
        for name, value, low, high in (
            ("count_threshold", self._count_threshold, 0.0, 1.0),
            ("nms_iou", self._nms_iou, 0.0, 1.0),
            ("identity_match", self._identity_match, 0.0, 1.0),
            ("full_shift", self._full_shift, 1e-6, 1.0),
            ("stay_tolerance", self._stay_tolerance, 1e-6, 1.0),
            ("background_weight", self._background_weight, 0.0, 1.0),
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
        support_box = spec.get("support_box")
        if not obj or (target_box is None and support_box is None and direction not in DIRECTIONS):
            raise ValueError(
                f"object_move spec needs object and a direction in {DIRECTIONS}, "
                "a target_box or a support_box"
            )
        if source.size != edited.size:
            raise ValueError("object_move compares images of equal size")
        width, height = edited.size
        goal = support = None
        if target_box is not None:
            x0, y0, x1, y1 = (float(v) for v in target_box)
            goal = (x0 * width, y0 * height, x1 * width, y1 * height)
        elif support_box is not None:
            x0, y0, x1, y1 = (float(v) for v in support_box)
            support = (x0 * width, y0 * height, x1 * width, y1 * height)
        src_dets, edit_dets = self._detect([source, edited], obj)
        moved = _pick_source_instance(src_dets, spec.get("source_box"), (width, height))
        out = {
            "object_move": 0.0,
            "object_move_shaped": 0.0,
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
        # Each edited detection belongs to the source instance it looks most
        # like (DINOv2). Other members of the category (parked airplanes, a
        # second chair) keep their own counterparts; the moved object must end
        # up with exactly one detection -- two is a pasted copy, none is lost.
        # Similarity is relative, not an absolute bar: a correctly moved object
        # that was also resized loses sharpness and scores lower against the
        # source crop, but still more like itself than like anything else.
        embeds = self._embed(
            [_crop(source, det[0]) for det in src_dets]
            + [_crop(edited, det[0]) for det in edit_dets]
        )
        source_embeds, edit_embeds = embeds[: len(src_dets)], embeds[len(src_dets) :]
        moved_index = src_dets.index(moved)
        assigned: list[tuple[Detection, float]] = []
        if len(edit_dets):
            sims = edit_embeds @ source_embeds.T
            for det, row in zip(edit_dets, sims.tolist(), strict=True):
                best = max(range(len(row)), key=row.__getitem__)
                if best == moved_index and row[best] >= self._identity_match:
                    assigned.append((det, row[best]))
        out["object_move_source_count"] = 1.0
        out["object_move_edit_count"] = float(len(assigned))
        kp0, kp1 = self._match(source, edited)
        self_kp0, _ = self._match(source, source)
        if len(assigned) != 1:
            # A copy or a lost object: no move credit, but a scene kept in
            # place still ranks above a redrawn one.
            regions = (moved[0], *(det[0] for det, _ in assigned))
            if goal is not None:
                regions = (*regions, goal)
            background = self._background(kp0, kp1, self_kp0, regions, width, height)
            out["object_move_background"] = background
            out["object_move_shaped"] = self._shaped(0.0, background)
            return out
        out["object_move_count_ok"] = 1.0
        target, identity = assigned[0]
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
        elif support is not None:
            # Resting point: the bottom centre of the object's box has to land on
            # the support (a laptop set on the chair, not hanging beside it).
            # Credit is the share of the starting gap that was closed, so an
            # object left where it was earns nothing however near it started.
            start_gap = _gap(((sx0 + sx1) / 2.0, sy1), support, width, height)
            end_gap = _gap(((ex0 + ex1) / 2.0, ey1), support, width, height)
            displacement = start_gap - end_gap
            progress = min(max(displacement / max(start_gap, 1e-6), 0.0), 1.0)
            # Depth may shrink or grow it a little; halved or doubled area = 0.
            size_term = max(0.0, 1.0 - abs(log_ratio) / math.log(2.0))
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
                "object_move_shaped": self._shaped(math.sqrt(geometry * consistency), background),
                "object_move_geometry": geometry,
                "object_move_consistency": consistency,
                "object_move_displacement": displacement,
                "object_move_area_ratio": area_ratio,
                "object_move_identity": identity,
                "object_move_background": background,
            }
        )
        return out

    def _shaped(self, move: float, background: float) -> float:
        return (move + self._background_weight * background) / (1.0 + self._background_weight)

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


def _gap(point: tuple[float, float], box: Box, width: int, height: int) -> float:
    """Distance from ``point`` to ``box`` (zero inside it), in frame fractions."""

    x, y = point
    gap_x = max(box[0] - x, 0.0, x - box[2]) / width
    gap_y = max(box[1] - y, 0.0, y - box[3]) / height
    return math.hypot(gap_x, gap_y)


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
