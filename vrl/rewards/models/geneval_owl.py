"""GenEval compositional reward scored in-process with OWLv2 + CLIP.

Re-implements the official GenEval decision rules (djghosh13/geneval) over an
open-vocabulary detector instead of the COCO Mask2Former the official
evaluator needs (mmdet stack, absent here; Flow-GRPO reaches it through a
separate conda env behind an HTTP reward server):

* ``include`` — at least ``count`` detections of ``class``; an optional
  ``color`` (CLIP zero-shot over the detection crop) and an optional
  ``position`` relative to an earlier include item (box-centre offset with the
  official 0.1 size margin);
* ``exclude`` — fewer than ``count`` detections of ``class``.

``geneval_owl_strict`` is 1 only when every condition holds — the official
GenEval score. ``geneval_owl`` is the satisfied-condition fraction, a graded
training signal that keeps groups from collapsing to all-zero advantages
(Flow-GRPO's reward server likewise trains on a partial score and reports the
strict one). Same image, same score: there is no sampling anywhere. Absolute
numbers are not comparable to published GenEval tables because the detector
differs; before/after comparisons on one detector are.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vrl.rewards.models.base import LazyTorchModule
from vrl.rewards.models.media import artifact_middle_frame_image

# Official GenEval colour vocabulary (evaluation/evaluate_images.py).
GENEVAL_COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
)

Box = tuple[float, float, float, float]
Detection = tuple[Box, float]
Detector = Callable[[list[Any], list[str]], Sequence[Mapping[str, Sequence[Detection]]]]
ColorClassifier = Callable[[Any, Box, str], str]


@dataclass(frozen=True, slots=True)
class GenEvalVerdict:
    """One image's GenEval outcome: strict pass, graded score, and why."""

    strict: float
    partial: float
    why: str

    @property
    def scores(self) -> dict[str, float]:
        return {"geneval_owl": self.partial, "geneval_owl_strict": self.strict}


class GenEvalOwlRewardModel(LazyTorchModule):
    """Per-artifact GenEval verdict from ``metadata.geneval`` with OWLv2 + CLIP."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(worker_config)
        self.device = str(cfg.get("device", "cuda"))
        self._detector_model = str(cfg.get("detector_model", "google/owlv2-base-patch16-ensemble"))
        self._clip_model = str(cfg.get("clip_model", "openai/clip-vit-large-patch14"))
        self._detection_threshold = float(cfg.get("detection_threshold", 0.15))
        self._nms_iou = float(cfg.get("nms_iou", 0.5))
        self._position_threshold = float(cfg.get("position_threshold", 0.1))
        for name, value in (
            ("detection_threshold", self._detection_threshold),
            ("nms_iou", self._nms_iou),
            ("position_threshold", self._position_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must satisfy 0.0 <= {name} <= 1.0")
        self._query_template = str(cfg.get("query_template", "a photo of a {cls}"))
        self._color_template = str(cfg.get("color_template", "a photo of a {color} {cls}"))
        self._metadata_key = str(cfg.get("metadata_key", "geneval"))
        if not self._metadata_key:
            raise ValueError("geneval_owl metadata_key must be non-empty")
        # Test seams: callables replace the detector / colour classifier.
        self._detector: Detector | None = cfg.get("detector")
        self._color_fn: ColorClassifier | None = cfg.get("color_fn")
        self._processors: tuple[Any, Any] | None = None

    # -- RewardModel protocol ----------------------------------------------

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        specs = [self._spec(artifact) for artifact in artifacts]
        images = [artifact_middle_frame_image(artifact) for artifact in artifacts]
        return [verdict.scores for verdict in self.judge_images(images, specs)]

    def __call__(self, artifact: Any) -> dict[str, float]:
        return self.score_batch([artifact])[0]

    def judge_images(
        self, images: Sequence[Any], specs: Sequence[Mapping[str, Any]]
    ) -> list[GenEvalVerdict]:
        """GenEval verdicts for PIL images paired with their prompt specs."""

        if len(images) != len(specs):
            raise ValueError("judge_images needs one spec per image")
        return [self.judge(image, spec) for image, spec in zip(images, specs, strict=True)]

    def judge(self, image: Any, spec: Mapping[str, Any]) -> GenEvalVerdict:
        include = list(spec.get("include") or [])
        exclude = list(spec.get("exclude") or [])
        if not include and not exclude:
            raise ValueError("geneval spec carries neither include nor exclude items")
        classes = sorted({str(item["class"]) for item in include + exclude})
        detections = self._detect([image], classes)[0]
        satisfied = 0
        total = 0
        failures: list[str] = []
        # Reference boxes for position items, indexed like ``include``; None
        # when that item found nothing.
        matched: list[Box | None] = []
        for item in include:
            cls = str(item["class"])
            count = int(item.get("count", 1))
            objs = list(detections.get(cls, ()))
            total += 1
            if len(objs) >= count:
                satisfied += 1
            else:
                failures.append(f"missing:{cls}")
            if "color" in item:
                total += 1
                wanted = str(item["color"])
                colored = [obj for obj in objs if self._color_of(image, obj[0], cls) == wanted]
                if len(colored) >= count:
                    satisfied += 1
                    objs = colored
                else:
                    failures.append(f"color:{cls}!={wanted}")
            if "position" in item:
                total += 1
                relation, ref_index = item["position"]
                reference = matched[int(ref_index)] if int(ref_index) < len(matched) else None
                if reference is not None and any(
                    str(relation) in relative_position(obj[0], reference, self._position_threshold)
                    for obj in objs
                ):
                    satisfied += 1
                else:
                    failures.append(f"position:{relation}")
            matched.append(objs[0][0] if objs else None)
        for item in exclude:
            cls = str(item["class"])
            count = int(item.get("count", 1))
            total += 1
            if len(detections.get(cls, ())) < count:
                satisfied += 1
            else:
                failures.append(f"exclude:{cls}>={count}")
        return GenEvalVerdict(
            strict=0.0 if failures else 1.0,
            partial=satisfied / total,
            why=";".join(failures) or "ok",
        )

    # -- detector / colour backends ----------------------------------------

    def _spec(self, artifact: Any) -> Mapping[str, Any]:
        raw = artifact.metadata.get(self._metadata_key)
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"geneval_owl requires metadata[{self._metadata_key!r}] to be a GenEval "
                f"spec mapping, got {type(raw).__name__}"
            )
        return raw

    def _load_module(self) -> Any:
        from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor

        self._processors = (
            Owlv2Processor.from_pretrained(self._detector_model),
            CLIPProcessor.from_pretrained(self._clip_model),
        )
        detector = Owlv2ForObjectDetection.from_pretrained(self._detector_model).eval()
        clip = CLIPModel.from_pretrained(self._clip_model).eval()
        return (detector.to(self.device), clip.to(self.device))

    def _detect(
        self, images: Sequence[Any], classes: list[str]
    ) -> list[dict[str, list[Detection]]]:
        if self._detector is not None:
            results = self._detector(list(images), classes)
            if len(results) != len(images):
                raise ValueError(
                    f"geneval_owl detector returned {len(results)} results for {len(images)} images"
                )
            return [
                {
                    str(cls): [(tuple(map(float, box)), float(score)) for box, score in dets]
                    for cls, dets in result.items()
                }
                for result in results
            ]
        import torch

        detector, _ = self._module_for_inference()
        owl_processor, _ = self._processors
        queries = [self._query_template.format(cls=cls) for cls in classes]
        out: list[dict[str, list[Detection]]] = []
        with torch.no_grad():
            for image in images:
                inputs = owl_processor(text=[queries], images=image, return_tensors="pt").to(
                    self.device
                )
                result = owl_processor.post_process_grounded_object_detection(
                    detector(**inputs),
                    threshold=self._detection_threshold,
                    target_sizes=torch.tensor([image.size[::-1]], device=self.device),
                )[0]
                per_class: dict[str, list[Detection]] = {cls: [] for cls in classes}
                for box, score, label in zip(
                    result["boxes"].tolist(),
                    result["scores"].tolist(),
                    result["labels"].tolist(),
                    strict=True,
                ):
                    per_class[classes[int(label)]].append((tuple(box), float(score)))
                out.append({cls: nms(dets, self._nms_iou) for cls, dets in per_class.items()})
        return out

    def _color_of(self, image: Any, box: Box, cls: str) -> str:
        if self._color_fn is not None:
            return str(self._color_fn(image, box, cls))
        import torch

        _, clip = self._module_for_inference()
        _, clip_processor = self._processors
        x0, y0, x1, y1 = (int(v) for v in box)
        crop = image.crop((max(0, x0), max(0, y0), max(x0 + 1, x1), max(y0 + 1, y1)))
        texts = [self._color_template.format(color=color, cls=cls) for color in GENEVAL_COLORS]
        with torch.no_grad():
            inputs = clip_processor(text=texts, images=crop, return_tensors="pt", padding=True).to(
                self.device
            )
            logits = clip(**inputs).logits_per_image[0]
        return GENEVAL_COLORS[int(logits.argmax())]


def nms(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy per-class non-maximum suppression, then drop group boxes.

    OWLv2 often emits one extra box spanning several instances of a class
    ("two vases" also yields a box around both). Such a box survives IoU
    suppression because it overlaps each instance only partially, so it is
    removed explicitly: a kept box that contains at least two other kept boxes
    of the same class is a group box, not an instance.
    """

    kept: list[Detection] = []
    for box, score in sorted(detections, key=lambda det: -det[1]):
        if all(_iou(box, other[0]) < iou_threshold for other in kept):
            kept.append((box, score))
    return [
        det
        for det in kept
        if sum(other is not det and _contains(det[0], other[0]) for other in kept) < 2
    ]


def _contains(outer: Box, inner: Box, ratio: float = 0.8) -> bool:
    """Whether ``outer`` covers at least ``ratio`` of ``inner``'s area."""

    inter_w = max(0.0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    inter_h = max(0.0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inner_area > 0 and inter_w * inter_h / inner_area >= ratio


def _iou(a: Box, b: Box) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def relative_position(box: Box, reference: Box, threshold: float) -> set[str]:
    """Where ``box`` sits relative to ``reference`` — the official GenEval rule.

    Centre offsets smaller than ``threshold`` times the summed box extents are
    zeroed so touching or overlapping objects claim no relation.
    """

    import numpy as np

    a = np.asarray(box, dtype=float).reshape(2, 2)
    b = np.asarray(reference, dtype=float).reshape(2, 2)
    offset = a.mean(axis=0) - b.mean(axis=0)
    extents = np.abs(a[1] - a[0]) + np.abs(b[1] - b[0])
    revised = np.maximum(np.abs(offset) - threshold * extents, 0.0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    unit = revised / np.linalg.norm(offset)
    relations: set[str] = set()
    if unit[0] < -0.5:
        relations.add("left of")
    if unit[0] > 0.5:
        relations.add("right of")
    if unit[1] < -0.5:
        relations.add("above")
    if unit[1] > 0.5:
        relations.add("below")
    return relations


__all__ = [
    "GENEVAL_COLORS",
    "GenEvalOwlRewardModel",
    "GenEvalVerdict",
    "nms",
    "relative_position",
]
