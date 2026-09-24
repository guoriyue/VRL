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

Three scores come out of one detector pass:

* ``geneval_owl_strict`` is 1 only when every condition holds — the official
  GenEval score, and what a held-out evaluation reports;
* ``geneval_owl_partial`` is the satisfied-condition fraction;
* ``geneval_owl_dense`` replaces each condition's yes/no with the continuous
  quantity the verdict thresholds, and is the training signal.

The dense score exists because GRPO consumes the *ordering* of rewards within a
prompt group, and a thresholded verdict barely orders anything: measured on 8
prompts x 16 samples, the partial score took only 2-4 distinct values per group
(2 on counting prompts), so the advantage collapsed to "which half of the batch
am I in" and the policy could not move (SPRINT_anima_geneval_spatial_rl 7.5).
Detection confidence, colour probability, and position margin are continuous in
exactly the region the thresholds cut, so scoring them directly restores a full
ordering without changing what the objective means. Detection runs at a lower
floor so a near-miss is visible; the strict verdict still sees only boxes above
its own threshold and is unchanged.

Same image, same score: there is no sampling anywhere. Absolute numbers are not
comparable to published GenEval tables because the detector differs;
before/after comparisons on one detector are.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vrl.rewards.inference import RewardInferenceResult
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
ColorClassifier = Callable[[Any, Box, str], "str | Mapping[str, float]"]


@dataclass(frozen=True, slots=True)
class GenEvalVerdict:
    """One image's GenEval outcome: strict pass, graded scores, and why."""

    strict: float
    partial: float
    dense: float
    why: str

    @property
    def scores(self) -> dict[str, float]:
        return {
            "geneval_owl_dense": self.dense,
            "geneval_owl_partial": self.partial,
            "geneval_owl_strict": self.strict,
        }


class GenEvalOwlRewardModel(LazyTorchModule):
    """Per-artifact GenEval verdict from ``metadata.geneval`` with OWLv2 + CLIP."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(worker_config)
        self.device = str(cfg.get("device", "cuda"))
        self._detector_model = str(cfg.get("detector_model", "google/owlv2-base-patch16-ensemble"))
        self._clip_model = str(cfg.get("clip_model", "openai/clip-vit-large-patch14"))
        self._detection_threshold = float(cfg.get("detection_threshold", 0.15))
        # Detection floor for the dense score only: a box between this and
        # detection_threshold is a near-miss the dense term should still see,
        # while the strict verdict keeps ignoring it.
        self._dense_detection_threshold = float(cfg.get("dense_detection_threshold", 0.05))
        self._nms_iou = float(cfg.get("nms_iou", 0.5))
        self._position_threshold = float(cfg.get("position_threshold", 0.1))
        for name, value in (
            ("detection_threshold", self._detection_threshold),
            ("dense_detection_threshold", self._dense_detection_threshold),
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
        """Compatibility API for direct callers that only consume numeric scores."""
        return [result.scores for result in self.score_results(artifacts)]

    def score_results(self, artifacts: Sequence[Any]) -> list[RewardInferenceResult]:
        """Retain the verdict explanation through local/Ray/HTTP result auditing."""
        specs = [self._spec(artifact) for artifact in artifacts]
        images = [artifact_middle_frame_image(artifact) for artifact in artifacts]
        return [
            RewardInferenceResult(
                artifact_id=artifact.artifact_id,
                scores=verdict.scores,
                diagnostics={
                    "why": verdict.why,
                    "spec": spec,
                    "kind": "detector-based-geneval-verdict",
                },
            )
            for artifact, spec, verdict in zip(
                artifacts, specs, self.judge_images(images, specs), strict=True
            )
        ]

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
        """Score one image against one GenEval spec: strict, partial, and dense.

        One detection pass at the dense floor feeds both readings. The strict
        and partial scores see only boxes above ``detection_threshold``, so they
        are exactly what a verdict-only evaluator would report; the dense score
        sees the whole set and reads each condition's underlying continuous
        quantity instead of its yes/no.
        """

        include = list(spec.get("include") or [])
        exclude = list(spec.get("exclude") or [])
        if not include and not exclude:
            raise ValueError("geneval spec carries neither include nor exclude items")
        classes = sorted({str(item["class"]) for item in include + exclude})
        raw = self._detect([image], classes)[0]
        dense_dets = {cls: nms(dets, self._nms_iou) for cls, dets in raw.items()}
        strict_dets = {
            cls: nms([d for d in dets if d[1] >= self._detection_threshold], self._nms_iou)
            for cls, dets in raw.items()
        }

        satisfied = 0
        total = 0
        dense_terms: list[float] = []
        failures: list[str] = []
        # Reference boxes for position items, indexed like ``include``; None
        # when that item found nothing.
        matched: list[Box | None] = []
        dense_matched: list[Box | None] = []
        for item in include:
            cls = str(item["class"])
            count = int(item.get("count", 1))
            objs = list(strict_dets.get(cls, ()))
            # A couple of spares past ``count`` is enough to order near-misses,
            # and the colour term runs CLIP once per candidate.
            dense_objs = list(dense_dets.get(cls, ()))[: count + 2]
            total += 1
            if len(objs) >= count:
                satisfied += 1
            else:
                failures.append(f"missing:{cls}")
            dense_terms.append(
                self._mean_top(
                    [self._soft_detection(obj[1]) for obj in dense_objs],
                    count,
                )
            )
            if "color" in item:
                total += 1
                wanted = str(item["color"])
                colored = [obj for obj in objs if self._color_of(image, obj[0], cls) == wanted]
                if len(colored) >= count:
                    satisfied += 1
                    objs = colored
                else:
                    failures.append(f"color:{cls}!={wanted}")
                dense_terms.append(
                    self._mean_top(
                        [
                            self._color_probs(image, obj[0], cls).get(wanted, 0.0)
                            for obj in dense_objs
                        ],
                        count,
                    )
                )
            if "position" in item:
                total += 1
                relation, ref_index = item["position"]
                relation = str(relation)
                reference = matched[int(ref_index)] if int(ref_index) < len(matched) else None
                if reference is not None and any(
                    relation in relative_position(obj[0], reference, self._position_threshold)
                    for obj in objs
                ):
                    satisfied += 1
                else:
                    failures.append(f"position:{relation}")
                dense_reference = (
                    dense_matched[int(ref_index)] if int(ref_index) < len(dense_matched) else None
                )
                dense_terms.append(
                    0.0
                    if dense_reference is None or not dense_objs
                    else max(
                        relative_position_margin(
                            obj[0], dense_reference, self._position_threshold, relation
                        )
                        for obj in dense_objs
                    )
                )
            matched.append(objs[0][0] if objs else None)
            dense_matched.append(dense_objs[0][0] if dense_objs else None)
        for item in exclude:
            cls = str(item["class"])
            count = int(item.get("count", 1))
            total += 1
            if len(strict_dets.get(cls, ())) < count:
                satisfied += 1
            else:
                failures.append(f"exclude:{cls}>={count}")
            # How far the forbidden count is from being reached: the confidence
            # of the count-th instance, inverted.
            dense_objs = list(dense_dets.get(cls, ()))
            nth = dense_objs[count - 1][1] if len(dense_objs) >= count else 0.0
            dense_terms.append(1.0 - self._soft_detection(nth))
        return GenEvalVerdict(
            strict=0.0 if failures else 1.0,
            partial=satisfied / total,
            dense=sum(dense_terms) / len(dense_terms),
            why=";".join(failures) or "ok",
        )

    def _soft_detection(self, score: float) -> float:
        """Detection confidence in ``[0, 1]``, saturating at twice the verdict floor."""

        return min(max(score / (2.0 * self._detection_threshold), 0.0), 1.0)

    @staticmethod
    def _mean_top(values: Sequence[float], count: int) -> float:
        """Mean of the ``count`` largest values, missing slots counting as zero."""

        top = sorted(values, reverse=True)[:count]
        return (sum(top) + 0.0) / count if count else 0.0

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
                    threshold=min(self._detection_threshold, self._dense_detection_threshold),
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
                out.append(
                    {cls: sorted(dets, key=lambda det: -det[1]) for cls, dets in per_class.items()}
                )
        return out

    def _color_of(self, image: Any, box: Box, cls: str) -> str:
        """Verdict colour of one detection: the most likely of the vocabulary."""

        probs = self._color_probs(image, box, cls)
        return max(probs, key=lambda color: probs[color])

    def _color_probs(self, image: Any, box: Box, cls: str) -> dict[str, float]:
        """Distribution over the GenEval colour vocabulary for one detection.

        The dense score reads the requested colour's probability, which orders
        images the argmax verdict cannot: "almost yellow" and "clearly blue"
        are the same failure to the verdict and different numbers here.
        """

        if self._color_fn is not None:
            # Test seam: a label is read as a certain classification, a mapping
            # as the distribution itself.
            chosen = self._color_fn(image, box, cls)
            if isinstance(chosen, Mapping):
                return {color: float(chosen.get(color, 0.0)) for color in GENEVAL_COLORS}
            return {color: float(color == str(chosen)) for color in GENEVAL_COLORS}
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
            probs = clip(**inputs).logits_per_image[0].softmax(dim=-1)
        return dict(zip(GENEVAL_COLORS, (float(v) for v in probs), strict=True))


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
    for relation, component in _RELATION_COMPONENTS.items():
        axis, sign = component
        if sign * unit[axis] > 0.5:
            relations.add(relation)
    return relations


# Each relation reads one signed component of the unit offset; the verdict is
# that component exceeding 0.5. The dense score reads the component itself.
_RELATION_COMPONENTS: dict[str, tuple[int, float]] = {
    "left of": (0, -1.0),
    "right of": (0, 1.0),
    "above": (1, -1.0),
    "below": (1, 1.0),
}


def relative_position_margin(box: Box, reference: Box, threshold: float, relation: str) -> float:
    """How far ``box`` sits in ``relation`` to ``reference``, in ``[0, 1]``.

    The signed unit-offset component :func:`relative_position` thresholds at
    0.5, clamped to ``[0, 1]``. Continuous where the verdict is a step, so a
    group of samples that all fail the relation still orders by how close each
    one came.
    """

    import numpy as np

    if relation not in _RELATION_COMPONENTS:
        raise ValueError(f"unknown GenEval position relation: {relation!r}")
    a = np.asarray(box, dtype=float).reshape(2, 2)
    b = np.asarray(reference, dtype=float).reshape(2, 2)
    offset = a.mean(axis=0) - b.mean(axis=0)
    norm = float(np.linalg.norm(offset))
    if norm == 0.0:
        return 0.0
    extents = np.abs(a[1] - a[0]) + np.abs(b[1] - b[0])
    revised = np.maximum(np.abs(offset) - threshold * extents, 0.0) * np.sign(offset)
    axis, sign = _RELATION_COMPONENTS[relation]
    return float(min(max(sign * revised[axis] / norm, 0.0), 1.0))


__all__ = [
    "GENEVAL_COLORS",
    "GenEvalOwlRewardModel",
    "GenEvalVerdict",
    "nms",
    "relative_position",
    "relative_position_margin",
]
