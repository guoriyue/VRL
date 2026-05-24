"""OpenPose-style anime anatomy structure reward for Anima RL."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardRollout

_REQUIRED_BODY_POINTS = tuple(range(1, 14))
_FOOT_ANCHOR_POINTS = (10, 13)
_ARM_CHAINS = ((2, 3, 4), (5, 6, 7))
_LEG_CHAINS = ((8, 9, 10), (11, 12, 13))
_PAIRED_LIMB_SEGMENTS = (
    ((2, 3), (5, 6)),
    ((3, 4), (6, 7)),
    ((8, 9), (11, 12)),
    ((9, 10), (12, 13)),
)
_BODY_SCALE_SEGMENTS = ((1, 8), (1, 11), (2, 5), (8, 11))
_BOTH_HAND_HINTS = ("both hands",)
_HAND_HINTS = ("hand", "hands", "finger", "fingers", "holding", "gesture", "clenched")
_FEET_HINTS = (
    "feet",
    "foot",
    "shoe",
    "shoes",
    "boot",
    "boots",
    "full body",
    "standing",
    "walking",
    "running",
    "jumping",
    "kicking",
)
_HINT_TRANSLATION = str.maketrans({char: " " for char in "_-/.,;:()[]{}!?"})

# Fixed geometry penalty weights — not tunable per-instance.
_MISSING_KEYPOINT_PENALTY = 0.35
_IMPOSSIBLE_ANGLE_PENALTY = 0.25
_ASYMMETRIC_LIMB_PENALTY = 0.15
_HAND_MISSING_PENALTY = 0.30
_COLLAPSED_HAND_PENALTY = 0.15
_MULTI_PERSON_PENALTY = 0.10
_MIN_JOINT_ANGLE_DEGREES = 18.0
_MAX_SEGMENT_RATIO = 4.0
_MAX_LIMB_ASYMMETRY_RATIO = 2.5
_MIN_HAND_SPREAD_RATIO = 0.035
_MIN_HAND_KEYPOINTS = 6


class AnimeAnatomyStructureReward(RewardFunction):
    """Reward anime images using whole-body OpenPose-style keypoint geometry."""

    def __init__(
        self,
        device: str = "cuda",
        model_repo: str = "yzd-v/DWPose",
        detector_file: str = "yolox_l.onnx",
        pose_file: str = "dw-ll_ucoco_384.onnx",
        cache_dir: str = "",
        local_files_only: bool = False,
        detector: Callable[[list[Any]], list[Any]] | None = None,
        detect_resolution: int = 512,
        min_keypoint_confidence: float = 0.25,
        min_body_keypoints: int = 8,
        require_hands: str | bool = "prompt",
        require_feet: str | bool = "prompt",
    ) -> None:
        self._model_repo = str(model_repo)
        self._detector_file = str(detector_file)
        self._pose_file = str(pose_file)
        self._cache_dir = str(cache_dir or "")
        self._local_files_only = bool(local_files_only)
        self._device = str(device)
        self._detector_factory = detector
        self._detector: Any = None
        if int(detect_resolution) <= 0:
            raise ValueError("detect_resolution must be > 0")
        self._detect_resolution = int(detect_resolution)
        if not (0.0 <= min_keypoint_confidence <= 1.0):
            raise ValueError("min_keypoint_confidence must be in [0, 1]")
        self._min_keypoint_confidence = float(min_keypoint_confidence)
        if int(min_body_keypoints) <= 0:
            raise ValueError("min_body_keypoints must be > 0")
        self._min_body_keypoints = int(min_body_keypoints)
        for name, val in (("require_hands", require_hands), ("require_feet", require_feet)):
            if isinstance(val, bool):
                val = "always" if val else "never"
            if str(val).strip().lower() not in {"always", "never", "prompt"}:
                raise ValueError(f"{name} must be one of: always, never, prompt")
        self._require_hands = "always" if require_hands is True else "never" if require_hands is False else str(require_hands).strip().lower()
        self._require_feet = "always" if require_feet is True else "never" if require_feet is False else str(require_feet).strip().lower()

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        image_groups = [_extract_images(rollout.trajectory.output) for rollout in rollouts]
        flat_images: list[Any] = []
        flat_requirements: list[tuple[int, bool]] = []
        for rollout, group in zip(rollouts, image_groups, strict=True):
            requirements = self._requirements_from_rollout(rollout)
            for image in group:
                flat_images.append(image)
                flat_requirements.append(requirements)

        if not flat_images:
            flat_results: list[Any] = []
        elif self._detector_factory is not None:
            flat_results = list(self._detector_factory(flat_images))
        else:
            if self._detector is None:
                self._detector = _DWPoseONNXDetector(
                    model_repo=self._model_repo,
                    detector_file=self._detector_file,
                    pose_file=self._pose_file,
                    cache_dir=self._cache_dir or None,
                    local_files_only=self._local_files_only,
                    device=self._device,
                    detect_resolution=self._detect_resolution,
                )
            flat_results = [self._detector(image) for image in flat_images]

        flat_scores = [
            self._score_pose_result(result, requirements)
            for result, requirements in zip(flat_results, flat_requirements, strict=True)
        ]

        out: list[float] = []
        cursor = 0
        for group in image_groups:
            count = len(group)
            scores = flat_scores[cursor : cursor + count]
            cursor += count
            out.append(sum(scores) / len(scores) if scores else 0.0)
        return out

    def _score_pose_result(self, result: Any, requirements: tuple[int, bool]) -> float:
        people = _people_from_result(result, min_score=self._min_keypoint_confidence)
        if not people:
            return 0.0

        hand_count, require_feet = requirements
        person = max(people, key=_person_confidence)
        score = 1.0
        body_coverage = _coverage(person.body, _REQUIRED_BODY_POINTS)
        if _present_count(person.body) < self._min_body_keypoints:
            body_coverage = min(
                body_coverage, _present_count(person.body) / self._min_body_keypoints
            )
        score -= _MISSING_KEYPOINT_PENALTY * (1.0 - body_coverage)

        if require_feet:
            score -= _MISSING_KEYPOINT_PENALTY * _feet_missing_fraction(person)

        if hand_count > 0:
            min_hand_spread = _MIN_HAND_SPREAD_RATIO * max(_body_scale(person), 1e-6)
            visible_hands = _visible_hand_count(
                person,
                min_points=_MIN_HAND_KEYPOINTS,
                min_spread=min_hand_spread,
            )
            score -= _HAND_MISSING_PENALTY * max(0.0, hand_count - visible_hands) / hand_count
            score -= _COLLAPSED_HAND_PENALTY * _collapsed_hand_fraction(
                person,
                min_points=_MIN_HAND_KEYPOINTS,
                min_spread=min_hand_spread,
            )

        score -= _IMPOSSIBLE_ANGLE_PENALTY * _joint_geometry_penalty(
            person,
            min_angle_degrees=_MIN_JOINT_ANGLE_DEGREES,
            max_segment_ratio=_MAX_SEGMENT_RATIO,
        )
        score -= _ASYMMETRIC_LIMB_PENALTY * _limb_asymmetry_penalty(
            person,
            max_ratio=_MAX_LIMB_ASYMMETRY_RATIO,
        )
        if len(people) > 1:
            score -= _MULTI_PERSON_PENALTY
        return max(0.0, min(1.0, score))

    def _requirements_from_rollout(self, rollout: RewardRollout) -> tuple[int, bool]:
        prompt = str(getattr(rollout.trajectory, "prompt", "") or "")
        constraints = _constraint_texts(getattr(rollout, "metadata", None))
        texts = constraints if constraints else (prompt,)

        if self._require_hands == "never":
            hand_count = 0
        elif self._require_hands == "always" or _contains_hint(texts, _BOTH_HAND_HINTS):
            hand_count = 2
        else:
            hand_count = 1 if _contains_hint(texts, _HAND_HINTS) else 0

        if self._require_feet == "never":
            feet_required = False
        elif self._require_feet == "always":
            feet_required = True
        else:
            feet_required = _contains_hint(texts, _FEET_HINTS)

        return hand_count, feet_required


@dataclass(frozen=True)
class _Keypoint:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class _PersonPose:
    body: tuple[_Keypoint | None, ...]
    feet: tuple[_Keypoint | None, ...]
    hands: tuple[tuple[_Keypoint | None, ...], ...]


def _people_from_result(result: Any, *, min_score: float) -> list[_PersonPose]:
    if not isinstance(result, Mapping):
        return []
    if "keypoints" not in result:
        return []
    scores = result["scores"] if "scores" in result else result.get("keypoint_scores")
    return _people_from_arrays(result["keypoints"], scores, min_score=min_score)


def _people_from_arrays(keypoints: Any, scores: Any, *, min_score: float) -> list[_PersonPose]:
    keypoint_arr = np.asarray(keypoints, dtype=float)
    if keypoint_arr.ndim == 2:
        keypoint_arr = keypoint_arr[None, ...]
    if keypoint_arr.ndim != 3 or keypoint_arr.shape[-1] < 2:
        return []
    if scores is None:
        score_arr = np.ones(keypoint_arr.shape[:2], dtype=float)
    else:
        score_arr = np.asarray(scores, dtype=float)
        if score_arr.ndim == 1:
            score_arr = score_arr[None, ...]
    people = []
    for idx in range(keypoint_arr.shape[0]):
        body = _points_from_array(keypoint_arr[idx, :18], score_arr[idx, :18], min_score)
        feet = (
            _points_from_array(keypoint_arr[idx, 18:24], score_arr[idx, 18:24], min_score)
            if keypoint_arr.shape[1] >= 24
            else ()
        )
        hands: list[tuple[_Keypoint | None, ...]] = []
        if keypoint_arr.shape[1] >= 113:
            hands.append(
                _points_from_array(keypoint_arr[idx, 92:113], score_arr[idx, 92:113], min_score),
            )
        if keypoint_arr.shape[1] >= 134:
            hands.append(
                _points_from_array(keypoint_arr[idx, 113:134], score_arr[idx, 113:134], min_score),
            )
        people.append(_PersonPose(body=body, feet=feet, hands=tuple(hands)))
    return people


def _points_from_array(
    points: Any,
    scores: Any,
    min_score: float,
) -> tuple[_Keypoint | None, ...]:
    point_arr = np.asarray(points, dtype=float)
    if point_arr.ndim != 2 or point_arr.shape[-1] < 2:
        return ()
    if scores is None:
        score_arr = np.ones(point_arr.shape[0], dtype=float)
    else:
        score_arr = np.asarray(scores, dtype=float)
        if score_arr.ndim == 0:
            score_arr = np.full(point_arr.shape[0], float(score_arr))
    parsed: list[_Keypoint | None] = []
    for coords, score in zip(point_arr[:, :2], score_arr, strict=False):
        x = float(coords[0])
        y = float(coords[1])
        parsed.append(
            None
            if score < min_score or x < 0.0 or y < 0.0 or x > 1.0 or y > 1.0
            else _Keypoint(x=x, y=y, score=float(score))
        )
    return tuple(parsed)


def _coverage(points: Sequence[_Keypoint | None], indices: Sequence[int]) -> float:
    if not indices:
        return 1.0
    present = sum(1 for idx in indices if idx < len(points) and points[idx] is not None)
    return present / len(indices)


def _present_count(points: Sequence[_Keypoint | None]) -> int:
    return sum(1 for point in points if point is not None)


def _person_confidence(person: _PersonPose) -> float:
    present = [point.score for point in person.body if point is not None]
    if not present:
        return 0.0
    return sum(present) / len(present)


def _constraint_texts(metadata: Any) -> tuple[str, ...]:
    if not isinstance(metadata, Mapping):
        return ()
    value = metadata.get("constraints")
    if not value:
        return ()
    if isinstance(value, str):
        return (value.strip().lower(),) if value.strip() else ()
    return tuple(
        dict.fromkeys(
            item.strip().lower() for item in value if isinstance(item, str) and item.strip()
        )
    )


def _contains_hint(texts: Sequence[str], hints: Sequence[str]) -> bool:
    for raw_text in texts:
        text = f" {_normalize_hint_text(raw_text)} "
        for raw_hint in hints:
            hint = f" {_normalize_hint_text(raw_hint)} "
            if hint in text:
                return True
    return False


def _normalize_hint_text(text: str) -> str:
    return " ".join(str(text).lower().translate(_HINT_TRANSLATION).split())


def _feet_missing_fraction(person: _PersonPose) -> float:
    ankle_coverage = _coverage(person.body, _FOOT_ANCHOR_POINTS)
    if len(person.feet) >= 6:
        foot_coverage = _coverage(person.feet, range(len(person.feet)))
        return 1.0 - max(ankle_coverage, foot_coverage)
    return 1.0 - ankle_coverage


def _visible_hand_count(person: _PersonPose, *, min_points: int, min_spread: float) -> int:
    return sum(
        1
        for hand in person.hands
        if _present_count(hand) >= min_points and _point_spread(hand) >= min_spread
    )


def _collapsed_hand_fraction(person: _PersonPose, *, min_points: int, min_spread: float) -> float:
    visible = [hand for hand in person.hands if _present_count(hand) >= min_points]
    if not visible:
        return 0.0
    collapsed = sum(1 for hand in visible if _point_spread(hand) < min_spread)
    return collapsed / len(visible)


def _point_spread(points: Sequence[_Keypoint | None]) -> float:
    present = [point for point in points if point is not None]
    if len(present) < 2:
        return 0.0
    xs = [point.x for point in present]
    ys = [point.y for point in present]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _body_scale(person: _PersonPose) -> float:
    lengths = []
    for start, end in _BODY_SCALE_SEGMENTS:
        distance = _distance(person.body, start, end)
        if distance is not None:
            lengths.append(distance)
    return max(lengths) if lengths else 1.0


def _joint_geometry_penalty(
    person: _PersonPose,
    *,
    min_angle_degrees: float,
    max_segment_ratio: float,
) -> float:
    bad = 0.0
    total = 0
    for start, joint, end in (*_ARM_CHAINS, *_LEG_CHAINS):
        a = _distance(person.body, start, joint)
        b = _distance(person.body, joint, end)
        angle = _angle_degrees(person.body, start, joint, end)
        if a is None or b is None or angle is None:
            continue
        total += 1
        if min(a, b) <= 1e-6:
            bad += 1.0
            continue
        if angle < min_angle_degrees:
            bad += 1.0
        ratio = max(a, b) / min(a, b)
        if ratio > max_segment_ratio:
            bad += min(1.0, (ratio - max_segment_ratio) / max_segment_ratio)
    return min(1.0, bad / total) if total else 0.0


def _limb_asymmetry_penalty(
    person: _PersonPose,
    *,
    max_ratio: float,
) -> float:
    penalties = []
    for left, right in _PAIRED_LIMB_SEGMENTS:
        left_length = _distance(person.body, *left)
        right_length = _distance(person.body, *right)
        if left_length is None or right_length is None or min(left_length, right_length) <= 1e-6:
            continue
        ratio = max(left_length, right_length) / min(left_length, right_length)
        if ratio > max_ratio:
            penalties.append(min(1.0, (ratio - max_ratio) / max_ratio))
    return sum(penalties) / len(penalties) if penalties else 0.0


def _distance(points: Sequence[_Keypoint | None], start: int, end: int) -> float | None:
    if start >= len(points) or end >= len(points):
        return None
    a = points[start]
    b = points[end]
    if a is None or b is None:
        return None
    return math.hypot(a.x - b.x, a.y - b.y)


def _angle_degrees(
    points: Sequence[_Keypoint | None],
    start: int,
    joint: int,
    end: int,
) -> float | None:
    if start >= len(points) or joint >= len(points) or end >= len(points):
        return None
    a = points[start]
    b = points[joint]
    c = points[end]
    if a is None or b is None or c is None:
        return None
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    norm = math.hypot(*v1) * math.hypot(*v2)
    if norm <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / norm))
    return math.degrees(math.acos(cosine))


def _extract_images(output: Any) -> list[Any]:
    if output is None:
        return []

    from PIL import Image

    if isinstance(output, Image.Image):
        return [output.convert("RGB")]
    if isinstance(output, (list, tuple)):
        images: list[Any] = []
        for item in output:
            images.extend(_extract_images(item))
        return images

    try:
        import torch

        if torch.is_tensor(output):
            tensor = output.detach().cpu()
            if tensor.ndim == 4:
                return _extract_images(tensor[tensor.shape[0] // 2])
            if tensor.ndim == 3:
                if tensor.shape[0] in {1, 3, 4}:
                    tensor = tensor.permute(1, 2, 0)
                arr = tensor.float().numpy()
                if arr.max(initial=0.0) <= 1.0:
                    arr = arr * 255.0
                return [Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB")]
    except ImportError:
        pass
    return []



# ---------------------------------------------------------------------------
# DWPose ONNX whole-body keypoint detector (private — only used above)
# Mirrors the Apache-2.0 easy-dwpose implementation; weights fetched via
# hf_hub_download's normal cache rather than a bespoke downloader.
# ---------------------------------------------------------------------------

class _DWPoseONNXDetector:
    def __init__(
        self,
        *,
        model_repo: str,
        detector_file: str,
        pose_file: str,
        cache_dir: str | None,
        local_files_only: bool,
        device: str,
        detect_resolution: int,
    ) -> None:
        try:
            import cv2
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "anime_anatomy_structure requires opencv-python, "
                "onnxruntime, and huggingface_hub",
            ) from exc

        self._cv2 = cv2
        self._detect_resolution = detect_resolution
        providers, provider_options = _onnx_providers(ort, device)
        detector_path = hf_hub_download(
            repo_id=model_repo,
            filename=detector_file,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        pose_path = hf_hub_download(
            repo_id=model_repo,
            filename=pose_file,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._det_session = ort.InferenceSession(
            detector_path,
            providers=providers,
            provider_options=provider_options,
        )
        self._pose_session = ort.InferenceSession(
            pose_path,
            providers=providers,
            provider_options=provider_options,
        )

    def __call__(self, image: Any) -> Mapping[str, Any]:
        from PIL import Image

        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.array(image, dtype=np.uint8)
        height, width, _ = arr.shape
        scale = float(self._detect_resolution) / min(height, width)
        target_width = int(np.round(width * scale / 64.0)) * 64
        target_height = int(np.round(height * scale / 64.0)) * 64
        interpolation = self._cv2.INTER_LANCZOS4 if scale > 1 else self._cv2.INTER_AREA
        arr = self._cv2.resize(arr, (target_width, target_height), interpolation=interpolation)
        height, width = arr.shape[:2]
        boxes = _onnx_inference_detector(self._det_session, self._cv2, arr)
        keypoints, scores = _onnx_inference_pose(self._pose_session, self._cv2, boxes, arr)
        keypoints, scores = _onnx_openpose_order(keypoints, scores)
        keypoints = keypoints.astype(float)
        keypoints[..., 0] /= float(width)
        keypoints[..., 1] /= float(height)
        return {"keypoints": keypoints, "scores": scores}


def _onnx_providers(
    ort: Any,
    device: str,
) -> tuple[list[str], list[dict[str, int] | dict[str, Any]] | None]:
    device = str(device)
    if device == "cpu":
        return ["CPUExecutionProvider"], None
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"], None
    gpu_id = 0
    if ":" in device:
        try:
            gpu_id = int(device.rsplit(":", 1)[1])
        except ValueError:
            gpu_id = 0
    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": gpu_id}, {}]


def _onnx_inference_detector(session: Any, cv2: Any, image: np.ndarray) -> np.ndarray:
    input_shape = (640, 640)
    padded = np.ones((input_shape[0], input_shape[1], 3), dtype=np.uint8) * 114
    ratio = min(input_shape[0] / image.shape[0], input_shape[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded[: int(image.shape[0] * ratio), : int(image.shape[1] * ratio)] = resized
    processed = np.ascontiguousarray(padded.transpose((2, 0, 1)), dtype=np.float32)
    ort_inputs = {session.get_inputs()[0].name: processed[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = _onnx_detector_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    detections = _onnx_multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if detections is None:
        return np.array([])
    final_boxes = detections[:, :4]
    final_scores = detections[:, 4]
    final_classes = detections[:, 5]
    keep = np.logical_and(final_scores > 0.3, final_classes == 0)
    return final_boxes[keep]


def _onnx_detector_postprocess(
    outputs: np.ndarray,
    image_size: tuple[int, int],
    *,
    p6: bool = False,
) -> np.ndarray:
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
    heights = [image_size[0] // stride for stride in strides]
    widths = [image_size[1] // stride for stride in strides]
    for height, width, stride in zip(heights, widths, strides, strict=True):
        xv, yv = np.meshgrid(np.arange(width), np.arange(height))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grids_arr = np.concatenate(grids, 1)
    strides_arr = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids_arr) * strides_arr
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides_arr
    return outputs


def _onnx_multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    nms_thr: float,
    score_thr: float,
) -> np.ndarray | None:
    detections = []
    for class_idx in range(scores.shape[1]):
        class_scores = scores[:, class_idx]
        valid = class_scores > score_thr
        if valid.sum() == 0:
            continue
        valid_scores = class_scores[valid]
        valid_boxes = boxes[valid]
        keep = _onnx_nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            class_ids = np.ones((len(keep), 1)) * class_idx
            detections.append(
                np.concatenate([valid_boxes[keep], valid_scores[keep, None], class_ids], 1),
            )
    return np.concatenate(detections, 0) if detections else None


def _onnx_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / (areas[current] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def _onnx_inference_pose(
    session: Any,
    cv2: Any,
    boxes: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = session.get_inputs()[0].shape[2:]
    input_size = (width, height)
    images, centers, scales = _onnx_preprocess_pose(cv2, image, boxes, input_size)
    outputs = []
    output_names = [out.name for out in session.get_outputs()]
    input_name = session.get_inputs()[0].name
    for processed in images:
        outputs.append(session.run(output_names, {input_name: [processed.transpose(2, 0, 1)]}))
    return _onnx_postprocess_pose(outputs, input_size, centers, scales)


def _onnx_preprocess_pose(
    cv2: Any,
    image: np.ndarray,
    boxes: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    image_shape = image.shape[:2]
    if len(boxes) == 0:
        boxes = np.array([[0, 0, image_shape[1], image_shape[0]]])
    out_images, out_centers, out_scales = [], [], []
    for box in boxes:
        x1, y1, x2, y2 = np.hsplit(np.asarray(box)[None, :], [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2])[0] * 0.5
        scale = np.hstack([x2 - x1, y2 - y1])[0] * 1.25
        resized, scale = _onnx_top_down_affine(cv2, input_size, scale, center, image)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        out_images.append((resized - mean) / std)
        out_centers.append(center)
        out_scales.append(scale)
    return out_images, out_centers, out_scales


def _onnx_postprocess_pose(
    outputs: list[np.ndarray],
    model_input_size: tuple[int, int],
    centers: list[np.ndarray],
    scales: list[np.ndarray],
    *,
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    all_keypoints, all_scores = [], []
    for idx, output in enumerate(outputs):
        simcc_x, simcc_y = output
        keypoints, scores = _onnx_simcc_maximum(simcc_x, simcc_y)
        keypoints /= simcc_split_ratio
        keypoints = keypoints / model_input_size * scales[idx] + centers[idx] - scales[idx] / 2
        all_keypoints.append(keypoints[0])
        all_scores.append(scores[0])
    return np.asarray(all_keypoints), np.asarray(all_scores)


def _onnx_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, keypoints, width_x = simcc_x.shape
    _, _, width_y = simcc_y.shape
    x_scores = simcc_x.reshape(count * keypoints, width_x)
    y_scores = simcc_y.reshape(count * keypoints, width_y)
    x_locs = np.argmax(x_scores, axis=1)
    y_locs = np.argmax(y_scores, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    values = np.minimum(np.amax(x_scores, axis=1), np.amax(y_scores, axis=1))
    locs[values <= 0.0] = -1
    return locs.reshape(count, keypoints, 2), values.reshape(count, keypoints)


def _onnx_top_down_affine(
    cv2: Any,
    input_size: tuple[int, int],
    box_scale: np.ndarray,
    box_center: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = input_size
    aspect_ratio = width / height
    box_width, box_height = np.hsplit(box_scale[None, :], [1])
    box_scale = np.where(
        box_width > box_height * aspect_ratio,
        np.hstack([box_width, box_width / aspect_ratio]),
        np.hstack([box_height * aspect_ratio, box_height]),
    )[0]
    warp_matrix = _onnx_warp_matrix(box_center, box_scale, output_size=(width, height))
    return cv2.warpAffine(image, warp_matrix, (int(width), int(height)), flags=cv2.INTER_LINEAR), box_scale


def _onnx_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    import cv2

    src_width = scale[0]
    dst_width, dst_height = output_size
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + np.array([0.0, src_width * -0.5])
    direction = src[0, :] - src[1, :]
    src[2, :] = src[1, :] + np.r_[-direction[1], direction[0]]
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_width * 0.5, dst_height * 0.5]
    dst[1, :] = [dst_width * 0.5, dst_height * 0.5 - dst_width * 0.5]
    direction = dst[0, :] - dst[1, :]
    dst[2, :] = dst[1, :] + np.r_[-direction[1], direction[0]]
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _onnx_openpose_order(keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keypoint_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(keypoint_info[:, [5, 6]], axis=1)
    neck[:, 2] = np.logical_and(keypoint_info[:, 5, 2] > 0.3, keypoint_info[:, 6, 2] > 0.3)
    reordered = np.insert(keypoint_info, 17, neck, axis=1)
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    reordered[:, openpose_idx] = reordered[:, mmpose_idx]
    return reordered[..., :2], reordered[..., 2]


__all__ = ["AnimeAnatomyStructureReward"]
