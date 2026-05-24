"""OpenPose-style anime anatomy structure reward for Anima RL."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardRollout

_OPENPOSE18 = {
    "nose": 0,
    "neck": 1,
    "right_shoulder": 2,
    "right_elbow": 3,
    "right_wrist": 4,
    "left_shoulder": 5,
    "left_elbow": 6,
    "left_wrist": 7,
    "right_hip": 8,
    "right_knee": 9,
    "right_ankle": 10,
    "left_hip": 11,
    "left_knee": 12,
    "left_ankle": 13,
}


def _kp(*names: str) -> tuple[int, ...]:
    return tuple(_OPENPOSE18[name] for name in names)


_OPENPOSE18_LAYOUT = {
    "required_body_points": _kp(
        "neck",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
    ),
    "foot_anchor_points": _kp("right_ankle", "left_ankle"),
    "arm_chains": (
        _kp("right_shoulder", "right_elbow", "right_wrist"),
        _kp("left_shoulder", "left_elbow", "left_wrist"),
    ),
    "leg_chains": (
        _kp("right_hip", "right_knee", "right_ankle"),
        _kp("left_hip", "left_knee", "left_ankle"),
    ),
    "paired_limb_segments": (
        (_kp("right_shoulder", "right_elbow"), _kp("left_shoulder", "left_elbow")),
        (_kp("right_elbow", "right_wrist"), _kp("left_elbow", "left_wrist")),
        (_kp("right_hip", "right_knee"), _kp("left_hip", "left_knee")),
        (_kp("right_knee", "right_ankle"), _kp("left_knee", "left_ankle")),
    ),
    "body_scale_segments": (
        _kp("neck", "right_hip"),
        _kp("neck", "left_hip"),
        _kp("right_shoulder", "left_shoulder"),
        _kp("right_hip", "left_hip"),
    ),
}

_CONSTRAINT_LEXICON = {
    "both_hand_phrases": ("both hands",),
    "hand_terms": ("hand", "hands", "finger", "fingers", "holding", "gesture", "clenched"),
    "feet_terms": (
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
    ),
}


class AnimeAnatomyStructureReward(RewardFunction):
    """Reward anime images using whole-body OpenPose-style keypoint geometry."""

    def __init__(
        self,
        device: str = "cuda",
        backend: str = "dwpose",
        model_repo: str = "yzd-v/DWPose",
        detector_file: str = "yolox_l.onnx",
        pose_file: str = "dw-ll_ucoco_384.onnx",
        cache_dir: str = "",
        local_files_only: bool = False,
        detector_device: str = "",
        detect_resolution: int = 512,
        detector: Callable[..., Sequence[Any] | Any] | None = None,
        min_keypoint_confidence: float = 0.25,
        min_body_keypoints: int = 8,
        min_hand_keypoints: int = 6,
        require_hands: str | bool = "prompt",
        require_feet: str | bool = "prompt",
        missing_required_keypoint_penalty: float = 0.35,
        impossible_angle_penalty: float = 0.25,
        asymmetric_limb_penalty: float = 0.15,
        hand_missing_penalty: float = 0.30,
        collapsed_hand_penalty: float = 0.15,
        multi_person_penalty: float = 0.10,
        min_joint_angle_degrees: float = 18.0,
        max_segment_ratio: float = 4.0,
        max_limb_asymmetry_ratio: float = 2.5,
        min_hand_spread_ratio: float = 0.035,
    ) -> None:
        self._backend = str(backend)
        self._model_repo = str(model_repo)
        self._detector_file = str(detector_file)
        self._pose_file = str(pose_file)
        self._cache_dir = str(cache_dir or "")
        self._local_files_only = bool(local_files_only)
        self._device = str(detector_device or device)
        self._detect_resolution = _validate_positive_int("detect_resolution", detect_resolution)
        self._detector_factory = detector
        self._detector: Callable[[Any], Any] | None = None
        self._layout = _OPENPOSE18_LAYOUT
        self._constraint_lexicon = _CONSTRAINT_LEXICON
        self._min_keypoint_confidence = _validate_probability(
            "min_keypoint_confidence",
            min_keypoint_confidence,
        )
        self._min_body_keypoints = _validate_positive_int("min_body_keypoints", min_body_keypoints)
        self._min_hand_keypoints = _validate_positive_int("min_hand_keypoints", min_hand_keypoints)
        self._require_hands = _validate_requirement_mode("require_hands", require_hands)
        self._require_feet = _validate_requirement_mode("require_feet", require_feet)
        self._missing_required_keypoint_penalty = _validate_non_negative(
            "missing_required_keypoint_penalty",
            missing_required_keypoint_penalty,
        )
        self._impossible_angle_penalty = _validate_non_negative(
            "impossible_angle_penalty",
            impossible_angle_penalty,
        )
        self._asymmetric_limb_penalty = _validate_non_negative(
            "asymmetric_limb_penalty",
            asymmetric_limb_penalty,
        )
        self._hand_missing_penalty = _validate_non_negative(
            "hand_missing_penalty",
            hand_missing_penalty,
        )
        self._collapsed_hand_penalty = _validate_non_negative(
            "collapsed_hand_penalty",
            collapsed_hand_penalty,
        )
        self._multi_person_penalty = _validate_non_negative(
            "multi_person_penalty",
            multi_person_penalty,
        )
        self._min_joint_angle_degrees = _validate_non_negative(
            "min_joint_angle_degrees",
            min_joint_angle_degrees,
        )
        self._max_segment_ratio = _validate_positive_float("max_segment_ratio", max_segment_ratio)
        self._max_limb_asymmetry_ratio = _validate_positive_float(
            "max_limb_asymmetry_ratio",
            max_limb_asymmetry_ratio,
        )
        self._min_hand_spread_ratio = _validate_non_negative(
            "min_hand_spread_ratio",
            min_hand_spread_ratio,
        )

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        image_groups = [_extract_images(rollout.trajectory.output) for rollout in rollouts]
        flat_images: list[Any] = []
        flat_requirements: list[_PoseRequirements] = []
        for rollout, group in zip(rollouts, image_groups, strict=True):
            requirements = _requirements_from_rollout(
                rollout,
                require_hands=self._require_hands,
                require_feet=self._require_feet,
                lexicon=self._constraint_lexicon,
            )
            for image in group:
                flat_images.append(image)
                flat_requirements.append(requirements)

        flat_results = self._detect_images(flat_images)
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

    def _detect_images(self, images: list[Any]) -> list[Any]:
        if not images:
            return []
        if self._detector_factory is not None:
            return _call_detector(self._detector_factory, images)
        detector = self._ensure_detector()
        return [detector(image) for image in images]

    def _ensure_detector(self) -> Callable[[Any], Any]:
        if self._detector is not None:
            return self._detector
        backend = self._backend.lower().replace("-", "_")
        if backend in {"dwpose", "dwpose_onnx"}:
            self._detector = _DWPoseONNXDetector(
                model_repo=self._model_repo,
                detector_file=self._detector_file,
                pose_file=self._pose_file,
                cache_dir=self._cache_dir or None,
                local_files_only=self._local_files_only,
                device=self._device,
                detect_resolution=self._detect_resolution,
            )
        elif backend == "openpose":
            self._detector = _ControlNetOpenPoseDetector(
                model_repo=self._model_repo,
                cache_dir=self._cache_dir or None,
                local_files_only=self._local_files_only,
                device=self._device,
                detect_resolution=self._detect_resolution,
            )
        else:
            raise ValueError("anime_anatomy_structure.backend must be 'dwpose' or 'openpose'")
        return self._detector

    def _score_pose_result(self, result: Any, requirements: _PoseRequirements) -> float:
        people = _people_from_result(result, min_score=self._min_keypoint_confidence)
        if not people:
            return 0.0

        person = max(people, key=_person_confidence)
        score = 1.0
        body_coverage = _coverage(person.body, self._layout["required_body_points"])
        if _present_count(person.body) < self._min_body_keypoints:
            body_coverage = min(body_coverage, _present_count(person.body) / self._min_body_keypoints)
        score -= self._missing_required_keypoint_penalty * (1.0 - body_coverage)

        if requirements.require_feet:
            score -= self._missing_required_keypoint_penalty * _feet_missing_fraction(
                person,
                layout=self._layout,
            )

        hand_count = requirements.hand_count
        if hand_count > 0:
            visible_hands = _visible_hand_count(
                person,
                min_points=self._min_hand_keypoints,
                min_spread=self._min_hand_spread_ratio * max(_body_scale(person, self._layout), 1e-6),
            )
            score -= self._hand_missing_penalty * max(0.0, hand_count - visible_hands) / hand_count
            score -= self._collapsed_hand_penalty * _collapsed_hand_fraction(
                person,
                min_points=self._min_hand_keypoints,
                min_spread=self._min_hand_spread_ratio * max(_body_scale(person, self._layout), 1e-6),
            )

        score -= self._impossible_angle_penalty * _joint_geometry_penalty(
            person,
            layout=self._layout,
            min_angle_degrees=self._min_joint_angle_degrees,
            max_segment_ratio=self._max_segment_ratio,
        )
        score -= self._asymmetric_limb_penalty * _limb_asymmetry_penalty(
            person,
            layout=self._layout,
            max_ratio=self._max_limb_asymmetry_ratio,
        )
        if len(people) > 1:
            score -= self._multi_person_penalty
        return _clamp01(score)


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


@dataclass(frozen=True, slots=True)
class _PoseRequirements:
    hand_count: int
    require_feet: bool


def _call_detector(detector: Callable[..., Any], images: list[Any]) -> list[Any]:
    raw = detector(images)
    if _is_batch_result(raw, expected_count=len(images)):
        return list(raw)
    if len(images) == 1:
        return [raw]
    return [detector(image) for image in images]


def _is_batch_result(value: Any, *, expected_count: int) -> bool:
    if isinstance(value, Mapping):
        return False
    if isinstance(value, np.ndarray):
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) == expected_count
    return False


def _people_from_result(result: Any, *, min_score: float) -> list[_PersonPose]:
    if result is None:
        return []
    if isinstance(result, Mapping):
        return _people_from_mapping(result, min_score=min_score)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        people = []
        for item in result:
            if _looks_like_openpose_person(item):
                people.append(_person_from_openpose(item, min_score=min_score))
            elif isinstance(item, Mapping):
                people.extend(_people_from_mapping(item, min_score=min_score))
        return people
    if _looks_like_openpose_person(result):
        return [_person_from_openpose(result, min_score=min_score)]
    return []


def _people_from_mapping(result: Mapping[str, Any], *, min_score: float) -> list[_PersonPose]:
    if "keypoints" in result:
        scores = result["scores"] if "scores" in result else result.get("keypoint_scores")
        return _people_from_arrays(
            result["keypoints"],
            scores,
            min_score=min_score,
        )
    if "body" in result:
        return _people_from_explicit_mapping(result, min_score=min_score)
    if "bodies" in result:
        bodies = result["bodies"]
        if isinstance(bodies, Mapping):
            return _people_from_controlnet_dwpose(result, min_score=min_score)
        return _people_from_flat_dwpose(result, min_score=min_score)
    return []


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


def _people_from_explicit_mapping(
    result: Mapping[str, Any],
    *,
    min_score: float,
) -> list[_PersonPose]:
    body = _points_from_maybe_scored(result.get("body"), min_score=min_score)
    feet = _points_from_maybe_scored(result.get("feet", ()), min_score=min_score)
    hands_raw = result.get("hands", ())
    hands = tuple(
        _points_from_maybe_scored(hand, min_score=min_score)
        for hand in hands_raw
        if hand is not None
    )
    return [_PersonPose(body=body, feet=feet, hands=hands)] if body else []


def _people_from_flat_dwpose(result: Mapping[str, Any], *, min_score: float) -> list[_PersonPose]:
    bodies = np.asarray(result["bodies"], dtype=float)
    if bodies.ndim != 2 or bodies.shape[0] % 18 != 0:
        return []
    person_count = bodies.shape[0] // 18
    body_scores = np.asarray(result.get("body_scores", np.ones((person_count, 18))), dtype=float)
    body_points = bodies.reshape(person_count, 18, bodies.shape[-1])
    hands = np.asarray(result.get("hands", np.empty((0, 21, 2))), dtype=float)
    hand_scores = np.asarray(result.get("hands_scores", np.ones(hands.shape[:2])), dtype=float)
    people = []
    for idx in range(person_count):
        person_hands: list[tuple[_Keypoint | None, ...]] = []
        for hand_idx in (idx, idx + person_count):
            if hand_idx < len(hands):
                person_hands.append(
                    _points_from_array(hands[hand_idx], hand_scores[hand_idx], min_score),
                )
        people.append(
            _PersonPose(
                body=_points_from_array(body_points[idx], body_scores[idx], min_score),
                feet=(),
                hands=tuple(person_hands),
            ),
        )
    return people


def _people_from_controlnet_dwpose(
    result: Mapping[str, Any],
    *,
    min_score: float,
) -> list[_PersonPose]:
    bodies = result["bodies"]
    candidate = np.asarray(bodies.get("candidate", ()), dtype=float)
    subset = np.asarray(bodies.get("subset", ()), dtype=float)
    if subset.ndim != 2:
        return []
    hands = np.asarray(result.get("hands", np.empty((0, 21, 2))), dtype=float)
    people = []
    for person_idx in range(subset.shape[0]):
        points: list[_Keypoint | None] = []
        for key_idx in range(min(18, subset.shape[1])):
            candidate_idx = int(subset[person_idx, key_idx])
            if candidate_idx < 0 or candidate_idx >= len(candidate):
                points.append(None)
            else:
                points.append(_point(candidate[candidate_idx], 1.0, min_score))
        while len(points) < 18:
            points.append(None)
        person_hands: list[tuple[_Keypoint | None, ...]] = []
        for hand_idx in (person_idx, person_idx + subset.shape[0]):
            if hand_idx < len(hands):
                person_hands.append(_points_from_array(hands[hand_idx], None, min_score))
        people.append(_PersonPose(body=tuple(points), feet=(), hands=tuple(person_hands)))
    return people


def _points_from_maybe_scored(value: Any, *, min_score: float) -> tuple[_Keypoint | None, ...]:
    if value is None:
        return ()
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[-1] < 2:
        return ()
    scores = arr[:, 2] if arr.shape[-1] >= 3 else np.ones(arr.shape[0], dtype=float)
    return _points_from_array(arr[:, :2], scores, min_score)


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
    return tuple(
        _point(coords, float(score), min_score)
        for coords, score in zip(point_arr[:, :2], score_arr, strict=False)
    )


def _point(coords: Any, score: float, min_score: float) -> _Keypoint | None:
    x = float(coords[0])
    y = float(coords[1])
    if score < min_score or x < 0.0 or y < 0.0 or x > 1.0 or y > 1.0:
        return None
    return _Keypoint(x=x, y=y, score=float(score))


def _looks_like_openpose_person(value: Any) -> bool:
    return hasattr(value, "body") and hasattr(value, "left_hand") and hasattr(value, "right_hand")


def _person_from_openpose(value: Any, *, min_score: float) -> _PersonPose:
    body_points = []
    for keypoint in getattr(value.body, "keypoints", ()):
        body_points.append(_point_from_attr(keypoint, min_score=min_score))
    while len(body_points) < 18:
        body_points.append(None)
    hands = tuple(
        tuple(_point_from_attr(keypoint, min_score=min_score) for keypoint in (hand or ()))
        for hand in (value.left_hand, value.right_hand)
        if hand is not None
    )
    return _PersonPose(body=tuple(body_points[:18]), feet=(), hands=hands)


def _point_from_attr(value: Any, *, min_score: float) -> _Keypoint | None:
    if value is None:
        return None
    score = float(getattr(value, "score", 1.0) or 1.0)
    return _point((value.x, value.y), score, min_score)


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


def _requirements_from_rollout(
    rollout: RewardRollout,
    *,
    require_hands: str,
    require_feet: str,
    lexicon: Mapping[str, Sequence[str]],
) -> _PoseRequirements:
    prompt = str(getattr(rollout.trajectory, "prompt", "") or "")
    constraints = _constraint_texts(getattr(rollout, "metadata", None))
    texts = constraints if constraints else (prompt,)
    return _PoseRequirements(
        hand_count=_required_hand_count(require_hands, texts, lexicon),
        require_feet=_required_feet(require_feet, texts, lexicon),
    )


def _constraint_texts(metadata: Any) -> tuple[str, ...]:
    constraints: list[str] = []
    if isinstance(metadata, Mapping):
        _collect_constraints(metadata.get("constraints"), constraints)
        nested_metadata = metadata.get("metadata")
        if isinstance(nested_metadata, Mapping):
            _collect_constraints(nested_metadata.get("constraints"), constraints)
        manifest_row = metadata.get("manifest_row")
        if isinstance(manifest_row, Mapping):
            _collect_constraints(manifest_row.get("constraints"), constraints)
            row_metadata = manifest_row.get("metadata")
            if isinstance(row_metadata, Mapping):
                _collect_constraints(row_metadata.get("constraints"), constraints)
    return tuple(dict.fromkeys(text.strip().lower() for text in constraints if text.strip()))


def _collect_constraints(value: Any, out: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        out.append(value)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _collect_constraints(item, out)


def _required_hand_count(
    mode: str,
    texts: Sequence[str],
    lexicon: Mapping[str, Sequence[str]],
) -> int:
    if mode == "never":
        return 0
    if mode == "always":
        return 2
    if _contains_hint(texts, lexicon["both_hand_phrases"]):
        return 2
    return 1 if _contains_hint(texts, lexicon["hand_terms"]) else 0


def _required_feet(
    mode: str,
    texts: Sequence[str],
    lexicon: Mapping[str, Sequence[str]],
) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    return _contains_hint(texts, lexicon["feet_terms"])


def _contains_hint(texts: Sequence[str], hints: Sequence[str]) -> bool:
    for text in texts:
        words = _words(text)
        for hint in hints:
            if _contains_phrase(words, _words(hint)):
                return True
    return False


def _words(text: str) -> tuple[str, ...]:
    normalized = str(text).lower()
    for char in "_-/.,;:()[]{}!?":
        normalized = normalized.replace(char, " ")
    return tuple(normalized.split())


def _contains_phrase(words: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase:
        return False
    if len(phrase) == 1:
        return phrase[0] in words
    size = len(phrase)
    return any(tuple(words[idx : idx + size]) == tuple(phrase) for idx in range(len(words) - size + 1))


def _feet_missing_fraction(person: _PersonPose, *, layout: Mapping[str, Any]) -> float:
    ankle_coverage = _coverage(person.body, layout["foot_anchor_points"])
    if len(person.feet) >= 6:
        foot_coverage = _coverage(person.feet, range(len(person.feet)))
        return 1.0 - max(ankle_coverage, foot_coverage)
    return 1.0 - ankle_coverage


def _visible_hand_count(person: _PersonPose, *, min_points: int, min_spread: float) -> int:
    return sum(1 for hand in person.hands if _hand_is_visible(hand, min_points, min_spread))


def _collapsed_hand_fraction(person: _PersonPose, *, min_points: int, min_spread: float) -> float:
    visible = [hand for hand in person.hands if _present_count(hand) >= min_points]
    if not visible:
        return 0.0
    collapsed = sum(1 for hand in visible if _point_spread(hand) < min_spread)
    return collapsed / len(visible)


def _hand_is_visible(
    hand: Sequence[_Keypoint | None],
    min_points: int,
    min_spread: float,
) -> bool:
    return _present_count(hand) >= min_points and _point_spread(hand) >= min_spread


def _point_spread(points: Sequence[_Keypoint | None]) -> float:
    present = [point for point in points if point is not None]
    if len(present) < 2:
        return 0.0
    xs = [point.x for point in present]
    ys = [point.y for point in present]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _body_scale(person: _PersonPose, layout: Mapping[str, Any]) -> float:
    lengths = []
    for start, end in layout["body_scale_segments"]:
        distance = _distance(person.body, start, end)
        if distance is not None:
            lengths.append(distance)
    return max(lengths) if lengths else 1.0


def _joint_geometry_penalty(
    person: _PersonPose,
    *,
    layout: Mapping[str, Any],
    min_angle_degrees: float,
    max_segment_ratio: float,
) -> float:
    bad = 0.0
    total = 0
    for start, joint, end in (*layout["arm_chains"], *layout["leg_chains"]):
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
    layout: Mapping[str, Any],
    max_ratio: float,
) -> float:
    penalties = []
    for left, right in layout["paired_limb_segments"]:
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
    images: list[Any] = []
    _append_images(output, images)
    return images


def _append_images(value: Any, images: list[Any]) -> None:
    if value is None:
        return

    from PIL import Image

    if isinstance(value, Image.Image):
        images.append(value.convert("RGB"))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _append_images(item, images)
        return

    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.ndim == 4:
                _append_images(tensor[tensor.shape[0] // 2], images)
                return
            if tensor.ndim == 3:
                if tensor.shape[0] in {1, 3, 4}:
                    tensor = tensor.permute(1, 2, 0)
                arr = tensor.float().numpy()
                if arr.max(initial=0.0) <= 1.0:
                    arr = arr * 255.0
                images.append(Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB"))
                return
    except ImportError:
        pass


class _DWPoseONNXDetector:
    """Run DWPose ONNX and return normalized whole-body keypoints."""

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
                "anime_anatomy_structure backend='dwpose' requires opencv-python, "
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
        arr = _dwpose_resize_image(self._cv2, arr, target_resolution=self._detect_resolution)
        height, width = arr.shape[:2]
        boxes = _dwpose_inference_detector(self._det_session, self._cv2, arr)
        keypoints, scores = _dwpose_inference_pose(self._pose_session, self._cv2, boxes, arr)
        keypoints, scores = _dwpose_openpose_order(keypoints, scores)
        keypoints = keypoints.astype(float)
        keypoints[..., 0] /= float(width)
        keypoints[..., 1] /= float(height)
        return {"keypoints": keypoints, "scores": scores}


class _ControlNetOpenPoseDetector:
    def __init__(
        self,
        *,
        model_repo: str,
        cache_dir: str | None,
        local_files_only: bool,
        device: str,
        detect_resolution: int,
    ) -> None:
        try:
            from controlnet_aux import OpenposeDetector
        except ImportError as exc:
            raise ImportError(
                "anime_anatomy_structure backend='openpose' requires controlnet_aux",
            ) from exc

        self._detector = OpenposeDetector.from_pretrained(
            model_repo,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device)
        self._detect_resolution = detect_resolution

    def __call__(self, image: Any) -> Any:
        from PIL import Image

        arr = np.array(image.convert("RGB") if isinstance(image, Image.Image) else image)
        return self._detector.detect_poses(
            arr,
            include_hand=True,
            include_face=False,
        )


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


# The DWPose ONNX preprocessing mirrors the Apache-2.0 easy-dwpose implementation,
# but downloads weights through hf_hub_download's normal cache instead of a
# repository-local checkpoints directory.
def _dwpose_resize_image(cv2: Any, image: np.ndarray, *, target_resolution: int) -> np.ndarray:
    height, width, _ = image.shape
    scale = float(target_resolution) / min(height, width)
    target_width = int(np.round(width * scale / 64.0)) * 64
    target_height = int(np.round(height * scale / 64.0)) * 64
    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def _dwpose_preprocess_detector(
    cv2: Any,
    image: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded[: int(image.shape[0] * ratio), : int(image.shape[1] * ratio)] = resized
    padded = padded.transpose((2, 0, 1))
    return np.ascontiguousarray(padded, dtype=np.float32), ratio


def _dwpose_inference_detector(session: Any, cv2: Any, image: np.ndarray) -> np.ndarray:
    input_shape = (640, 640)
    processed, ratio = _dwpose_preprocess_detector(cv2, image, input_shape)
    ort_inputs = {session.get_inputs()[0].name: processed[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = _dwpose_detector_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    detections = _dwpose_multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if detections is None:
        return np.array([])
    final_boxes = detections[:, :4]
    final_scores = detections[:, 4]
    final_classes = detections[:, 5]
    keep = np.logical_and(final_scores > 0.3, final_classes == 0)
    return final_boxes[keep]


def _dwpose_detector_postprocess(
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


def _dwpose_multiclass_nms(
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
        keep = _dwpose_nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            class_ids = np.ones((len(keep), 1)) * class_idx
            detections.append(np.concatenate([valid_boxes[keep], valid_scores[keep, None], class_ids], 1))
    return np.concatenate(detections, 0) if detections else None


def _dwpose_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        inter = width * height
        overlap = inter / (areas[current] + areas[order[1:]] - inter)
        indices = np.where(overlap <= threshold)[0]
        order = order[indices + 1]
    return keep


def _dwpose_inference_pose(
    session: Any,
    cv2: Any,
    boxes: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = session.get_inputs()[0].shape[2:]
    input_size = (width, height)
    images, centers, scales = _dwpose_preprocess_pose(cv2, image, boxes, input_size)
    outputs = []
    output_names = [out.name for out in session.get_outputs()]
    input_name = session.get_inputs()[0].name
    for processed in images:
        outputs.append(session.run(output_names, {input_name: [processed.transpose(2, 0, 1)]}))
    return _dwpose_postprocess_pose(outputs, input_size, centers, scales)


def _dwpose_preprocess_pose(
    cv2: Any,
    image: np.ndarray,
    boxes: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    image_shape = image.shape[:2]
    if len(boxes) == 0:
        boxes = np.array([[0, 0, image_shape[1], image_shape[0]]])
    out_images = []
    out_centers = []
    out_scales = []
    for box in boxes:
        center, scale = _dwpose_bbox_xyxy_to_center_scale(np.asarray(box), padding=1.25)
        resized, scale = _dwpose_top_down_affine(cv2, input_size, scale, center, image)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        out_images.append((resized - mean) / std)
        out_centers.append(center)
        out_scales.append(scale)
    return out_images, out_centers, out_scales


def _dwpose_postprocess_pose(
    outputs: list[np.ndarray],
    model_input_size: tuple[int, int],
    centers: list[np.ndarray],
    scales: list[np.ndarray],
    *,
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    all_keypoints = []
    all_scores = []
    for idx, output in enumerate(outputs):
        simcc_x, simcc_y = output
        keypoints, scores = _dwpose_decode_simcc(simcc_x, simcc_y, simcc_split_ratio)
        keypoints = keypoints / model_input_size * scales[idx] + centers[idx] - scales[idx] / 2
        all_keypoints.append(keypoints[0])
        all_scores.append(scores[0])
    return np.asarray(all_keypoints), np.asarray(all_scores)


def _dwpose_decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_split_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints, scores = _dwpose_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio
    return keypoints, scores


def _dwpose_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, keypoints, width_x = simcc_x.shape
    _, _, width_y = simcc_y.shape
    x_scores = simcc_x.reshape(count * keypoints, width_x)
    y_scores = simcc_y.reshape(count * keypoints, width_y)
    x_locs = np.argmax(x_scores, axis=1)
    y_locs = np.argmax(y_scores, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_x = np.amax(x_scores, axis=1)
    max_y = np.amax(y_scores, axis=1)
    values = np.minimum(max_x, max_y)
    locs[values <= 0.0] = -1
    return locs.reshape(count, keypoints, 2), values.reshape(count, keypoints)


def _dwpose_bbox_xyxy_to_center_scale(
    box: np.ndarray,
    *,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = np.hsplit(box[None, :], [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2])[0] * 0.5
    scale = np.hstack([x2 - x1, y2 - y1])[0] * padding
    return center, scale


def _dwpose_top_down_affine(
    cv2: Any,
    input_size: tuple[int, int],
    box_scale: np.ndarray,
    box_center: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = input_size
    box_scale = _dwpose_fix_aspect_ratio(box_scale, aspect_ratio=width / height)
    warp_matrix = _dwpose_warp_matrix(box_center, box_scale, output_size=(width, height))
    return cv2.warpAffine(image, warp_matrix, (int(width), int(height)), flags=cv2.INTER_LINEAR), box_scale


def _dwpose_fix_aspect_ratio(box_scale: np.ndarray, *, aspect_ratio: float) -> np.ndarray:
    width, height = np.hsplit(box_scale[None, :], [1])
    fixed = np.where(
        width > height * aspect_ratio,
        np.hstack([width, width / aspect_ratio]),
        np.hstack([height * aspect_ratio, height]),
    )
    return fixed[0]


def _dwpose_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    src_width = scale[0]
    dst_width, dst_height = output_size
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + np.array([0.0, src_width * -0.5])
    src[2, :] = _dwpose_third_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_width * 0.5, dst_height * 0.5]
    dst[1, :] = [dst_width * 0.5, dst_height * 0.5 - dst_width * 0.5]
    dst[2, :] = _dwpose_third_point(dst[0, :], dst[1, :])
    import cv2

    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _dwpose_third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def _dwpose_openpose_order(keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keypoint_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(keypoint_info[:, [5, 6]], axis=1)
    neck[:, 2] = np.logical_and(keypoint_info[:, 5, 2] > 0.3, keypoint_info[:, 6, 2] > 0.3)
    reordered = np.insert(keypoint_info, 17, neck, axis=1)
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    reordered[:, openpose_idx] = reordered[:, mmpose_idx]
    return reordered[..., :2], reordered[..., 2]


def _validate_requirement_mode(name: str, value: str | bool) -> str:
    if isinstance(value, bool):
        return "always" if value else "never"
    text = str(value).strip().lower()
    if text not in {"always", "never", "prompt"}:
        raise ValueError(f"{name} must be one of: always, never, prompt")
    return text


def _validate_positive_int(name: str, value: int) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _validate_probability(name: str, value: float) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _validate_non_negative(name: str, value: float) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _validate_positive_float(name: str, value: float) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["AnimeAnatomyStructureReward"]
