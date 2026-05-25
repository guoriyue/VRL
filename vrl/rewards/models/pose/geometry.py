"""Pose/keypoint constants and geometric rule helpers for structure scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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
