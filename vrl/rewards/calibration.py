"""Frozen linear reward combinations applied identically offline and online.

A combination is fitted offline in ``reward_lab.calibration``; this module
only validates a saved combination against its scoring recipe and applies it.
No model inference or fitting happens here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from vrl.utils.json_files import canonical_json_sha256


def combination_vectors(
    evaluation: dict[str, Any], combination: dict[str, Any]
) -> tuple[dict[str, Any], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Validate the artifact and recipe before either application or evaluation."""
    payload = {k: v for k, v in combination.items() if k != "combination_id"}
    if payload.get("schema") != "vrl.reward-combination.v1" or canonical_json_sha256(
        payload, allow_nan=False
    ) != combination.get("combination_id"):
        raise ValueError("invalid frozen combination digest or schema")
    if (
        canonical_json_sha256(evaluation["config"], allow_nan=False)
        != payload["scoring_config_hash"]
    ):
        raise ValueError("scoring recipe differs from calibration")
    axes = payload["axes"]
    if (
        not isinstance(axes, list)
        or not axes
        or any(not isinstance(axis, str) or not axis for axis in axes)
        or len(set(axes)) != len(axes)
    ):
        raise ValueError("invalid combination axes")
    weights, means, scales = (
        np.asarray(payload[key], dtype=np.float64) for key in ("weights", "means", "scales")
    )
    if (
        any(
            value.shape != (len(axes),) or not np.isfinite(value).all()
            for value in (weights, means, scales)
        )
        or (scales <= 0).any()
        or not math.isfinite(payload["tie_margin"])
        or payload["tie_margin"] < 0
    ):
        raise ValueError("invalid combination vectors or tie margin")
    return payload, axes, weights, means, scales


@dataclass(frozen=True, slots=True, init=False)
class FrozenRewardCombination:
    """Immutable, recipe-bound arithmetic shared by offline and runtime aggregation.

    Construction checks the saved digest and exact scoring recipe. Only detached
    tuples survive construction, so modifying caller dictionaries cannot change
    the objective of an already loaded run. This performs no model inference.
    """

    combination_id: str
    scoring_config_hash: str
    dimension: str
    axes: tuple[str, ...]
    weights: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __init__(self, combination: dict[str, Any], *, scoring_config: Mapping[str, Any]) -> None:
        payload, axes, weights, means, scales = combination_vectors(
            {"config": dict(scoring_config)}, combination
        )
        for name, value in (
            ("combination_id", combination["combination_id"]),
            ("scoring_config_hash", payload["scoring_config_hash"]),
            ("dimension", payload["dimension"]),
            ("axes", tuple(axes)),
            ("weights", tuple(weights.tolist())),
            ("means", tuple(means.tolist())),
            ("scales", tuple(scales.tolist())),
        ):
            object.__setattr__(self, name, value)

    def apply(
        self, scores: Mapping[str, float], *, sample_id: str = "sample"
    ) -> tuple[float, dict[str, float]]:
        """Return one total and signed axis contributions without fitting batch statistics."""
        try:
            raw = np.asarray([scores[axis] for axis in self.axes], dtype=np.float64)
        except KeyError as error:
            raise ValueError(f"combination score axis missing: {sample_id}") from error
        if raw.shape != (len(self.axes),) or not np.isfinite(raw).all():
            raise ValueError(f"combination scores must be finite scalars: {sample_id}")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            contributions = ((raw - self.means) / self.scales) * self.weights
            score = float(contributions.sum())
        if not np.isfinite(contributions).all() or not math.isfinite(score):
            raise ValueError(f"combination arithmetic is nonfinite: {sample_id}")
        return score, dict(zip(self.axes, contributions.tolist(), strict=True))
