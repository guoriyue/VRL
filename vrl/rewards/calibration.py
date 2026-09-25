"""A frozen linear reward combination, applied identically offline and online.

A combination is fitted offline (``reward_lab.calibration``) from pairwise
preferences over standardized score axes. This class only checks a saved
combination against the scoring recipe it was fitted on and applies it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from vrl.utils.json_files import canonical_json_sha256

COMBINATION_SCHEMA = "vrl.reward-combination.v1"


@dataclass(frozen=True, slots=True, init=False)
class FrozenRewardCombination:
    """Standardize the named axes with fitted means and scales, then weight them.

    Only detached tuples survive construction, so the objective of a loaded run
    cannot change under it. This performs no model inference.
    """

    combination_id: str
    scoring_config_hash: str
    dimension: str
    axes: tuple[str, ...]
    weights: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __init__(self, combination: dict[str, Any], *, scoring_config: Mapping[str, Any]) -> None:
        if combination.get("schema") != COMBINATION_SCHEMA:
            raise ValueError("unsupported reward combination schema")
        expected = canonical_json_sha256(dict(scoring_config), allow_nan=False)
        if combination["scoring_config_hash"] != expected:
            raise ValueError("scoring recipe differs from calibration")
        axes = combination["axes"]
        if not axes or len(set(axes)) != len(axes):
            raise ValueError("invalid combination axes")
        weights, means, scales = (
            np.asarray(combination[key], dtype=np.float64)
            for key in ("weights", "means", "scales")
        )
        if (
            any(
                v.shape != (len(axes),) or not np.isfinite(v).all()
                for v in (weights, means, scales)
            )
            or (scales <= 0).any()
        ):
            raise ValueError("invalid combination vectors")
        for name, value in (
            ("combination_id", combination["combination_id"]),
            ("scoring_config_hash", combination["scoring_config_hash"]),
            ("dimension", combination["dimension"]),
            ("axes", tuple(axes)),
            ("weights", tuple(weights.tolist())),
            ("means", tuple(means.tolist())),
            ("scales", tuple(scales.tolist())),
        ):
            object.__setattr__(self, name, value)

    def apply(
        self, scores: Mapping[str, float], *, sample_id: str = "sample"
    ) -> tuple[float, dict[str, float]]:
        """One total and the signed contribution of each axis."""

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


__all__ = ["COMBINATION_SCHEMA", "FrozenRewardCombination"]
