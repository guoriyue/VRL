"""Frozen linear reward combinations fitted to source-separated preferences.

No model inference or training configuration mutation. Pairwise logistic fitting
uses calibration labels only; holdout evaluation never refits scales or weights.
These preferences supervise a combination of pointwise scores, not a new
image-to-image pairwise judge.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from vrl.config.base import ConfigBase
from vrl.rewards.evaluation import fingerprint
from vrl.scripts.eval.score_report import bootstrap_mean_interval


class PreferencePair(ConfigBase):
    """One annotation; repeated annotations retain distinct pair IDs."""

    pair_id: str = Field(min_length=1)
    left: str = Field(min_length=1)
    right: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    split: Literal["calibration", "holdout"]
    preference: Literal["left", "right", "tie", "unsure"]
    dimension: str = Field(default="overall", min_length=1)
    tags: list[str] = Field(default_factory=list)


def load_preferences(path: Path) -> list[PreferencePair]:
    pairs = [
        PreferencePair.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not pairs or len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("preference pair IDs must be non-empty and unique")
    return pairs


def validate_preferences(evaluation: dict[str, Any], pairs: list[PreferencePair]) -> None:
    """Reject source, prompt, sample, and media leakage across declared splits."""
    assignments: dict[tuple[str, str], str] = {}
    sources: dict[str, str] = {}
    records = evaluation["records"]
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("duplicate preference pair IDs")
    for pair in pairs:
        if pair.left == pair.right:
            raise ValueError("preference cannot compare a sample with itself")
        prompt_ids = set()
        identities = [("source", pair.source_group)]
        for sample_id in (pair.left, pair.right):
            if sample_id not in records or records[sample_id]["status"] != "success":
                raise ValueError(f"preference references an unscored sample: {sample_id}")
            row = records[sample_id]["input"]
            prompt_ids.add(row["prompt_id"])
            identities.extend(
                (("sample", sample_id), ("prompt", row["prompt_id"]), ("media", row["sha256"]))
            )
            identities.extend(("asset", value) for value in row.get("asset_sha256", {}).values())
            previous = sources.setdefault(sample_id, pair.source_group)
            if previous != pair.source_group:
                raise ValueError("one sample cannot belong to multiple source groups")
        if len(prompt_ids) != 1:
            raise ValueError("preference pairs must compare samples from the same prompt")
        for identity in identities:
            previous = assignments.setdefault(identity, pair.split)
            if previous != pair.split:
                raise ValueError(
                    f"calibration/holdout leakage through {identity[0]}: {identity[1]}"
                )


def fit_combination(
    evaluation: dict[str, Any],
    pairs: list[PreferencePair],
    *,
    axes: list[str],
    dimension: str = "overall",
    l2: float = 0.1,
    tie_margin: float = 0.1,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """Fit convex L2 logistic preference loss with equal source-group weight.

    Axes are standardized on unique calibration samples only. A constant axis
    gets unit scale and zero information. Ties target 0.5; unsure labels are
    excluded. A positive regularizer makes the Hessian invertible even with
    redundant axes. Failure to converge raises instead of publishing weights.
    """
    validate_preferences(evaluation, pairs)
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("axes must be non-empty and unique")
    if not math.isfinite(l2) or l2 <= 0 or not math.isfinite(tie_margin) or tie_margin < 0:
        raise ValueError("l2 must be positive and tie_margin non-negative, both finite")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    selected = [
        p
        for p in pairs
        if p.split == "calibration" and p.dimension == dimension and p.preference != "unsure"
    ]
    if len({p.source_group for p in selected}) < 2 or not any(
        p.preference in {"left", "right"} for p in selected
    ):
        raise ValueError(
            "fitting needs two source groups and at least one decisive calibration label"
        )
    ids = sorted({sample_id for p in selected for sample_id in (p.left, p.right)})
    try:
        raw = np.asarray(
            [
                [evaluation["records"][sample_id]["result"]["scores"][axis] for axis in axes]
                for sample_id in ids
            ],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError("calibration score axis missing") from error
    if not np.isfinite(raw).all():
        raise ValueError("calibration scores must be finite")
    means, scales = raw.mean(axis=0), raw.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    features = dict(zip(ids, (raw - means) / scales, strict=True))
    x = np.asarray([features[p.left] - features[p.right] for p in selected])
    y = np.asarray(
        [
            1.0 if p.preference == "left" else 0.0 if p.preference == "right" else 0.5
            for p in selected
        ]
    )
    counts = Counter(p.source_group for p in selected)
    sample_weights = np.asarray([1.0 / (len(counts) * counts[p.source_group]) for p in selected])
    weights = np.zeros(len(axes), dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        logits = x @ candidate
        return float(
            sample_weights @ (np.logaddexp(0, logits) - y * logits)
            + l2 * (candidate @ candidate) / 2
        )

    for _iteration in range(max_iterations):
        logits = x @ weights
        probability = np.exp(-np.logaddexp(0, -logits))
        gradient = x.T @ (sample_weights * (probability - y)) + l2 * weights
        if np.linalg.norm(gradient, ord=np.inf) < 1e-8:
            break
        hessian = x.T @ (
            (sample_weights * probability * (1 - probability))[:, None] * x
        ) + l2 * np.eye(len(axes))
        step = np.linalg.solve(hessian, gradient)
        rate = 1.0
        current = objective(weights)
        while rate > 1e-10 and objective(weights - rate * step) > current - 1e-4 * rate * float(
            gradient @ step
        ):
            rate *= 0.5
        if rate <= 1e-10:
            raise RuntimeError("calibration line search failed")
        weights -= rate * step
    else:
        raise RuntimeError("calibration did not converge")
    payload = {
        "schema": "vrl.reward-combination.v1",
        "scoring_config_hash": fingerprint(evaluation["config"]),
        "fit_run_id": evaluation["run_id"],
        "calibration_labels_hash": fingerprint([p.model_dump(mode="json") for p in selected]),
        "dimension": dimension,
        "axes": axes,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": weights.tolist(),
        "raw_weights": (weights / scales).tolist(),
        "raw_bias": float(-(means * weights / scales).sum()),
        "tie_margin": tie_margin,
        "l2": l2,
        "calibration_source_groups": sorted(counts),
        "calibration_prompt_ids": sorted(
            {evaluation["records"][key]["input"]["prompt_id"] for key in ids}
        ),
        "calibration_media_hashes": sorted(
            {evaluation["records"][key]["input"]["sha256"] for key in ids}
        ),
        "calibration_asset_hashes": sorted(
            {
                value
                for key in ids
                for value in evaluation["records"][key]["input"].get("asset_sha256", {}).values()
            }
        ),
        "fit_pairs": len(selected),
        "fit_samples": len(ids),
        "iterations": _iteration,
        "objective": objective(weights),
    }
    return {"combination_id": fingerprint(payload), **payload}


def evaluate_combination(
    evaluation: dict[str, Any],
    pairs: list[PreferencePair],
    combination: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen combination on holdout with explicit three-way ties."""
    payload = {k: v for k, v in combination.items() if k != "combination_id"}
    if payload.get("schema") != "vrl.reward-combination.v1" or fingerprint(
        payload
    ) != combination.get("combination_id"):
        raise ValueError("invalid frozen combination digest or schema")
    if fingerprint(evaluation["config"]) != payload["scoring_config_hash"]:
        raise ValueError("scoring recipe differs from calibration")
    validate_preferences(evaluation, pairs)
    selected = [p for p in pairs if p.split == "holdout" and p.dimension == payload["dimension"]]
    if not selected:
        raise ValueError("holdout has no labels for this dimension")
    axes = payload["axes"]
    weights, means, scales = (np.asarray(payload[key]) for key in ("weights", "means", "scales"))
    if (
        any(
            value.shape != (len(axes),) or not np.isfinite(value).all()
            for value in (weights, means, scales)
        )
        or (scales <= 0).any()
    ):
        raise ValueError("invalid combination vectors")
    outcomes, grouped = [], defaultdict(list)
    tags: dict[str, list[float]] = defaultdict(list)
    for pair in selected:
        if pair.source_group in payload["calibration_source_groups"]:
            raise ValueError("holdout source group was used for calibration")
        scores = []
        for sample_id in (pair.left, pair.right):
            row = evaluation["records"][sample_id]
            item = row["input"]
            if (
                item["prompt_id"] in payload["calibration_prompt_ids"]
                or item["sha256"] in payload["calibration_media_hashes"]
                or set(item.get("asset_sha256", {}).values())
                & set(payload["calibration_asset_hashes"])
            ):
                raise ValueError("holdout media, prompt, or reference was used for calibration")
            try:
                raw = np.asarray(
                    [row["result"]["scores"][axis] for axis in axes], dtype=np.float64
                )
            except KeyError as error:
                raise ValueError("holdout score axis missing") from error
            if not np.isfinite(raw).all():
                raise ValueError("holdout scores must be finite")
            scores.append(float(((raw - means) / scales) @ weights))
        delta = scores[0] - scores[1]
        prediction = (
            "tie" if abs(delta) <= payload["tie_margin"] else "left" if delta > 0 else "right"
        )
        correct = None if pair.preference == "unsure" else float(prediction == pair.preference)
        outcomes.append(
            {
                **pair.model_dump(mode="json"),
                "delta": delta,
                "prediction": prediction,
                "correct": correct,
            }
        )
        if correct is not None:
            grouped[pair.source_group].append(correct)
            for tag in set(pair.tags):
                tags[tag].append(correct)
    group_means = [statistics.fmean(values) for values in grouped.values()]
    return {
        "combination_id": combination["combination_id"],
        "evaluation_run_id": evaluation["run_id"],
        "holdout_labels_hash": fingerprint([p.model_dump(mode="json") for p in selected]),
        "source_balanced_agreement": statistics.fmean(group_means) if group_means else None,
        "bootstrap_95ci": list(
            bootstrap_mean_interval(
                group_means,
                schema="vrl.reward-preferences.v1",
                label=combination["combination_id"],
                score_key=payload["dimension"],
            )
        )
        if len(group_means) >= 2
        else None,
        "judged_source_groups": len(group_means),
        "unsure_count": sum(p.preference == "unsure" for p in selected),
        "by_tag": {
            tag: {"annotations": len(values), "agreement": statistics.fmean(values)}
            for tag, values in tags.items()
        },
        "outcomes": outcomes,
        "tie_policy": "Three-way exact agreement; unsure excluded; tie margin frozen at fit.",
    }
