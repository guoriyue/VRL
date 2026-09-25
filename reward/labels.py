"""Agreement between a reward axis and independent per-sample labels.

A learned judge is only usable as a reward for the dimension it demonstrably
ranks. Labels are categorical verdicts per sample (``{"outcome": "done",
"collateral": false}``) from blind judges; a *contrast* names which labelled
samples count as positive and which as negative for one question ("done vs
not_done", "clean done vs done-with-collateral"). The report gives, per
contrast, the probability that the axis ranks a random positive above a random
negative (AUC) with a bootstrap interval, so a scorer's blind spots are stated
next to its strengths instead of hidden in one overall number.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from vrl.config.base import ConfigBase
from vrl.rewards.evaluation import Evaluation

LABELS_SCHEMA = "vrl.reward-labels.v1"


class OutcomeLabel(ConfigBase):
    """One judged sample: categorical verdicts keyed by dimension."""

    sample_id: str = Field(min_length=1)
    labels: dict[str, str | bool | int] = Field(min_length=1)
    note: str = ""
    judge: str = ""

    @classmethod
    def load_jsonl(cls, path: Path) -> list[OutcomeLabel]:
        rows = []
        seen = set()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = cls.model_validate_json(line)
            if row.sample_id in seen:
                raise ValueError(
                    f"duplicate label sample_id at {path}:{number}: {row.sample_id!r}"
                )
            seen.add(row.sample_id)
            rows.append(row)
        if not rows:
            raise ValueError("label file is empty")
        return rows


class Contrast(ConfigBase):
    """Which labelled samples are positive and which negative for one question.

    Each side is a conjunction over dimensions; the list per dimension is the
    accepted values. ``{"outcome": ["done"], "collateral": [false]}`` selects
    samples judged done without collateral.
    """

    name: str = Field(min_length=1)
    positive: dict[str, list[str | bool | int]] = Field(min_length=1)
    negative: dict[str, list[str | bool | int]] = Field(min_length=1)

    @classmethod
    def load_json(cls, path: Path) -> list[Contrast]:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("contrasts file must hold a non-empty JSON list")
        contrasts = [cls.model_validate(item) for item in payload]
        names = [contrast.name for contrast in contrasts]
        if len(set(names)) != len(names):
            raise ValueError("contrast names must be unique")
        return contrasts

    def side(self, label: OutcomeLabel) -> str | None:
        """``"positive"``, ``"negative"`` or None for a label outside both sides."""

        def matches(selector: dict[str, list[Any]]) -> bool:
            return all(
                dimension in label.labels and label.labels[dimension] in accepted
                for dimension, accepted in selector.items()
            )

        positive, negative = matches(self.positive), matches(self.negative)
        if positive and negative:
            raise ValueError(f"contrast {self.name!r} selects {label.sample_id!r} on both sides")
        return "positive" if positive else "negative" if negative else None


def rank_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Share of (positive, negative) pairs the scores order correctly; ties count half."""

    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("AUC needs at least one positive and one negative score")
    pos = np.asarray(positive, dtype=float)[:, None]
    neg = np.asarray(negative, dtype=float)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def agreement(
    evaluation: Evaluation,
    labels: Sequence[OutcomeLabel],
    contrasts: Sequence[Contrast],
    *,
    axis: str,
    direction: int = 1,
    gate: float = 0.85,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-contrast AUC of ``axis`` against the labels, with a bootstrap 95% interval.

    ``direction=-1`` scores an axis where lower is better. Labelled samples
    the run did not score successfully are listed, never treated as zeros.
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be 1 or -1")
    if not 0.5 <= gate <= 1.0:
        raise ValueError("gate must lie in [0.5, 1]")
    scores: dict[str, float] = {}
    unscored = []
    for label in labels:
        row = evaluation.records.get(label.sample_id)
        if row is None or row["status"] != "success" or axis not in row["result"]["scores"]:
            unscored.append(label.sample_id)
            continue
        scores[label.sample_id] = direction * float(row["result"]["scores"][axis])
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "schema": LABELS_SCHEMA,
        "run_id": evaluation.run_id,
        "axis": axis,
        "direction": direction,
        "gate": gate,
        "labelled": len(labels),
        "scored": len(scores),
        "unscored_sample_ids": sorted(unscored),
        "contrasts": {},
    }
    for contrast in contrasts:
        sides: dict[str, list[float]] = {"positive": [], "negative": []}
        for label in labels:
            if label.sample_id not in scores:
                continue
            side = contrast.side(label)
            if side is not None:
                sides[side].append(scores[label.sample_id])
        positive, negative = sides["positive"], sides["negative"]
        entry: dict[str, Any] = {
            "positive_count": len(positive),
            "negative_count": len(negative),
            "auc": None,
            "auc_bootstrap_95ci": None,
            "positive_median": statistics.median(positive) if positive else None,
            "negative_median": statistics.median(negative) if negative else None,
            "passed": None,
        }
        if positive and negative:
            auc = rank_auc(positive, negative)
            draws = []
            pos, neg = np.asarray(positive), np.asarray(negative)
            for _ in range(resamples):
                draws.append(
                    rank_auc(
                        rng.choice(pos, size=pos.size, replace=True),
                        rng.choice(neg, size=neg.size, replace=True),
                    )
                )
            low, high = np.percentile(draws, [2.5, 97.5])
            entry.update(
                auc=auc,
                auc_bootstrap_95ci=[float(low), float(high)],
                passed=bool(auc >= gate and math.isfinite(auc)),
            )
        report["contrasts"][contrast.name] = entry
    return report


__all__ = ["Contrast", "OutcomeLabel", "agreement", "rank_auc"]
