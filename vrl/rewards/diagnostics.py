"""Model-free reward health and paired ranking reports from persisted evidence."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrl.rewards.evaluation import fingerprint
from vrl.rewards.inference import RewardInferenceResult
from vrl.scripts.eval.score_report import bootstrap_mean_interval, distribution


def read_evaluation(directory: Path) -> dict[str, Any]:
    """Read an immutable scoring snapshot, retaining failures and missing rows.

    Checks stored provenance and identities, without reopening potentially moved
    media or loading any model. Hashes attest the recorded inputs, not score truth.
    """
    if (directory / ".writer.lock").exists():
        raise ValueError(f"evaluation is still being written: {directory}")
    provenance = json.loads((directory / "provenance.json").read_text())
    run_id = provenance.pop("run_id")
    if provenance.get("schema") != "vrl.reward-evaluation.v1":
        raise ValueError("unsupported evaluation schema")
    if fingerprint(provenance) != run_id:
        raise ValueError("evaluation provenance digest mismatch")
    inputs = provenance["inputs"]
    identities = [row["sample_id"] for row in inputs]
    if not inputs or len(set(identities)) != len(inputs):
        raise ValueError("evaluation inputs must be non-empty and uniquely identified")
    expected = {f"{fingerprint(sample_id)}.json" for sample_id in identities}
    unexpected = {p.name for p in (directory / "samples").glob("*.json")} - expected
    if unexpected:
        raise ValueError(f"unexpected sample records: {sorted(unexpected)}")
    records = {}
    for item in inputs:
        sample_id = item["sample_id"]
        path = directory / "samples" / f"{fingerprint(sample_id)}.json"
        if not path.exists():
            records[sample_id] = {"input": item, "status": "missing"}
            continue
        record = json.loads(path.read_text())
        if record.get("run_id") != run_id or record.get("input") != item:
            raise ValueError(f"incompatible sample record: {sample_id}")
        if record.get("status") == "success":
            result = RewardInferenceResult(**record["result"])
            if result.artifact_id != sample_id or not result.scores:
                raise ValueError(f"invalid result for {sample_id}")
            record["result"] = asdict(result)
        elif record.get("status") != "error" or not isinstance(record.get("error"), dict):
            raise ValueError(f"invalid sample status: {sample_id}")
        records[sample_id] = record
    return {"run_id": run_id, **provenance, "records": records}


def health_report(evaluation: dict[str, Any], *, tie_epsilon: float = 0.0) -> dict[str, Any]:
    """Raw distributions, group spread, missingness and observed model versions.

    A zero-spread group carries no ranking signal for this score. It is not
    classified as too difficult, too easy, incorrect, or safe to discard.
    """
    if not math.isfinite(tie_epsilon) or tie_epsilon < 0:
        raise ValueError("tie_epsilon must be finite and non-negative")
    records = list(evaluation["records"].values())
    successful = [row for row in records if row["status"] == "success"]
    axes = sorted({key for row in successful for key in row["result"]["scores"]})
    report: dict[str, Any] = {
        "run_id": evaluation["run_id"],
        "status_counts": dict(Counter(row["status"] for row in records)),
        "error_types": dict(
            Counter(
                row["error"].get("type", "unknown") for row in records if row["status"] == "error"
            )
        ),
        "model_versions": dict(
            Counter(str(row["result"].get("reward_model_version")) for row in successful)
        ),
        "axes": {},
        "tie_epsilon": tie_epsilon,
        "timing_ms": {},
    }
    timing_keys = sorted({key for row in successful for key in row["result"].get("timing_ms", {})})
    for key in timing_keys:
        report["timing_ms"][key] = distribution(
            [
                row["result"]["timing_ms"][key]
                for row in successful
                if key in row["result"].get("timing_ms", {})
            ]
        )
    for axis in axes:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in successful:
            if axis in row["result"]["scores"]:
                groups[row["input"]["prompt_id"]].append(float(row["result"]["scores"][axis]))
        values = [value for group in groups.values() for value in group]
        prompt_means = [statistics.fmean(group) for group in groups.values()]
        eligible = {key: group for key, group in groups.items() if len(group) >= 2}
        ties = [key for key, group in eligible.items() if max(group) - min(group) <= tie_epsilon]
        report["axes"][axis] = {
            "distribution": distribution(values),
            "missing_from_successful": len(successful) - len(values),
            "prompt_count": len(groups),
            "prompt_balanced_mean": statistics.fmean(statistics.fmean(g) for g in groups.values()),
            "prompt_mean_bootstrap_95ci": list(
                bootstrap_mean_interval(
                    prompt_means,
                    schema="vrl.reward-health.v1",
                    label=evaluation["run_id"],
                    score_key=axis,
                )
            )
            if len(prompt_means) >= 2
            else None,
            "multi_sample_prompt_count": len(eligible),
            "zero_spread_prompt_ids": sorted(ties),
            "zero_spread_fraction": len(ties) / len(eligible) if eligible else None,
            "prompt_ranges": {key: max(group) - min(group) for key, group in groups.items()},
        }
    return report


def compare_rankings(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    first_axis: str,
    second_axis: str,
    first_direction: int = 1,
    second_direction: int = 1,
    first_tie_epsilon: float = 0.0,
    second_tie_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Compare within-prompt orderings on the identical complete media grid.

    Agreement is consistency, not human preference accuracy. Score scales need
    not match. No implicit intersection drops failed or missing samples.
    """
    if first_direction not in {-1, 1} or second_direction not in {-1, 1}:
        raise ValueError("axis directions must be -1 or 1")
    for epsilon in (first_tie_epsilon, second_tie_epsilon):
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("tie tolerances must be finite and non-negative")
    a, b = first["records"], second["records"]
    if a.keys() != b.keys():
        raise ValueError("candidate sample grids differ")
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id, row in a.items():
        other = b[sample_id]
        if row["input"] != other["input"]:
            raise ValueError(f"candidate inputs differ: {sample_id}")
        if row["status"] != "success" or other["status"] != "success":
            raise ValueError(f"candidate comparison requires complete scoring: {sample_id}")
        if (
            first_axis not in row["result"]["scores"]
            or second_axis not in other["result"]["scores"]
        ):
            raise ValueError(f"candidate score axis missing: {sample_id}")
        groups[row["input"]["prompt_id"]].append(sample_id)
    per_prompt, disagreements = {}, []
    for prompt_id, ids in sorted(groups.items()):
        counts: Counter[str] = Counter()
        for left, right in itertools.combinations(sorted(ids), 2):
            delta_a = first_direction * (
                a[left]["result"]["scores"][first_axis] - a[right]["result"]["scores"][first_axis]
            )
            delta_b = second_direction * (
                b[left]["result"]["scores"][second_axis]
                - b[right]["result"]["scores"][second_axis]
            )
            sign_a = 0 if abs(delta_a) <= first_tie_epsilon else 1 if delta_a > 0 else -1
            sign_b = 0 if abs(delta_b) <= second_tie_epsilon else 1 if delta_b > 0 else -1
            category = (
                "agree"
                if sign_a == sign_b
                else "tie_disagreement"
                if 0 in (sign_a, sign_b)
                else "reversed"
            )
            counts[category] += 1
            if category != "agree":
                disagreements.append(
                    {
                        "prompt_id": prompt_id,
                        "left": left,
                        "right": right,
                        "first_delta": delta_a,
                        "second_delta": delta_b,
                        "kind": category,
                    }
                )
        if counts:
            per_prompt[prompt_id] = {
                "pairs": sum(counts.values()),
                **counts,
                "agreement": counts["agree"] / sum(counts.values()),
            }
    means = [row["agreement"] for row in per_prompt.values()]
    return {
        "first_run_id": first["run_id"],
        "second_run_id": second["run_id"],
        "first_axis": first_axis,
        "second_axis": second_axis,
        "directions": [first_direction, second_direction],
        "tie_epsilons": [first_tie_epsilon, second_tie_epsilon],
        "prompt_balanced_agreement": statistics.fmean(means) if means else None,
        "bootstrap_95ci": list(
            bootstrap_mean_interval(
                means,
                schema="vrl.reward-ranking.v1",
                label=first["run_id"],
                score_key=second["run_id"],
            )
        )
        if len(means) >= 2
        else None,
        "per_prompt": per_prompt,
        "disagreements": disagreements,
        "interpretation": "Ranking consistency only; no preference or quality labels used.",
    }
