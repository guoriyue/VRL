"""Model-free reward health and paired ranking reports from persisted evidence."""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrl.rewards.evaluation import evaluation_access
from vrl.rewards.inference import RewardInferenceResult
from vrl.utils.json_files import canonical_json_sha256
from vrl.utils.score_statistics import bootstrap_mean_interval, distribution


def read_evaluation(directory: Path) -> dict[str, Any]:
    """Read an immutable scoring snapshot, retaining failures and missing rows.

    Checks stored provenance and identities, without reopening potentially moved
    media or loading any model. Hashes attest the recorded inputs, not score truth.
    """
    with evaluation_access(directory, writing=False):
        provenance = json.loads((directory / "provenance.json").read_text())
        run_id = provenance.pop("run_id")
        if provenance.get("schema") != "vrl.reward-evaluation.v1":
            raise ValueError("unsupported evaluation schema")
        if canonical_json_sha256(provenance, allow_nan=False) != run_id:
            raise ValueError("evaluation provenance digest mismatch")
        inputs = provenance["inputs"]
        identities = [row["sample_id"] for row in inputs]
        if not inputs or len(set(identities)) != len(inputs):
            raise ValueError("evaluation inputs must be non-empty and uniquely identified")
        expected = {
            f"{canonical_json_sha256(sample_id, allow_nan=False)}.json" for sample_id in identities
        }
        unexpected = {p.name for p in (directory / "samples").glob("*.json")} - expected
        if unexpected:
            raise ValueError(f"unexpected sample records: {sorted(unexpected)}")
        records = {}
        for item in inputs:
            sample_id = item["sample_id"]
            path = (
                directory / "samples" / f"{canonical_json_sha256(sample_id, allow_nan=False)}.json"
            )
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


def join_evaluations(evaluations: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Align independent scorers without inference, imputation, or sample loss.

    Inputs must match exactly, including prompt, media and auxiliary file hashes.
    Names scope all score axes. The recipe excludes dataset/run identities so a
    frozen combination can score a separate holdout with the same scorer recipes.
    The derived run identity binds the complete observed results as well as source
    run IDs; it is an in-memory calibration view, not a rescorable ScoringConfig.
    """
    if not evaluations or any(
        not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) is None
        for name in evaluations
    ):
        raise ValueError("scorer aliases must be non-empty identifiers without slashes")
    ordered = dict(sorted(evaluations.items()))
    first = next(iter(ordered.values()))["records"]
    if not first:
        raise ValueError("cannot join empty evaluations")
    for name, evaluation in ordered.items():
        if evaluation["records"].keys() != first.keys():
            raise ValueError(f"scorer sample grids differ: {name}")
    records = {}
    for sample_id in sorted(first):
        item = first[sample_id]["input"]
        scores, timings, sources, evidence = {}, {}, {}, {}
        for name, evaluation in ordered.items():
            row = evaluation["records"][sample_id]
            if row["input"] != item:
                raise ValueError(f"scorer inputs differ: {name}/{sample_id}")
            if row["status"] != "success":
                raise ValueError(f"joining requires complete scoring: {name}/{sample_id}")
            result = RewardInferenceResult(**row["result"])
            if result.artifact_id != sample_id or not result.scores:
                raise ValueError(f"invalid scorer result: {name}/{sample_id}")
            scores.update({f"{name}/{key}": value for key, value in result.scores.items()})
            timings.update({f"{name}/{key}": value for key, value in result.timing_ms.items()})
            sources[name] = asdict(result)
            if result.diagnostics:
                evidence[name] = result.diagnostics
        records[sample_id] = {
            "input": item,
            "status": "success",
            "result": asdict(
                RewardInferenceResult(
                    artifact_id=sample_id,
                    scores=scores,
                    timing_ms=timings,
                    diagnostics={"scorers": evidence} if evidence else {},
                )
            ),
            "source_results": sources,
        }
    recipe = {
        "schema": "vrl.reward-scoring-join.v1",
        "scorers": {name: evaluation["config"] for name, evaluation in ordered.items()},
    }
    payload = {
        "schema": "vrl.reward-evaluation-join.v1",
        "config": recipe,
        "source_run_ids": {name: evaluation["run_id"] for name, evaluation in ordered.items()},
        "records": records,
    }
    return {"run_id": canonical_json_sha256(payload, allow_nan=False), **payload}


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
        "diagnostics": {
            "rows_with_evidence": sum(
                bool(row["result"].get("diagnostics")) for row in successful
            ),
            "why_counts": dict(
                Counter(
                    row["result"]["diagnostics"]["why"]
                    for row in successful
                    if isinstance(row["result"].get("diagnostics", {}).get("why"), str)
                )
            ),
        },
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


def compare_paired_outputs(
    baselines: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    axis: str,
    direction: int = 1,
    stratify_by: str | None = None,
) -> dict[str, Any]:
    """Compare generated outputs under one frozen scorer, grouping repeated sources.

    Each baseline/candidate snapshot pair must have the same sample grid, prompt,
    auxiliary assets and metadata; only output media may differ. This does not
    independently establish sampling parity unless it is recorded in metadata.
    Failed/missing rows remain visible and never acquire zero-valued measurements.
    """
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("paired outputs need equally many non-empty baseline/candidate runs")
    if not axis or type(direction) is not int or direction not in (-1, 1):
        raise ValueError("paired outputs need an axis and direction +1 or -1")
    if stratify_by is not None and (not isinstance(stratify_by, str) or not stratify_by):
        raise ValueError("paired stratification needs a non-empty metadata key")
    recipe = baselines[0]["config"]
    observations, assignments = [], {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for repeat, (baseline, candidate) in enumerate(zip(baselines, candidates, strict=True)):
        if baseline["config"] != recipe or candidate["config"] != recipe:
            raise ValueError("paired scoring recipes differ")
        first, second = baseline["records"], candidate["records"]
        if not first or first.keys() != second.keys():
            raise ValueError("paired sample grids differ or are empty")
        for sample_id in sorted(first):
            left, right = first[sample_id], second[sample_id]
            item = left["input"]
            if any(
                canonical_json_sha256(item.get(key), allow_nan=False)
                != canonical_json_sha256(right["input"].get(key), allow_nan=False)
                for key in (
                    "sample_id",
                    "prompt_id",
                    "prompt",
                    "assets",
                    "asset_sha256",
                    "metadata",
                )
            ):
                raise ValueError(f"paired task inputs differ: {sample_id}")
            source_hash = item.get("asset_sha256", {}).get("reference_image")
            source = f"reference:{source_hash}" if source_hash else f"prompt:{item['prompt_id']}"
            group = item.get("metadata", {}).get("source_group", source)
            if not isinstance(group, str) or not group:
                raise ValueError("paired source_group must be a non-empty string")
            if assignments.setdefault(source, group) != group:
                raise ValueError("one source cannot belong to multiple paired groups")
            row = {
                "repeat": repeat,
                "sample_id": sample_id,
                "prompt_id": item["prompt_id"],
                "source_group": group,
                "baseline_status": left["status"],
                "candidate_status": right["status"],
                "baseline_sha256": item["sha256"],
                "candidate_sha256": right["input"]["sha256"],
                "baseline": None,
                "candidate": None,
                "directed_delta": None,
            }
            for arm, record in (("baseline", left), ("candidate", right)):
                if record["status"] == "success":
                    try:
                        value = float(record["result"]["scores"][axis])
                    except KeyError as error:
                        raise ValueError(f"paired score axis missing: {sample_id}") from error
                    if not math.isfinite(value):
                        raise ValueError(f"paired score is nonfinite: {sample_id}")
                    row[arm] = value
                elif record["status"] == "error":
                    row[f"{arm}_error"] = record["error"]
                elif record["status"] != "missing":
                    raise ValueError(f"invalid paired sample status: {sample_id}")
            if row["baseline"] is not None and row["candidate"] is not None:
                delta = direction * (row["candidate"] - row["baseline"])
                if not math.isfinite(delta):
                    raise ValueError(f"paired difference is nonfinite: {sample_id}")
                row["directed_delta"] = delta
            observations.append(row)
            groups[group].append(row)
    per_source = {}
    for group, rows in sorted(groups.items()):
        paired = [row for row in rows if row["directed_delta"] is not None]
        per_source[group] = {
            "expected_pairs": len(rows),
            "scored_pairs": len(paired),
            "prompt_ids": sorted({row["prompt_id"] for row in rows}),
            "baseline": statistics.fmean(row["baseline"] for row in paired) if paired else None,
            "candidate": statistics.fmean(row["candidate"] for row in paired) if paired else None,
            "directed_delta": statistics.fmean(row["directed_delta"] for row in paired)
            if paired
            else None,
        }
    scored = [row for row in per_source.values() if row["directed_delta"] is not None]
    deltas = [row["directed_delta"] for row in scored]
    runs = {
        "baseline": [r["run_id"] for r in baselines],
        "candidate": [r["run_id"] for r in candidates],
    }
    report = {
        "schema": "vrl.paired-output-scores.v1",
        "run_ids": runs,
        "scoring_config_hash": canonical_json_sha256(recipe, allow_nan=False),
        "axis": axis,
        "direction": direction,
        "complete": all(row["directed_delta"] is not None for row in observations),
        "expected_pairs": len(observations),
        "scored_pairs": sum(row["directed_delta"] is not None for row in observations),
        "expected_source_groups": len(groups),
        "scored_source_groups": len(scored),
        "baseline": distribution([row["baseline"] for row in scored]) if scored else None,
        "candidate": distribution([row["candidate"] for row in scored]) if scored else None,
        "directed_delta": distribution(deltas) if deltas else None,
        "source_bootstrap_95ci": list(
            bootstrap_mean_interval(
                deltas,
                schema="vrl.paired-output-scores.v1",
                label=canonical_json_sha256(runs, allow_nan=False),
                score_key=axis,
            )
        )
        if len(deltas) >= 2
        else None,
        "per_source": per_source,
        "observations": observations,
        "interpretation": (
            "Source-balanced frozen-score comparison, conditional on successfully paired scores. "
            "Groups use metadata.source_group, then reference_image digest, then prompt_id. "
            "Missing results are not imputed. Sampling parity and semantic source independence "
            "remain operator-owned; score gain alone is not human-preference validation."
        ),
    }
    if stratify_by is not None:
        strata = {}
        for repeat, (baseline, candidate) in enumerate(zip(baselines, candidates, strict=True)):
            for sample_id, row in baseline["records"].items():
                metadata = row["input"].get("metadata", {})
                if stratify_by not in metadata:
                    raise ValueError(f"stratification metadata missing: {sample_id}")
                value = metadata[stratify_by]
                if type(value) not in (str, int, bool):
                    raise ValueError("stratification values must be strings, integers or booleans")
                # JSON distinguishes 1, true and "1"; ordinary dict keys do not.
                key = json.dumps(value, ensure_ascii=False)
                bucket = strata.setdefault(key, {"value": value, "runs": {}})
                left, right = bucket["runs"].setdefault(repeat, ({}, {}))
                left[sample_id] = row
                right[sample_id] = candidate["records"][sample_id]
        report["stratification"] = {
            "metadata_key": stratify_by,
            "strata": [
                {
                    "value": bucket["value"],
                    "comparison": compare_paired_outputs(
                        [
                            {**baselines[index], "records": records[0]}
                            for index, records in sorted(bucket["runs"].items())
                        ],
                        [
                            {**candidates[index], "records": records[1]}
                            for index, records in sorted(bucket["runs"].items())
                        ],
                        axis=axis,
                        direction=direction,
                    ),
                }
                for _, bucket in sorted(strata.items())
            ],
            "interpretation": (
                "Each stratum retains source grouping and missing/failed pairs. "
                "Strata may share sources; their intervals are descriptive and not "
                "adjusted for multiple comparisons. The overall comparison is unchanged."
            ),
        }
    return {"comparison_id": canonical_json_sha256(report, allow_nan=False), **report}


def stress_report(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Pair each perturbation with its own baseline, retaining missing/failed scores.

    Positive deltas mean numerically larger scores, not better quality. Axes may
    have opposite directions and perturbations do not establish semantic labels.
    """
    records = evaluation["records"]
    observations, by_transform = [], defaultdict(lambda: defaultdict(list))
    for sample_id, row in records.items():
        probe = row["input"].get("metadata", {}).get("reward_stress")
        if not isinstance(probe, dict) or probe.get("schema") != "vrl.reward-stress.v1":
            raise ValueError("stress report requires a generated stress manifest")
        baseline = records.get(probe.get("baseline_id"))
        if baseline is None:
            raise ValueError("stress baseline is absent from evaluation inputs")
        base_probe = baseline["input"].get("metadata", {}).get("reward_stress", {})
        if (
            base_probe.get("transform") != "baseline"
            or base_probe.get("baseline_id") != probe.get("baseline_id")
            or any(
                base_probe.get(key) != probe.get(key)
                for key in ("source_sample_id", "source_sha256", "seed")
            )
            or any(
                baseline["input"].get(key) != row["input"].get(key)
                for key in ("prompt_id", "prompt", "assets", "asset_sha256")
            )
        ):
            raise ValueError("stress baseline identity or task differs")
        baseline_metadata = {
            key: value
            for key, value in baseline["input"].get("metadata", {}).items()
            if key != "reward_stress"
        }
        candidate_metadata = {
            key: value
            for key, value in row["input"].get("metadata", {}).items()
            if key != "reward_stress"
        }
        if baseline_metadata != candidate_metadata:
            raise ValueError("stress baseline identity or task differs")
        transform = probe.get("transform")
        if not isinstance(transform, str) or not transform:
            raise ValueError("stress transform must be a non-empty string")
        if transform == "baseline":
            continue
        observation = {
            "sample_id": sample_id,
            "source_sample_id": probe["source_sample_id"],
            "baseline_id": probe["baseline_id"],
            "transform": transform,
            "status": row["status"],
            "baseline_status": baseline["status"],
            "deltas": {},
        }
        if row["status"] == baseline["status"] == "success":
            scores, original = row["result"]["scores"], baseline["result"]["scores"]
            if scores.keys() != original.keys():
                raise ValueError("stress score axes differ from baseline")
            for axis, score in scores.items():
                delta = score - original[axis]
                observation["deltas"][axis] = delta
                by_transform[transform][axis].append(delta)
        observations.append(observation)
    return {
        "run_id": evaluation["run_id"],
        "observations": observations,
        "transforms": {
            transform: {
                axis: {
                    "paired_delta": distribution(values),
                    "increases": sum(v > 0 for v in values),
                    "decreases": sum(v < 0 for v in values),
                    "ties": sum(v == 0 for v in values),
                }
                for axis, values in axes.items()
            }
            for transform, axes in by_transform.items()
        },
        "limitations": [
            "Score increases under perturbation flag cases for review, not automatic reward-hacking verdicts.",
            "Failed or missing pairs remain observations and never receive fabricated zero deltas.",
            "Summaries weight source samples equally; correlated variants are not independent tasks.",
        ],
    }


def repeatability_report(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure repeat scoring variation on identical inputs under one frozen recipe.

    Run IDs may legitimately match: they identify the scoring recipe and inputs,
    not an inference attempt. The caller must supply independently scored runs.
    """
    if len(evaluations) < 2:
        raise ValueError("repeatability needs at least two scoring runs")
    first = evaluations[0]
    records = first["records"]
    if not records:
        raise ValueError("repeatability needs non-empty scoring inputs")
    for evaluation in evaluations[1:]:
        if evaluation["config"] != first["config"]:
            raise ValueError("repeatability scorer configurations differ")
        if evaluation["records"].keys() != records.keys():
            raise ValueError("repeatability sample grids differ")
        if any(
            evaluation["records"][key]["input"] != row["input"] for key, row in records.items()
        ):
            raise ValueError("repeatability inputs differ")
    source_groups = {}
    for key, row in records.items():
        item = row["input"]
        group = item.get("metadata", {}).get("source_group", item["prompt_id"])
        if not isinstance(group, str) or not group:
            raise ValueError("repeatability source groups must be non-empty strings")
        source_groups[key] = group
    axes = sorted(
        {
            axis
            for evaluation in evaluations
            for row in evaluation["records"].values()
            if row["status"] == "success"
            for axis in row["result"]["scores"]
        }
    )
    reports = {}
    for axis in axes:
        samples, grouped = [], defaultdict(list)
        statuses = Counter()
        for key in records:
            scores, sample_statuses = [], []
            for evaluation in evaluations:
                row = evaluation["records"][key]
                status = row["status"]
                value = None
                if status == "success":
                    value = row["result"]["scores"].get(axis)
                    if value is None:
                        status = "missing_axis"
                scores.append(value)
                sample_statuses.append(status)
                statuses[status] += 1
            observed = [value for value in scores if value is not None]
            sample = {
                "sample_id": key,
                "source_group": source_groups[key],
                "scores": scores,
                "statuses": sample_statuses,
                "observed_repeats": len(observed),
                "score_range": max(observed) - min(observed) if len(observed) >= 2 else None,
                "sample_std": statistics.stdev(observed) if len(observed) >= 2 else None,
            }
            samples.append(sample)
            if len(observed) >= 2:
                grouped[source_groups[key]].append(sample)
        reports[axis] = {
            "samples": samples,
            "observation_status_counts": dict(statuses),
            "samples_with_two_scores": sum(len(rows) for rows in grouped.values()),
            "samples_with_all_scores": sum(
                row["observed_repeats"] == len(evaluations) for row in samples
            ),
            "source_groups_with_two_scores": len(grouped),
            "excluded_source_groups": sorted(set(source_groups.values()) - grouped.keys()),
            "source_balanced_mean_score_range": statistics.fmean(
                statistics.fmean(row["score_range"] for row in rows) for rows in grouped.values()
            )
            if grouped
            else None,
            "source_balanced_mean_sample_std": statistics.fmean(
                statistics.fmean(row["sample_std"] for row in rows) for rows in grouped.values()
            )
            if grouped
            else None,
            "max_score_range": max(
                (row["score_range"] for rows in grouped.values() for row in rows), default=None
            ),
        }
    return {
        "run_ids": [evaluation["run_id"] for evaluation in evaluations],
        "repeat_count": len(evaluations),
        "sample_count": len(records),
        "source_group_count": len(set(source_groups.values())),
        "status_counts_by_repeat": [
            dict(Counter(row["status"] for row in evaluation["records"].values()))
            for evaluation in evaluations
        ],
        "axes": reports,
        "limitations": [
            "Independent inference attempts are required; copying cached scores is not repeatability evidence.",
            "Variation describes these observations, not a calibrated uncertainty estimate or quality verdict.",
            "Summaries require at least two observed scores per sample; failures and missing axes are not zero scores.",
            "Sources are weighted equally; source grouping and near-duplicate separation remain operator-owned.",
        ],
    }
