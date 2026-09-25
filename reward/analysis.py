"""Model-free reports over scoring runs: health, stress, ranking and paired comparisons."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

from vrl.rewards.evaluation import Evaluation
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.sequences import EditSequenceSpec
from vrl.utils.json_files import canonical_json_sha256
from vrl.utils.score_statistics import bootstrap_mean_interval, distribution

JOIN_SCHEMA = "vrl.reward-evaluation-join.v1"


class Analysis:
    """Reports over one scoring run; comparisons take the other runs as arguments.

    Every report keeps failed and missing samples visible and never turns them
    into zero scores. Positive deltas mean numerically larger scores, not
    better quality.
    """

    def __init__(self, evaluation: Evaluation) -> None:
        self.evaluation = evaluation

    # ── one run ──────────────────────────────────────────────────────────────

    def health(self, *, tie_epsilon: float = 0.0) -> dict[str, Any]:
        """Score distributions, per-prompt spread, missingness and observed model versions.

        A zero-spread prompt group carries no ranking signal for that axis; it
        is not classified as too easy, too hard, wrong or safe to discard.
        """

        if not math.isfinite(tie_epsilon) or tie_epsilon < 0:
            raise ValueError("tie_epsilon must be finite and non-negative")
        evaluation = self.evaluation
        records = list(evaluation.records.values())
        successful = [row for row in records if row["status"] == "success"]
        axes = sorted({key for row in successful for key in row["result"]["scores"]})
        report: dict[str, Any] = {
            "run_id": evaluation.run_id,
            "status_counts": dict(Counter(row["status"] for row in records)),
            "error_types": dict(
                Counter(
                    row["error"].get("type", "unknown")
                    for row in records
                    if row["status"] == "error"
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
        timing_keys = sorted(
            {key for row in successful for key in row["result"].get("timing_ms", {})}
        )
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
            ties = [
                key for key, group in eligible.items() if max(group) - min(group) <= tie_epsilon
            ]
            report["axes"][axis] = {
                "distribution": distribution(values),
                "missing_from_successful": len(successful) - len(values),
                "prompt_count": len(groups),
                "prompt_balanced_mean": statistics.fmean(prompt_means),
                "prompt_mean_bootstrap_95ci": list(
                    bootstrap_mean_interval(
                        prompt_means,
                        schema="vrl.reward-health.v1",
                        label=evaluation.run_id,
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

    def stress(self) -> dict[str, Any]:
        """Pair each perturbed sample with its baseline (see ``reward.stress``)."""

        records = self.evaluation.records
        observations, by_transform = [], defaultdict(lambda: defaultdict(list))
        for sample_id, row in records.items():
            probe = row["input"].get("metadata", {}).get("reward_stress")
            if not isinstance(probe, dict):
                raise ValueError("stress report requires a generated stress manifest")
            baseline = records.get(probe.get("baseline_id"))
            if baseline is None:
                raise ValueError("stress baseline is absent from evaluation inputs")
            if baseline["input"].get("prompt") != row["input"].get("prompt"):
                raise ValueError("stress baseline identity or task differs")
            transform = probe.get("transform")
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
                for axis, score in scores.items():
                    if axis in original:
                        delta = score - original[axis]
                        observation["deltas"][axis] = delta
                        by_transform[transform][axis].append(delta)
            observations.append(observation)
        return {
            "run_id": self.evaluation.run_id,
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
        }

    def sequence(self, spec: EditSequenceSpec) -> dict[str, Any]:
        """Requirement preservation across an ordered sequence of states."""

        return spec.report(self.evaluation)

    def spread(
        self,
        *,
        axis: str,
        threshold: float,
        band: tuple[float, float] = (0.2, 0.6),
        tie_epsilon: float = 0.0,
    ) -> dict[str, Any]:
        """Does a run sampled like training carry a GRPO signal on ``axis``?

        Rows share a ``prompt_id`` per group. A sample succeeds at
        ``score >= threshold`` (the operating point fitted on labels). The
        report gives the success rate against the target band, the share of
        prompts whose group holds both successes and failures, and the
        within-prompt score spread -- a high-AUC judge whose scores barely
        differ within a group still gives the policy nothing to climb.
        """

        if not (0.0 <= band[0] < band[1] <= 1.0):
            raise ValueError("band must be 0 <= low < high <= 1")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        groups: dict[str, list[float]] = defaultdict(list)
        for row in self.evaluation.records.values():
            if row["status"] == "success" and axis in row["result"]["scores"]:
                groups[row["input"]["prompt_id"]].append(float(row["result"]["scores"][axis]))
        if not groups:
            raise ValueError(f"no successful scores on axis {axis!r}")
        multi = {key: values for key, values in groups.items() if len(values) >= 2}
        if not multi:
            raise ValueError("spread needs prompts with at least two samples")
        prompts = {}
        for key, values in groups.items():
            successes = sum(value >= threshold for value in values)
            prompts[key] = {
                "samples": len(values),
                "success_share": successes / len(values),
                "mixed": 0 < successes < len(values),
                "within_std": statistics.pstdev(values) if len(values) >= 2 else None,
                "score_range": max(values) - min(values),
            }
        all_values = [value for values in groups.values() for value in values]
        success_rate = sum(value >= threshold for value in all_values) / len(all_values)
        return {
            "run_id": self.evaluation.run_id,
            "axis": axis,
            "threshold": threshold,
            "band": list(band),
            "sample_count": len(all_values),
            "prompt_count": len(groups),
            "multi_sample_prompt_count": len(multi),
            "success_rate": success_rate,
            "in_band": band[0] <= success_rate <= band[1],
            "mixed_prompt_share": sum(prompts[key]["mixed"] for key in multi) / len(multi),
            "mean_within_prompt_std": statistics.fmean(
                prompts[key]["within_std"] for key in multi
            ),
            "zero_spread_prompt_share": sum(
                prompts[key]["score_range"] <= tie_epsilon for key in multi
            )
            / len(multi),
            "prompts": prompts,
        }

    # ── two or more runs ─────────────────────────────────────────────────────

    def compare_rankings(
        self,
        other: Evaluation,
        *,
        first_axis: str,
        second_axis: str,
        first_direction: int = 1,
        second_direction: int = 1,
        first_tie_epsilon: float = 0.0,
        second_tie_epsilon: float = 0.0,
    ) -> dict[str, Any]:
        """Within-prompt ordering agreement between this run's axis and another's.

        Agreement is consistency between two scorers, not preference accuracy.
        Both runs must have scored the same complete sample grid.
        """

        if first_direction not in {-1, 1} or second_direction not in {-1, 1}:
            raise ValueError("axis directions must be -1 or 1")
        for epsilon in (first_tie_epsilon, second_tie_epsilon):
            if not math.isfinite(epsilon) or epsilon < 0:
                raise ValueError("tie tolerances must be finite and non-negative")
        first, second = self.evaluation, other
        a, b = first.records, second.records
        if a.keys() != b.keys():
            raise ValueError("candidate sample grids differ")
        groups: dict[str, list[str]] = defaultdict(list)
        for sample_id, row in a.items():
            mate = b[sample_id]
            if row["input"] != mate["input"]:
                raise ValueError(f"candidate inputs differ: {sample_id}")
            if row["status"] != "success" or mate["status"] != "success":
                raise ValueError(f"candidate comparison requires complete scoring: {sample_id}")
            if (
                first_axis not in row["result"]["scores"]
                or second_axis not in mate["result"]["scores"]
            ):
                raise ValueError(f"candidate score axis missing: {sample_id}")
            groups[row["input"]["prompt_id"]].append(sample_id)
        per_prompt, disagreements = {}, []
        for prompt_id, ids in sorted(groups.items()):
            counts: Counter[str] = Counter()
            for left, right in itertools.combinations(sorted(ids), 2):
                delta_a = first_direction * (
                    a[left]["result"]["scores"][first_axis]
                    - a[right]["result"]["scores"][first_axis]
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
            "first_run_id": first.run_id,
            "second_run_id": second.run_id,
            "first_axis": first_axis,
            "second_axis": second_axis,
            "directions": [first_direction, second_direction],
            "tie_epsilons": [first_tie_epsilon, second_tie_epsilon],
            "prompt_balanced_agreement": statistics.fmean(means) if means else None,
            "bootstrap_95ci": list(
                bootstrap_mean_interval(
                    means,
                    schema="vrl.reward-ranking.v1",
                    label=first.run_id,
                    score_key=second.run_id,
                )
            )
            if len(means) >= 2
            else None,
            "per_prompt": per_prompt,
            "disagreements": disagreements,
        }

    @staticmethod
    def paired(
        baselines: list[Evaluation],
        candidates: list[Evaluation],
        *,
        axis: str,
        direction: int = 1,
        stratify_by: str | None = None,
    ) -> dict[str, Any]:
        """Compare two generators' outputs under one frozen scorer, weighting sources equally.

        Each baseline/candidate run pair shares the same sample grid, prompts,
        assets and metadata; only the output media differ. Repeated draws of a
        source are averaged before sources are averaged.
        """

        if not baselines or len(baselines) != len(candidates):
            raise ValueError("paired outputs need equally many non-empty baseline/candidate runs")
        if not axis or type(direction) is not int or direction not in (-1, 1):
            raise ValueError("paired outputs need an axis and direction +1 or -1")
        if stratify_by is not None and (not isinstance(stratify_by, str) or not stratify_by):
            raise ValueError("paired stratification needs a non-empty metadata key")
        recipe = baselines[0].config
        observations, assignments = [], {}
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for repeat, (baseline, candidate) in enumerate(zip(baselines, candidates, strict=True)):
            if baseline.config != recipe or candidate.config != recipe:
                raise ValueError("paired scoring recipes differ")
            first, second = baseline.records, candidate.records
            if not first or first.keys() != second.keys():
                raise ValueError("paired sample grids differ or are empty")
            for sample_id in sorted(first):
                left, right = first[sample_id], second[sample_id]
                item = left["input"]
                if any(
                    item.get(key) != right["input"].get(key)
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
                source = (
                    f"reference:{source_hash}" if source_hash else f"prompt:{item['prompt_id']}"
                )
                group = item.get("metadata", {}).get("source_group", source)
                if assignments.setdefault(source, group) != group:
                    raise ValueError("one source cannot belong to multiple paired groups")
                row = {
                    "repeat": repeat,
                    "sample_id": sample_id,
                    "prompt_id": item["prompt_id"],
                    "source_group": group,
                    "baseline_status": left["status"],
                    "candidate_status": right["status"],
                    "baseline_sha256": item.get("sha256"),
                    "candidate_sha256": right["input"].get("sha256"),
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
                if row["baseline"] is not None and row["candidate"] is not None:
                    row["directed_delta"] = direction * (row["candidate"] - row["baseline"])
                observations.append(row)
                groups[group].append(row)
        per_source = {}
        for group, rows in sorted(groups.items()):
            scored_rows = [row for row in rows if row["directed_delta"] is not None]
            per_source[group] = {
                "expected_pairs": len(rows),
                "scored_pairs": len(scored_rows),
                "prompt_ids": sorted({row["prompt_id"] for row in rows}),
                "baseline": statistics.fmean(r["baseline"] for r in scored_rows)
                if scored_rows
                else None,
                "candidate": statistics.fmean(r["candidate"] for r in scored_rows)
                if scored_rows
                else None,
                "directed_delta": statistics.fmean(r["directed_delta"] for r in scored_rows)
                if scored_rows
                else None,
            }
        scored = [row for row in per_source.values() if row["directed_delta"] is not None]
        deltas = [row["directed_delta"] for row in scored]
        runs = {
            "baseline": [r.run_id for r in baselines],
            "candidate": [r.run_id for r in candidates],
        }
        report = {
            "run_ids": runs,
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
        }
        if stratify_by is not None:
            strata: dict[str, dict[str, Any]] = {}
            for repeat, (baseline, candidate) in enumerate(
                zip(baselines, candidates, strict=True)
            ):
                for sample_id, row in baseline.records.items():
                    metadata = row["input"].get("metadata", {})
                    if stratify_by not in metadata:
                        raise ValueError(f"stratification metadata missing: {sample_id}")
                    value = metadata[stratify_by]
                    if type(value) not in (str, int, bool):
                        raise ValueError(
                            "stratification values must be strings, integers or booleans"
                        )
                    # JSON distinguishes 1, true and "1"; ordinary dict keys do not.
                    key = json.dumps(value, ensure_ascii=False)
                    bucket = strata.setdefault(key, {"value": value, "runs": {}})
                    left, right = bucket["runs"].setdefault(repeat, ({}, {}))
                    left[sample_id] = row
                    right[sample_id] = candidate.records[sample_id]
            report["stratification"] = {
                "metadata_key": stratify_by,
                "strata": [
                    {
                        "value": bucket["value"],
                        "comparison": Analysis.paired(
                            [
                                replace(baselines[index], records=records[0])
                                for index, records in sorted(bucket["runs"].items())
                            ],
                            [
                                replace(candidates[index], records=records[1])
                                for index, records in sorted(bucket["runs"].items())
                            ],
                            axis=axis,
                            direction=direction,
                        ),
                    }
                    for _, bucket in sorted(strata.items())
                ],
            }
        return report

    @staticmethod
    def repeatability(evaluations: list[Evaluation]) -> dict[str, Any]:
        """Repeat-scoring variation on identical inputs under one recipe."""

        if len(evaluations) < 2:
            raise ValueError("repeatability needs at least two scoring runs")
        first = evaluations[0]
        records = first.records
        if not records:
            raise ValueError("repeatability needs non-empty scoring inputs")
        for evaluation in evaluations[1:]:
            if evaluation.config != first.config:
                raise ValueError("repeatability scorer configurations differ")
            if evaluation.records.keys() != records.keys():
                raise ValueError("repeatability sample grids differ")
            if any(
                evaluation.records[key]["input"] != row["input"] for key, row in records.items()
            ):
                raise ValueError("repeatability inputs differ")
        source_groups = {
            key: row["input"].get("metadata", {}).get("source_group", row["input"]["prompt_id"])
            for key, row in records.items()
        }
        axes = sorted(
            {
                axis
                for evaluation in evaluations
                for row in evaluation.records.values()
                if row["status"] == "success"
                for axis in row["result"]["scores"]
            }
        )
        reports = {}
        for axis in axes:
            samples, grouped = [], defaultdict(list)
            statuses: Counter[str] = Counter()
            for key in records:
                scores, sample_statuses = [], []
                for evaluation in evaluations:
                    row = evaluation.records[key]
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
                    statistics.fmean(r["score_range"] for r in rows) for rows in grouped.values()
                )
                if grouped
                else None,
                "source_balanced_mean_sample_std": statistics.fmean(
                    statistics.fmean(r["sample_std"] for r in rows) for rows in grouped.values()
                )
                if grouped
                else None,
                "max_score_range": max(
                    (r["score_range"] for rows in grouped.values() for r in rows), default=None
                ),
            }
        return {
            "run_ids": [evaluation.run_id for evaluation in evaluations],
            "repeat_count": len(evaluations),
            "sample_count": len(records),
            "source_group_count": len(set(source_groups.values())),
            "status_counts_by_repeat": [
                dict(Counter(row["status"] for row in evaluation.records.values()))
                for evaluation in evaluations
            ],
            "axes": reports,
        }

    @staticmethod
    def join(evaluations: Mapping[str, Evaluation]) -> Evaluation:
        """Align independent scorers of the same samples into one run with prefixed axes.

        Inputs must match exactly across scorers. The joined recipe holds the
        scorer recipes under their aliases, so a combination fitted on it can
        score a separate holdout scored by the same scorers.
        """

        if not evaluations or any(not name or "/" in name for name in evaluations):
            raise ValueError("scorer aliases must be non-empty identifiers without slashes")
        ordered = dict(sorted(evaluations.items()))
        first = next(iter(ordered.values())).records
        if not first:
            raise ValueError("cannot join empty evaluations")
        for name, evaluation in ordered.items():
            if evaluation.records.keys() != first.keys():
                raise ValueError(f"scorer sample grids differ: {name}")
        records = {}
        for sample_id in sorted(first):
            item = first[sample_id]["input"]
            scores, timings, sources, evidence = {}, {}, {}, {}
            for name, evaluation in ordered.items():
                row = evaluation.records[sample_id]
                if row["input"] != item:
                    raise ValueError(f"scorer inputs differ: {name}/{sample_id}")
                if row["status"] != "success":
                    raise ValueError(f"joining requires complete scoring: {name}/{sample_id}")
                result = RewardInferenceResult(**row["result"])
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
            "scorers": {name: evaluation.config for name, evaluation in ordered.items()},
        }
        source_run_ids = {name: evaluation.run_id for name, evaluation in ordered.items()}
        run_id = canonical_json_sha256(
            {"config": recipe, "source_run_ids": source_run_ids, "records": records},
            allow_nan=False,
        )
        return Evaluation(
            run_id, recipe, records, schema=JOIN_SCHEMA, source_run_ids=source_run_ids
        )


__all__ = ["JOIN_SCHEMA", "Analysis"]
