"""Bind exact-count targets to paired Anima archives and summarize comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vrl.scripts.eval._score_summary import bootstrap_mean_interval, distribution
from vrl.scripts.families.cosmos.anima.generation_protocol import (
    ANIMA_ANCHOR_MANIFEST_SCHEMA,
    AnimaGenerationArchive,
    AnimaGenerationCell,
)

# Persisted report protocol and its fixed blinding policy.
REPORT_SCHEMA = "vrl.anima-exact-count-checkpoint-eval/v3"
DEFAULT_BLIND_SEED = 20260831
_GRID_DIGEST_SCHEMA = "vrl.anima-exact-count-grid-digest/v1"
_GRID_DIGEST_HEADER = _GRID_DIGEST_SCHEMA.encode("ascii") + b"\0"


@dataclass(frozen=True, slots=True)
class ExactCountCell:
    """Evaluator-owned arm and target bound to one generic archive cell."""

    arm: str
    generated: AnimaGenerationCell
    expected_people: int


@dataclass(frozen=True, slots=True)
class ScoredCell:
    """One generated cell with exact-count reward diagnostics."""

    cell: ExactCountCell
    scores: dict[str, float]


def _bind_exact_count_arm(
    archive: AnimaGenerationArchive,
    *,
    label: str,
) -> list[ExactCountCell]:
    cells: list[ExactCountCell] = []
    for generated in archive.cells:
        expected_people = generated.reward_metadata.get("expected_people")
        if type(expected_people) is not int or expected_people < 1:
            raise ValueError(
                f"{archive.directory}/metadata.jsonl needs positive integer "
                "reward_metadata.expected_people",
            )
        prompt_expected_people = generated.prompt_metadata.get("expected_people")
        if "expected_people" in generated.prompt_metadata and (
            type(prompt_expected_people) is not int
            or prompt_expected_people < 1
            or prompt_expected_people != expected_people
        ):
            raise ValueError(
                f"{archive.directory}/metadata.jsonl has conflicting "
                "prompt_metadata.expected_people and reward_metadata.expected_people",
            )
        cells.append(
            ExactCountCell(
                arm=label,
                generated=generated,
                expected_people=expected_people,
            ),
        )
    return cells


def _pair_rows(
    scored: Sequence[ScoredCell],
    *,
    checkpoint_label: str,
    reward_key: str,
    observed_count_key: str | None,
) -> list[dict[str, Any]]:
    by_arm: dict[str, dict[tuple[int, int], ScoredCell]] = defaultdict(dict)
    for row in scored:
        by_arm[row.cell.arm][row.cell.generated.key] = row
    base = by_arm["base"]
    checkpoint = by_arm[checkpoint_label]
    rows: list[dict[str, Any]] = []
    for key in sorted(base):
        base_reward = int(base[key].scores[reward_key])
        checkpoint_reward = int(checkpoint[key].scores[reward_key])
        base_generated = base[key].cell.generated
        checkpoint_generated = checkpoint[key].cell.generated
        record = {
            "prompt_index": key[0],
            "sample_index": key[1],
            "seed": base_generated.seed,
            "prompt": base_generated.prompt,
            "expected_people": base[key].cell.expected_people,
            "base_image_path": str(base_generated.image_path),
            "base_image_sha256": base_generated.image_sha256,
            "checkpoint_image_path": str(checkpoint_generated.image_path),
            "checkpoint_image_sha256": checkpoint_generated.image_sha256,
            "checkpoint_label": checkpoint_label,
            "base_reward": base_reward,
            "checkpoint_reward": checkpoint_reward,
            "delta": checkpoint_reward - base_reward,
        }
        if observed_count_key is not None:
            record.update(
                {
                    "base_observed_count": int(base[key].scores[observed_count_key]),
                    "checkpoint_observed_count": int(checkpoint[key].scores[observed_count_key]),
                },
            )
        rows.append(record)
    return rows


def _summarize(
    scored: Sequence[ScoredCell],
    pair_rows: Sequence[dict[str, Any]],
    *,
    checkpoint_label: str,
    reward_record: dict[str, Any],
    base_archive: AnimaGenerationArchive,
    checkpoint_archive: AnimaGenerationArchive,
    base_cells: Sequence[ExactCountCell],
    checkpoint_cells: Sequence[ExactCountCell],
    reward_key: str,
    observed_count_key: str | None,
) -> dict[str, Any]:
    grouped: dict[str, list[ScoredCell]] = defaultdict(list)
    for row in scored:
        grouped[row.cell.arm].append(row)

    image_deltas = [float(row["delta"]) for row in pair_rows]
    prompt_deltas: dict[int, list[float]] = defaultdict(list)
    for row in pair_rows:
        prompt_deltas[int(row["prompt_index"])].append(float(row["delta"]))
    prompt_mean_deltas = [
        statistics.fmean(prompt_deltas[index]) for index in sorted(prompt_deltas)
    ]
    lower, upper = bootstrap_mean_interval(
        prompt_mean_deltas,
        schema=REPORT_SCHEMA,
        label=checkpoint_label,
        score_key="exact_count",
    )
    improved = sum(delta > 0 for delta in image_deltas)
    regressed = sum(delta < 0 for delta in image_deltas)
    non_tied = improved + regressed
    prompt_improved = sum(delta > 0 for delta in prompt_mean_deltas)
    prompt_regressed = sum(delta < 0 for delta in prompt_mean_deltas)

    arms = {
        label: _arm_summary(
            rows,
            reward_key=reward_key,
            observed_count_key=observed_count_key,
        )
        for label, rows in (
            ("base", grouped["base"]),
            (checkpoint_label, grouped[checkpoint_label]),
        )
    }
    return {
        "schema": REPORT_SCHEMA,
        "reward_protocol": reward_record,
        "sources": {
            "base": _source_record(base_archive, base_cells),
            checkpoint_label: _source_record(
                checkpoint_archive,
                checkpoint_cells,
            ),
        },
        "arms": arms,
        "paired": {
            "pairs": len(pair_rows),
            "delta_reward_rate": statistics.fmean(image_deltas),
            "improved": improved,
            "regressed": regressed,
            "tied": len(image_deltas) - non_tied,
            "image_delta_distribution": distribution(image_deltas),
            "prompt_mean_delta_distribution": distribution(prompt_mean_deltas),
            "prompt_cluster_bootstrap_95ci": [lower, upper],
            "two_sided_sign_test_p": _two_sided_sign_test(improved, regressed),
            "prompt_cluster_two_sided_sign_test_p": _two_sided_sign_test(
                prompt_improved,
                prompt_regressed,
            ),
            "reward_rate_by_target": _paired_reward_rate_by_target(pair_rows),
            "clear_improvement": lower > 0.0,
            "clear_regression": upper < 0.0,
        },
    }


def _arm_summary(
    rows: Sequence[ScoredCell],
    *,
    reward_key: str,
    observed_count_key: str | None,
) -> dict[str, Any]:
    rewards = [int(row.scores[reward_key]) for row in rows]
    grouped: dict[int, list[ScoredCell]] = defaultdict(list)
    for row in rows:
        grouped[row.cell.generated.prompt_index].append(row)
    per_prompt: dict[str, Any] = {}
    for prompt_index in sorted(grouped):
        prompt_rows = grouped[prompt_index]
        prompt_rewards = [int(row.scores[reward_key]) for row in prompt_rows]
        per_prompt[str(prompt_index)] = {
            "expected_people": prompt_rows[0].cell.expected_people,
            "positive": sum(prompt_rewards),
            "rewards": prompt_rewards,
            "active": len(set(prompt_rewards)) > 1,
        }
    targets = sorted({row.cell.expected_people for row in rows})
    summary = {
        "images": len(rows),
        "positive": sum(rewards),
        "reward_rate": statistics.fmean(rewards),
        "active_groups": sum(prompt["active"] for prompt in per_prompt.values()),
        "reward_rate_by_target": {
            str(target): statistics.fmean(
                int(row.scores[reward_key]) for row in rows if row.cell.expected_people == target
            )
            for target in targets
        },
        "per_prompt": per_prompt,
    }
    if "codex_image_qa_mirror_agreement" in rows[0].scores:
        summary["mirror_agreement_rate"] = statistics.fmean(
            row.scores["codex_image_qa_mirror_agreement"] for row in rows
        )
    if observed_count_key is not None:
        observed_counts = [int(row.scores[observed_count_key]) for row in rows]
        summary.update(
            {
                "observed_count_histogram": {
                    str(value): count for value, count in sorted(Counter(observed_counts).items())
                },
                "mean_absolute_count_error": statistics.fmean(
                    abs(observed - row.cell.expected_people)
                    for observed, row in zip(observed_counts, rows, strict=True)
                ),
            },
        )
    return summary


def _paired_reward_rate_by_target(
    pair_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[int(row["expected_people"])].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for target, rows in sorted(grouped.items()):
        deltas = [float(row["delta"]) for row in rows]
        improved = sum(delta > 0 for delta in deltas)
        regressed = sum(delta < 0 for delta in deltas)
        result[str(target)] = {
            "pairs": len(rows),
            "base_reward_rate": statistics.fmean(float(row["base_reward"]) for row in rows),
            "checkpoint_reward_rate": statistics.fmean(
                float(row["checkpoint_reward"]) for row in rows
            ),
            "delta_reward_rate": statistics.fmean(deltas),
            "improved": improved,
            "regressed": regressed,
            "tied": len(rows) - improved - regressed,
        }
    return result


def _source_record(
    archive: AnimaGenerationArchive,
    cells: Sequence[ExactCountCell],
) -> dict[str, Any]:
    return {
        "directory": str(archive.directory),
        "metadata_sha256": archive.metadata_sha256,
        "run_config_sha256": archive.run_config_sha256,
        "anchor_manifest": {
            "schema": ANIMA_ANCHOR_MANIFEST_SCHEMA,
            "sha256": archive.anchor_manifest_sha256,
        },
        "grid_digest": {
            "schema": _GRID_DIGEST_SCHEMA,
            "cells": len(cells),
            "sha256": _grid_digest(cells),
        },
        "model": archive.run_config.get("model"),
        "sampling": archive.run_config.get("sampling"),
        "prompt_source": archive.run_config.get("prompt_source"),
    }


def _grid_digest(cells: Sequence[ExactCountCell]) -> str:
    digest = hashlib.sha256(_GRID_DIGEST_HEADER)
    for cell in sorted(cells, key=lambda item: item.generated.key):
        generated = cell.generated
        record = {
            "prompt_index": generated.prompt_index,
            "sample_index": generated.sample_index,
            "seed": generated.seed,
            "prompt": generated.prompt,
            "expected_people": cell.expected_people,
            "reward_metadata": generated.reward_metadata,
            "image_sha256": generated.image_sha256,
        }
        payload = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _two_sided_sign_test(improved: int, regressed: int) -> float:
    count = improved + regressed
    if count == 0:
        return 1.0
    lower_tail = sum(math.comb(count, index) for index in range(min(improved, regressed) + 1))
    return min(1.0, 2.0 * lower_tail / (2**count))
