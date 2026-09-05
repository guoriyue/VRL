"""Compare two completed Anima generation grids with an exact-count reward.

This evaluator never generates images. It validates a paired base/checkpoint
grid from each directory's ``metadata.jsonl``, takes ``expected_people`` only
from typed ``reward_metadata``, and scores either complete prompt groups with
the production Codex exact-count judge or individual images with pinned
CountGD. The report keeps per-image diagnostics, prompt-clustered paired
statistics, and deterministic blind contact sheets for human review.

CountGD must run under its isolated interpreter because upstream exposes
generic top-level packages such as ``models`` and ``util``::

    data/external/countgd/env/bin/python -m \
      vrl.scripts.eval.anima_exact_count_checkpoint_eval \
      --reward-backend countgd --base-dir ... --checkpoint-dir ... \
      --output-dir ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageOps

from vrl.config.loading import load_config
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.codex_image_qa import CodexImageQARewardModel
from vrl.rewards.models.countgd import (
    COUNTGD_CHECKPOINT_SHA256,
    COUNTGD_MODEL_VERSION,
    COUNTGD_RUNTIME_TREE_SHA256,
    COUNTGD_SCORE_KEY,
    COUNTGD_SOURCE_REVISION,
    COUNTGD_SPACE_REVISION,
    CountGDModel,
)
from vrl.rewards.types import REWARD_GROUP_ID_METADATA_KEY
from vrl.scripts.eval._score_summary import bootstrap_mean_interval, distribution
from vrl.scripts.families.cosmos.anima.generation_protocol import (
    ANIMA_ANCHOR_MANIFEST_SCHEMA,
    AnimaGenerationArchive,
    AnimaGenerationCell,
    validate_paired_generation_archives,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Completed base generation directory containing metadata.jsonl.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Completed checkpoint generation directory containing metadata.jsonl.",
    )
    parser.add_argument(
        "--checkpoint-label",
        default="checkpoint",
        help="Human-readable candidate arm label stored in the report.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Codex backend config supplying reward.kwargs.codex_image_qa. "
            "Defaults to the checkpoint generation's run_config.json config value."
        ),
    )
    parser.add_argument(
        "--reward-backend",
        choices=("codex", "countgd"),
        default="codex",
        help="Exact-count scorer backend (default: codex).",
    )
    parser.add_argument(
        "--countgd-source-dir",
        type=Path,
        default=None,
        help=(
            "Pinned CountGD source directory. Defaults to "
            "data/external/countgd/source through the model adapter."
        ),
    )
    parser.add_argument(
        "--countgd-device",
        default=None,
        help="CountGD inference device (default: cpu, the qualified service path).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="CountGD progress interval in scored images.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-seed", type=int, default=DEFAULT_BLIND_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint_label = args.checkpoint_label.strip()
    summary = create_report(
        base_dir=args.base_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_label=checkpoint_label,
        config=args.config,
        reward_backend=args.reward_backend,
        countgd_source_dir=args.countgd_source_dir,
        countgd_device=args.countgd_device,
        progress_every=args.progress_every,
        output_dir=args.output_dir,
        blind_seed=args.blind_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "base_reward_rate": summary["arms"]["base"]["reward_rate"],
                "checkpoint_reward_rate": summary["arms"][checkpoint_label]["reward_rate"],
                "delta_reward_rate": summary["paired"]["delta_reward_rate"],
            },
            indent=2,
            sort_keys=True,
        ),
    )


def create_report(
    *,
    base_dir: Path,
    checkpoint_dir: Path,
    checkpoint_label: str,
    config: str | Path | None,
    output_dir: Path,
    blind_seed: int = DEFAULT_BLIND_SEED,
    reward_backend: str = "codex",
    countgd_source_dir: Path | None = None,
    countgd_device: str | None = None,
    progress_every: int | None = None,
) -> dict[str, Any]:
    """Score, pair, summarize, and atomically publish one checkpoint comparison."""

    checkpoint_label = checkpoint_label.strip()
    if not checkpoint_label or checkpoint_label == "base":
        raise ValueError("checkpoint_label must be non-empty and differ from 'base'")
    if type(blind_seed) is not int:
        raise TypeError("blind_seed must be an integer")
    if reward_backend not in {"codex", "countgd"}:
        raise ValueError("reward_backend must be 'codex' or 'countgd'")

    base_dir = base_dir.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"exact-count evaluation output already exists: {output_dir}")

    base_archive = AnimaGenerationArchive.load(base_dir)
    checkpoint_archive = AnimaGenerationArchive.load(checkpoint_dir)
    base_cells = _bind_exact_count_arm(base_archive, label="base")
    checkpoint_cells = _bind_exact_count_arm(
        checkpoint_archive,
        label=checkpoint_label,
    )
    validate_paired_generation_archives(base_archive, checkpoint_archive)

    if reward_backend == "codex":
        countgd_options = {
            "countgd_source_dir": countgd_source_dir,
            "countgd_device": countgd_device,
            "progress_every": progress_every,
        }
        explicitly_set = sorted(key for key, value in countgd_options.items() if value is not None)
        if explicitly_set:
            raise ValueError(
                f"Codex reward backend cannot set CountGD options: {explicitly_set}",
            )
        config_source = config or checkpoint_archive.run_config.get("config")
        if not isinstance(config_source, (str, Path)) or not str(config_source).strip():
            raise ValueError(
                "no exact-count config was supplied and checkpoint run_config.json has no config",
            )
        reward_model, reward_record = _build_reward_model(config_source)
        _validate_group_size(base_cells, reward_model.expected_group_size)
        _validate_group_size(checkpoint_cells, reward_model.expected_group_size)
        scored = _score_cells(reward_model, [*base_cells, *checkpoint_cells])
        reward_key = "codex_image_qa"
        observed_count_key = None
    else:
        if config is not None:
            raise ValueError("CountGD reward backend cannot set the Codex --config option")
        countgd_device = countgd_device or "cpu"
        progress_every = 16 if progress_every is None else progress_every
        if progress_every < 1:
            raise ValueError("progress_every must be positive")
        reward_model, reward_record = _build_countgd_reward_model(
            source_dir=countgd_source_dir,
            device=countgd_device,
        )
        scored = _score_countgd_cells(
            reward_model,
            [*base_cells, *checkpoint_cells],
            progress_every=progress_every,
        )
        reward_key = COUNTGD_SCORE_KEY
        observed_count_key = "countgd_observed_count"

    score_rows = [
        _score_row(
            row,
            reward_key=reward_key,
            observed_count_key=observed_count_key,
        )
        for row in scored
    ]
    pair_rows = _pair_rows(
        scored,
        checkpoint_label=checkpoint_label,
        reward_key=reward_key,
        observed_count_key=observed_count_key,
    )
    summary = _summarize(
        scored,
        pair_rows,
        checkpoint_label=checkpoint_label,
        reward_record=reward_record,
        base_archive=base_archive,
        checkpoint_archive=checkpoint_archive,
        base_cells=base_cells,
        checkpoint_cells=checkpoint_cells,
        reward_key=reward_key,
        observed_count_key=observed_count_key,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ) as temporary_root:
        staging = Path(temporary_root) / "report"
        staging.mkdir()
        _write_jsonl(staging / "scores.jsonl", score_rows)
        _write_jsonl(staging / "pairs.jsonl", pair_rows)
        _write_json(staging / "summary.json", summary)
        _write_contact_sheets(
            base_cells,
            checkpoint_cells,
            checkpoint_label=checkpoint_label,
            blind_seed=int(blind_seed),
            output_dir=staging,
        )
        staging.replace(output_dir)
    return summary


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


def _build_reward_model(
    config_source: str | Path,
) -> tuple[CodexImageQARewardModel, dict[str, Any]]:
    cfg = load_config(config_source)
    raw_kwargs = OmegaConf.select(cfg, "reward.kwargs.codex_image_qa", default=None)
    plain_kwargs = OmegaConf.to_container(raw_kwargs, resolve=True)
    if not isinstance(plain_kwargs, dict):
        raise TypeError("config reward.kwargs.codex_image_qa must be a mapping")
    worker_config = plain_kwargs
    # Training saves visual reward artifacts; this evaluator owns its report
    # tree and must not append into a training run's scored_rollouts directory.
    worker_config["scored_rollout_dir"] = ""
    model = CodexImageQARewardModel(worker_config)
    if model.comparison_mode != "exact_count":
        raise ValueError(
            f"evaluation requires comparison_mode='exact_count', got {model.comparison_mode!r}",
        )
    if model.prompt_metadata_key != "expected_people":
        raise ValueError(
            "exact-person-count evaluation requires prompt_metadata_key='expected_people'",
        )
    effective_json = json.dumps(worker_config, sort_keys=True, separators=(",", ":"))
    return model, {
        "backend": "codex",
        "config": str(config_source),
        "effective_config_sha256": hashlib.sha256(effective_json.encode()).hexdigest(),
        "comparison_mode": model.comparison_mode,
        "expected_group_size": model.expected_group_size,
        "images_per_call": model.images_per_call,
        "prompt_metadata_key": model.prompt_metadata_key,
    }


def _build_countgd_reward_model(
    *,
    source_dir: Path | None,
    device: str,
) -> tuple[CountGDModel, dict[str, Any]]:
    worker_config: dict[str, Any] = {
        "device": device,
        "reward_model_version": COUNTGD_MODEL_VERSION,
    }
    if source_dir is not None:
        worker_config["source_dir"] = str(source_dir)
    model = CountGDModel(worker_config)
    return model, {
        "backend": "countgd",
        "criterion": "observed_count == expected_people",
        "source_revision": COUNTGD_SOURCE_REVISION,
        "space_revision": COUNTGD_SPACE_REVISION,
        "checkpoint_sha256": COUNTGD_CHECKPOINT_SHA256,
        "runtime_tree_sha256": COUNTGD_RUNTIME_TREE_SHA256,
        "model_version": COUNTGD_MODEL_VERSION,
        "source_dir": str(model.config.source_dir),
        "device": model.config.device,
        "python_executable": sys.executable,
    }


def _validate_group_size(cells: Sequence[ExactCountCell], expected_size: int) -> None:
    grouped: dict[int, list[ExactCountCell]] = defaultdict(list)
    for cell in cells:
        grouped[cell.generated.prompt_index].append(cell)
    for prompt_index, group in grouped.items():
        if len(group) != expected_size:
            raise ValueError(
                f"prompt {prompt_index} has {len(group)} images; expected {expected_size}",
            )
        sample_indices = [cell.generated.sample_index for cell in group]
        if sample_indices != list(range(expected_size)):
            raise ValueError(
                f"prompt {prompt_index} sample indices must be 0..{expected_size - 1}; "
                f"got {sample_indices}",
            )


def _score_cells(
    model: CodexImageQARewardModel,
    cells: Sequence[ExactCountCell],
) -> list[ScoredCell]:
    artifacts: list[RewardInferenceArtifact] = []
    for cell in cells:
        generated = cell.generated
        with Image.open(generated.image_path) as source:
            media = source.convert("RGB").copy()
        artifact_id = f"{cell.arm}-p{generated.prompt_index:04d}-s{generated.sample_index:02d}"
        artifacts.append(
            RewardInferenceArtifact(
                artifact_id=artifact_id,
                sample_id=artifact_id,
                path=str(generated.image_path),
                size_bytes=generated.image_path.stat().st_size,
                sha256=generated.image_sha256,
                prompt=generated.prompt,
                metadata={
                    **generated.reward_metadata,
                    REWARD_GROUP_ID_METADATA_KEY: (f"{cell.arm}:prompt:{generated.prompt_index}"),
                },
                media=media,
            ),
        )
    score_maps = model.score_batch(artifacts)
    if len(score_maps) != len(cells):
        raise RuntimeError(
            f"exact-count reward returned {len(score_maps)} rows for {len(cells)} images",
        )
    return [
        ScoredCell(cell=cell, scores={str(key): float(value) for key, value in scores.items()})
        for cell, scores in zip(cells, score_maps, strict=True)
    ]


def _score_countgd_cells(
    model: CountGDModel,
    cells: Sequence[ExactCountCell],
    *,
    progress_every: int,
) -> list[ScoredCell]:
    model.prepare_for_inference()
    scored: list[ScoredCell] = []
    total = len(cells)
    for index, cell in enumerate(cells, 1):
        generated = cell.generated
        artifact_id = f"{cell.arm}-p{generated.prompt_index:04d}-s{generated.sample_index:02d}"
        artifact = RewardInferenceArtifact(
            artifact_id=artifact_id,
            sample_id=artifact_id,
            path=str(generated.image_path),
            size_bytes=generated.image_path.stat().st_size,
            sha256=generated.image_sha256,
            prompt=generated.prompt,
            # This historical person-count archive also serves the Codex
            # evaluator; translate its target only at the generic model boundary.
            metadata={
                **generated.reward_metadata,
                "object_class": "person",
                "expected_count": cell.expected_people,
            },
        )
        result = model.evaluate(artifact)
        scored.append(
            ScoredCell(
                cell=cell,
                scores={
                    **result.to_scores(),
                    "countgd_observed_count": float(result.observed_count),
                },
            ),
        )
        if index % progress_every == 0 or index == total:
            print(f"Scored {index}/{total} images", flush=True)
    return scored


def _score_row(
    row: ScoredCell,
    *,
    reward_key: str,
    observed_count_key: str | None,
) -> dict[str, Any]:
    cell = row.cell
    generated = cell.generated
    record = {
        "arm": cell.arm,
        "prompt_index": generated.prompt_index,
        "sample_index": generated.sample_index,
        "seed": generated.seed,
        "prompt": generated.prompt,
        "expected_people": cell.expected_people,
        "image_path": str(generated.image_path),
        "image_sha256": generated.image_sha256,
        "reward": int(row.scores[reward_key]),
        "scores": row.scores,
    }
    if observed_count_key is not None:
        record["observed_count"] = int(row.scores[observed_count_key])
    return record


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


def _write_contact_sheets(
    base_cells: Sequence[ExactCountCell],
    checkpoint_cells: Sequence[ExactCountCell],
    *,
    checkpoint_label: str,
    blind_seed: int,
    output_dir: Path,
) -> None:
    grouped: dict[str, dict[int, list[ExactCountCell]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for cell in (*base_cells, *checkpoint_cells):
        grouped[cell.arm][cell.generated.prompt_index].append(cell)

    sheet_dir = output_dir / "contact_sheets" / "blind"
    sheet_dir.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for prompt_index in sorted(grouped["base"]):
        order = ["base", checkpoint_label]
        random.Random(blind_seed + prompt_index).shuffle(order)
        arm_images = {arm: _arm_contact_panel(grouped[arm][prompt_index]) for arm in order}
        panel_width, panel_height = arm_images[order[0]].size
        gap = 4
        header = 48
        canvas = Image.new(
            "RGB",
            (2 * panel_width + 3 * gap, header + panel_height + gap),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"Prompt {prompt_index:04d}", fill="black")
        mapping: dict[str, str] = {}
        for cell_index, arm in enumerate(order):
            blind_cell = chr(ord("A") + cell_index)
            x = gap + cell_index * (panel_width + gap)
            draw.text((x + panel_width // 2 - 4, 24), blind_cell, fill="black")
            canvas.paste(arm_images[arm], (x, header))
            mapping[blind_cell] = arm
        path = sheet_dir / f"prompt{prompt_index:04d}.png"
        canvas.save(path, format="PNG")
        manifest_rows.append(
            {
                "prompt_index": prompt_index,
                "prompt": grouped["base"][prompt_index][0].generated.prompt,
                "expected_people": grouped["base"][prompt_index][0].expected_people,
                "sheet": str(path.relative_to(output_dir)),
                "cells": ["A", "B"],
                "sample_indices": [
                    cell.generated.sample_index for cell in grouped["base"][prompt_index]
                ],
            },
        )
        key_rows.append(
            {
                "prompt_index": prompt_index,
                "cell_to_arm": mapping,
            },
        )
    _write_jsonl(output_dir / "contact_sheets" / "manifest.jsonl", manifest_rows)
    _write_json(
        output_dir / "blind_key.json",
        {
            "schema": REPORT_SCHEMA,
            "blind_seed": blind_seed,
            "note": "Review contact sheets before opening this key.",
            "arm_mappings": key_rows,
        },
    )


def _arm_contact_panel(cells: Sequence[ExactCountCell]) -> Image.Image:
    cells = sorted(cells, key=lambda cell: cell.generated.sample_index)
    tile = 256
    columns = 2
    rows = math.ceil(len(cells) / columns)
    panel = Image.new("RGB", (columns * tile, rows * tile), "white")
    draw = ImageDraw.Draw(panel)
    for index, cell in enumerate(cells):
        generated = cell.generated
        with Image.open(generated.image_path) as source:
            image = ImageOps.fit(source.convert("RGB"), (tile, tile))
        x = (index % columns) * tile
        y = (index // columns) * tile
        panel.paste(image, (x, y))
        draw.rectangle((x, y, x + 34, y + 20), fill="black")
        draw.text((x + 4, y + 3), f"#{generated.sample_index:02d}", fill="white")
    return panel


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
