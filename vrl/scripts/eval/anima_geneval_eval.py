"""Held-out GenEval eval for Anima: seed-aligned generation, OWLv2+CLIP scoring.

Scores a checkpoint (or the base model) with the SAME detector code that
produces the training reward (``GenEvalOwlRewardModel``), so a strict-accuracy
gain is a gain on the reward's own terms and any disagreement with human
review is visible per row. Every row is generated with ``seed + index`` on the
same manifest, so two arms pair exactly (SPRINT_anima_rl_target_search §11.5:
unaligned seeds produced larger swings than any checkpoint effect).

``--compare`` takes another arm's ``geneval_report.json`` and prints the
paired strict delta with a prompt-level bootstrap interval, overall and per
task. The report also carries the Laplacian sharpness of every image as the
cel-shading guard the sprint requires.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCORE_BATCH = 16
_TASK_ORDER = ("single_object", "two_object", "counting", "colors", "position", "color_attr")


@dataclass(frozen=True)
class EvalRow:
    index: int
    prompt: str
    tag: str
    spec: dict[str, Any]


@dataclass(frozen=True)
class RowScore:
    index: int
    tag: str
    strict: float
    partial: float
    dense: float
    why: str
    sharpness: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Held-out GenEval (OWLv2+CLIP) eval for Anima")
    parser.add_argument("--config", default="model/cosmos/anima_preview3")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable config override forwarded to the fixed-eval sampler.",
    )
    parser.add_argument("--manifest", default="datasets/geneval/official_eval_553_anime.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = every manifest row")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lora-path", default="", help="Checkpoint to evaluate; empty = base model"
    )
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--steps", type=int, default=0, help="0 = follow the config")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--label", default="", help="Name for this run in the report")
    parser.add_argument(
        "--images-dir",
        default="",
        help="Score existing eval_NNNN.png files (one per manifest row, in order) "
        "instead of generating.",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.15)
    parser.add_argument(
        "--compare",
        default="",
        help="Another arm's geneval_report.json (same manifest and seed) for paired deltas.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.manifest)
    rows = rows[: args.limit] if args.limit else rows
    if args.images_dir:
        image_paths = [
            Path(args.images_dir).expanduser() / f"eval_{row.index:04d}.png" for row in rows
        ]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing: {missing[:5]}")
    else:
        image_paths = _generate_images(args, rows, out_dir)

    scores = score_rows(rows, image_paths, detection_threshold=args.detection_threshold)
    report = summarize(scores)
    report.update(
        {
            "label": args.label or ("base" if not args.lora_path else Path(args.lora_path).name),
            "lora_path": args.lora_path,
            "manifest": args.manifest,
            "images_dir": args.images_dir or str(out_dir / "images"),
            "seed": args.seed,
            "row_count": len(rows),
            "detection_threshold": args.detection_threshold,
        }
    )
    (out_dir / "geneval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(format_summary(report))
    if args.compare:
        base = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print(format_paired(base, report))


def load_rows(manifest: str | Path) -> list[EvalRow]:
    from vrl.trainers.data.prompts import load_prompt_dataset_index

    rows: list[EvalRow] = []
    for index, example in enumerate(load_prompt_dataset_index(manifest)):
        spec = example.metadata.get("geneval")
        if not isinstance(spec, Mapping) or not spec.get("tag"):
            raise ValueError(f"{manifest}: row {index} lacks metadata.geneval with a tag")
        rows.append(
            EvalRow(index=index, prompt=example.prompt, tag=str(spec["tag"]), spec=dict(spec))
        )
    return rows


def score_rows(
    rows: Sequence[EvalRow], image_paths: Sequence[Path], *, detection_threshold: float
) -> list[RowScore]:
    from PIL import Image

    from vrl.rewards.models.geneval_owl import GenEvalOwlRewardModel
    from vrl.scripts.eval._device import resolve_eval_device
    from vrl.scripts.eval.anima_tag_adherence_eval import laplacian_sharpness

    device = resolve_eval_device("auto")
    model = GenEvalOwlRewardModel(
        {"device": str(device), "detection_threshold": detection_threshold}
    )
    scores: list[RowScore] = []
    for start in range(0, len(rows), _SCORE_BATCH):
        chunk = rows[start : start + _SCORE_BATCH]
        images = [
            Image.open(path).convert("RGB") for path in image_paths[start : start + _SCORE_BATCH]
        ]
        verdicts = model.judge_images(images, [row.spec for row in chunk])
        scores.extend(
            RowScore(
                index=row.index,
                tag=row.tag,
                strict=verdict.strict,
                partial=verdict.partial,
                dense=verdict.dense,
                why=verdict.why,
                sharpness=laplacian_sharpness(image),
            )
            for row, image, verdict in zip(chunk, images, verdicts, strict=True)
        )
        logger.info("scored %s/%s", len(scores), len(rows))
    return scores


def summarize(scores: Sequence[RowScore]) -> dict[str, Any]:
    """Pure aggregation: overall + per-task strict/partial means, sharpness, failure reasons."""

    by_tag: dict[str, list[RowScore]] = defaultdict(list)
    for score in scores:
        by_tag[score.tag].append(score)
    per_task = {
        tag: {
            "n": len(group),
            "strict": statistics.fmean(s.strict for s in group),
            "partial": statistics.fmean(s.partial for s in group),
            "dense": statistics.fmean(s.dense for s in group),
        }
        for tag, group in by_tag.items()
    }
    reasons: dict[str, int] = defaultdict(int)
    for score in scores:
        if score.why != "ok":
            for reason in score.why.split(";"):
                reasons[reason.split(":")[0]] += 1
    sharp = sorted(s.sharpness for s in scores)
    return {
        "strict_overall": statistics.fmean(s.strict for s in scores) if scores else 0.0,
        # GenEval's headline number: the mean of per-task accuracies.
        "strict_task_mean": statistics.fmean(v["strict"] for v in per_task.values())
        if per_task
        else 0.0,
        "partial_mean": statistics.fmean(s.partial for s in scores) if scores else 0.0,
        "dense_mean": statistics.fmean(s.dense for s in scores) if scores else 0.0,
        "per_task": per_task,
        "failure_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "sharpness_x1e-3": {
            "median": sharp[len(sharp) // 2] if sharp else 0.0,
            "p10": sharp[len(sharp) // 10] if sharp else 0.0,
        },
        "per_row": [asdict(score) for score in scores],
    }


def paired_delta(
    base: Mapping[str, Any], other: Mapping[str, Any], *, tag: str | None = None
) -> dict[str, Any]:
    """Row-paired strict delta (other - base) with a prompt-level bootstrap interval."""

    from vrl.scripts.eval.score_report import bootstrap_mean_interval

    base_rows = {row["index"]: row for row in base["per_row"]}
    deltas: list[float] = []
    for row in other["per_row"]:
        if tag is not None and row["tag"] != tag:
            continue
        if row["index"] not in base_rows or base_rows[row["index"]]["tag"] != row["tag"]:
            raise ValueError(f"row {row['index']} is not paired with the base report")
        deltas.append(float(row["strict"]) - float(base_rows[row["index"]]["strict"]))
    if not deltas:
        raise ValueError(f"no paired rows for tag {tag!r}")
    mean = statistics.fmean(deltas)
    std = statistics.pstdev(deltas)
    lower, upper = bootstrap_mean_interval(
        deltas, schema="geneval_report", label=str(other.get("label", "")), score_key=tag or "all"
    )
    return {
        "n": len(deltas),
        "delta": mean,
        "z": mean / (std / math.sqrt(len(deltas))) if std else float("nan"),
        "wins": sum(d > 0 for d in deltas),
        "ties": sum(d == 0 for d in deltas),
        "losses": sum(d < 0 for d in deltas),
        "bootstrap_95ci": [lower, upper],
    }


def format_summary(report: Mapping[str, Any]) -> str:
    lines = [
        f"label={report['label']} rows={report['row_count']} lora={report['lora_path'] or '-'}",
        f"{'task':<16}{'n':>5}{'strict':>9}{'partial':>9}{'dense':>9}",
    ]
    for tag in (*_TASK_ORDER, *sorted(set(report["per_task"]) - set(_TASK_ORDER))):
        if tag in report["per_task"]:
            stats = report["per_task"][tag]
            lines.append(
                f"{tag:<16}{stats['n']:>5}{stats['strict']:>9.3f}{stats['partial']:>9.3f}"
            )
    lines.append(
        f"{'task mean':<16}{report['row_count']:>5}{report['strict_task_mean']:>9.3f}"
        f"{report['partial_mean']:>9.3f}{report['dense_mean']:>9.3f}"
    )
    lines.append(
        f"sharpness x1e-3 median {report['sharpness_x1e-3']['median']:.2f} "
        f"p10 {report['sharpness_x1e-3']['p10']:.2f}; failures {report['failure_reasons']}"
    )
    return "\n".join(lines)


def format_paired(base: Mapping[str, Any], other: Mapping[str, Any]) -> str:
    lines = [
        f"paired vs {base['label']}: {'task':<16}{'delta':>8}{'z':>7}{'W/T/L':>12}{'CI95':>20}",
    ]
    tags = [tag for tag in _TASK_ORDER if tag in other["per_task"]]
    for tag in (*tags, None):
        stats = paired_delta(base, other, tag=tag)
        lines.append(
            f"{'':<{len('paired vs ') + len(base['label']) + 2}}{tag or 'ALL':<16}"
            f"{stats['delta']:>+8.3f}{stats['z']:>7.2f}"
            f"{f'{stats["wins"]}/{stats["ties"]}/{stats["losses"]}':>12}"
            f"{f'[{stats["bootstrap_95ci"][0]:+.3f}, {stats["bootstrap_95ci"][1]:+.3f}]':>20}"
        )
    return "\n".join(lines)


def _generate_images(
    args: argparse.Namespace, rows: Sequence[EvalRow], out_dir: Path
) -> list[Path]:
    """One image per row through the fixed-eval sampler (seed + index), no reward endpoint."""

    from vrl.scripts.eval.anima_fixed_eval import _generate

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    generated = _generate(argparse.Namespace(**{**vars(args), "limit": len(rows)}), out_dir)
    return [Path(row["image_path"]) for row in generated]


if __name__ == "__main__":
    main()
