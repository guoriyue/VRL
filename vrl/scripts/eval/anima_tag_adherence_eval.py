"""Held-out tag-adherence eval for the adult-only NSFW prompt set (objective C).

Scores a checkpoint (or the base model) with the SAME tagger code that produces
the training reward (``WDTaggerRewardModel``), so a recall gain here is real
adherence rather than judge noise — every judge reward tried on this policy tied
on ~60% of prompts and moved between runs; the tagger gives the same score to
the same image every time.

Three guards ride alongside recall because recall alone is gameable:

* compliance (Falconsai P(nsfw) > threshold) — the policy could raise recall on
  clothing/pose tags by rendering the prompt as SFW; the baseline decision rule
  demands compliance stays >= 95% on explicit rows;
* sharpness (Laplacian variance) — cel-shading softening is invisible to every
  learned quality model, and the whitelist tagger tolerates soft images too;
* per-tag drop counts — a mean can rise while the hard tags (``lingerie``,
  ``bent_over``) stay dropped, which is not the gain the objective is after.

Numbers reproduce ``outputs/nsfw_compliance_base200/baseline_c_adherence.json``:
recall/perfect were computed there over the 170 filtered rows exactly as here.
The baseline's compliance split (97/100 explicit) and sharpness stats
(mean 19.07, n=200) were computed over all 200 unfiltered base rows; this
script reports them over the 170 filtered rows, so those two differ slightly
from the baseline file by construction — the report says so.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_BASELINE = "outputs/nsfw_compliance_base200/baseline_c_adherence.json"
_TAGGER_BATCH = 16


@dataclass(frozen=True)
class EvalRow:
    """One manifest row with what the scorers need, tags normalized like the reward."""

    index: int
    prompt: str
    tier: str
    tags: tuple[str, ...]
    source_row: int | None


@dataclass(frozen=True)
class RowScore:
    source_row: int | None
    tier: str
    recall: float
    missed_tags: list[str]
    p_nsfw: float
    sharpness: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Held-out tag adherence / compliance / sharpness eval for Anima"
    )
    parser.add_argument("--config", default="model/cosmos/anima_preview3")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable config override forwarded to the fixed-eval sampler.",
    )
    parser.add_argument("--manifest", default="datasets/danbooru/safety/eval_c_adherence.jsonl")
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
        help="Score existing PNGs instead of generating. A directory holding exactly one "
        "image per manifest row is read in order (eval_0000.png = row 0); any other "
        "size is treated as the unfiltered base set and rows map through "
        "metadata.source_row (eval_{source_row:04d}.png).",
    )
    parser.add_argument(
        "--tag-threshold",
        type=float,
        default=0.35,
        help="WD14 general-tag threshold (inclusive); must match reward/wd_tagger",
    )
    parser.add_argument(
        "--nsfw-threshold",
        type=float,
        default=0.35,
        help="P(nsfw) above which an image counts as compliant; matches reward/nsfw_safety",
    )
    parser.add_argument("--baseline", default=_DEFAULT_BASELINE)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_rows(args.manifest)
    rows = all_rows[: args.limit] if args.limit else all_rows
    if args.images_dir:
        image_paths = resolve_image_paths(
            Path(args.images_dir).expanduser(), rows, manifest_row_count=len(all_rows)
        )
    else:
        image_paths = _generate_images(args, rows, out_dir)

    report = score_rows(
        rows,
        image_paths,
        tag_threshold=args.tag_threshold,
        nsfw_threshold=args.nsfw_threshold,
    )
    report.update(
        {
            "label": args.label or ("base" if not args.lora_path else Path(args.lora_path).name),
            "lora_path": args.lora_path,
            "manifest": args.manifest,
            "images_dir": args.images_dir or str(out_dir / "images"),
            "seed": args.seed,
            "row_count": len(rows),
            "note": (
                "compliance and sharpness are over the filtered manifest rows; the baseline "
                "file computed those two over all 200 unfiltered base rows"
            ),
        }
    )
    (out_dir / "tag_adherence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    baseline_path = Path(args.baseline)
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    )
    print(format_summary(report, baseline))


def load_rows(manifest: str | Path) -> list[EvalRow]:
    from vrl.trainers.data.prompts import load_prompt_dataset_index

    rows: list[EvalRow] = []
    for index, example in enumerate(load_prompt_dataset_index(manifest)):
        raw_tags = example.metadata.get("adherence_tags")
        if not isinstance(raw_tags, (list, tuple)) or not raw_tags:
            raise ValueError(f"{manifest}: row {index} lacks a non-empty metadata.adherence_tags")
        source_row = example.metadata.get("source_row")
        rows.append(
            EvalRow(
                index=index,
                prompt=example.prompt,
                tier=parse_tier(example.prompt),
                # Same normalization as WDTaggerRewardModel._wanted_tags.
                tags=tuple(sorted({str(t).strip().lower() for t in raw_tags if str(t).strip()})),
                source_row=None if source_row is None else int(source_row),
            )
        )
    return rows


def parse_tier(prompt: str) -> str:
    """``rating:explicit`` / ``rating:questionable`` token -> tier name."""

    for token in prompt.split(","):
        token = token.strip().lower()
        if token.startswith("rating:"):
            return token.split(":", 1)[1].strip()
    raise ValueError(f"prompt carries no rating:<tier> token: {prompt!r}")


def resolve_image_paths(
    images_dir: Path, rows: Sequence[EvalRow], *, manifest_row_count: int
) -> list[Path]:
    """Map manifest rows to PNGs under ``images_dir``.

    Two layouts exist: this script's own output (one image per manifest row,
    in order) and the unfiltered 200-image base set the manifest was filtered
    from (rows reach it through ``metadata.source_row``). The directory size
    decides: exactly ``manifest_row_count`` images means in-order, anything
    else means source_row. Size is the discriminator rather than file
    existence because ``--limit`` makes low source_row files exist in both
    layouts.
    """

    png_count = len(list(images_dir.glob("eval_*.png")))
    if png_count == manifest_row_count:
        paths = [images_dir / f"eval_{row.index:04d}.png" for row in rows]
    else:
        if any(row.source_row is None for row in rows):
            raise ValueError(
                f"{images_dir} holds {png_count} images for a {manifest_row_count}-row manifest "
                "and rows lack metadata.source_row; cannot map rows to images"
            )
        paths = [images_dir / f"eval_{row.source_row:04d}.png" for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing under {images_dir}: {missing[:5]}")
    return paths


def score_rows(
    rows: Sequence[EvalRow],
    image_paths: Sequence[Path],
    *,
    tag_threshold: float,
    nsfw_threshold: float,
) -> dict[str, Any]:
    from PIL import Image

    from vrl.rewards.models.nsfw_safety import NSFWSafetyRewardModel
    from vrl.rewards.models.wd_tagger import WDTaggerRewardModel

    images = [Image.open(path).convert("RGB") for path in image_paths]
    tagger = WDTaggerRewardModel({"threshold": tag_threshold})
    tag_probs: list[dict[str, float]] = []
    for start in range(0, len(images), _TAGGER_BATCH):
        tag_probs.extend(tagger.tag_images(images[start : start + _TAGGER_BATCH]))
        logger.info("tagged %s/%s", min(start + _TAGGER_BATCH, len(images)), len(images))
    # ``_score_images`` is the probability the penalty is derived from; the
    # public penalty saturates to 0 below threshold, which would hide P(nsfw).
    nsfw = NSFWSafetyRewardModel({"classifier_device": "cpu", "threshold": nsfw_threshold})
    p_nsfw = nsfw._score_images(images)
    sharpness = [laplacian_sharpness(image) for image in images]
    return summarize(
        rows,
        tag_probs,
        p_nsfw,
        sharpness,
        tag_threshold=tag_threshold,
        nsfw_threshold=nsfw_threshold,
    )


def summarize(
    rows: Sequence[EvalRow],
    tag_probs: Sequence[Mapping[str, float]],
    p_nsfw: Sequence[float],
    sharpness: Sequence[float],
    *,
    tag_threshold: float,
    nsfw_threshold: float,
) -> dict[str, Any]:
    """Pure aggregation of per-image scores into the report body."""

    import numpy as np

    asked: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    per_row: list[RowScore] = []
    for row, probs, prob, sharp in zip(rows, tag_probs, p_nsfw, sharpness, strict=True):
        # Inclusive threshold, like WDTaggerRewardModel._recall.
        detected = {tag.lower() for tag, value in probs.items() if value >= tag_threshold}
        dropped = [tag for tag in row.tags if tag not in detected]
        asked.update(row.tags)
        missed.update(dropped)
        per_row.append(
            RowScore(
                source_row=row.source_row,
                tier=row.tier,
                recall=1.0 - len(dropped) / len(row.tags),
                missed_tags=dropped,
                p_nsfw=float(prob),
                sharpness=float(sharp),
            )
        )

    recalls = [score.recall for score in per_row]
    tiers = sorted({score.tier for score in per_row})
    compliance: dict[str, dict[str, float | int]] = {}
    for tier in (*tiers, "overall"):
        group = [s for s in per_row if tier == "overall" or s.tier == tier]
        triggered = sum(1 for s in group if s.p_nsfw > nsfw_threshold)
        compliance[tier] = {
            "triggered": triggered,
            "count": len(group),
            "rate": triggered / len(group) if group else 0.0,
        }
    # Only tags asked often enough for a drop count to mean something.
    drops = {
        tag: {"missed": missed[tag], "asked": asked[tag]}
        for tag, _ in sorted(missed.items(), key=lambda kv: (-kv[1], kv[0]))
        if asked[tag] >= 5
    }
    sharp = np.asarray(sharpness, dtype=np.float64)
    return {
        "tag_threshold": tag_threshold,
        "nsfw_threshold": nsfw_threshold,
        "tag_recall": {
            "mean": float(np.mean(recalls)) if recalls else 0.0,
            "perfect_frac": (
                sum(1 for r in recalls if r >= 0.999) / len(recalls) if recalls else 0.0
            ),
            "dropped": drops,
        },
        "compliance": compliance,
        "sharpness_x1e-3": {
            "mean": float(sharp.mean()) if sharp.size else 0.0,
            "median": float(np.median(sharp)) if sharp.size else 0.0,
            "p10": float(np.percentile(sharp, 10)) if sharp.size else 0.0,
            "n": int(sharp.size),
        },
        "per_row": [asdict(score) for score in per_row],
    }


def laplacian_sharpness(image: Image.Image) -> float:
    """Variance of the 3x3 Laplacian over [0, 1] luma, in 1e-3 units.

    Deliberately the baseline's exact formula (PIL ``L`` conversion, valid-only
    window) rather than ``ImageSharpnessRewardModel``'s hand-rolled luma, so
    the number is comparable with the recorded 19.07 / 17.10 / 4.97.
    """

    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    kernel = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = (sliding_window_view(gray, (3, 3)) * kernel).sum((-1, -2))
    return float(lap.var()) * 1e3


def format_summary(report: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> str:
    recall = report["tag_recall"]
    sharp = report["sharpness_x1e-3"]
    compliance = report["compliance"]
    lines = [
        f"label={report['label']} rows={report['row_count']} lora={report['lora_path'] or '-'}",
        f"{'metric':<28}{'this run':>14}{'baseline':>16}",
    ]

    def base(*keys: str) -> str:
        node: Any = baseline
        for key in keys:
            if not isinstance(node, Mapping) or key not in node:
                return "-"
            node = node[key]
        return str(node)

    lines.append(
        f"{'tag recall mean':<28}{recall['mean']:>14.3f}{base('tag_recall_whitelist', 'mean'):>16}"
    )
    lines.append(
        f"{'tag recall perfect frac':<28}{recall['perfect_frac']:>14.3f}"
        f"{base('tag_recall_whitelist', 'perfect_frac'):>16}"
    )
    for tier, stats in compliance.items():
        value = f"{stats['triggered']}/{stats['count']} ({stats['rate']:.1%})"
        lines.append(
            # Left-joined: the baseline's questionable entry carries a long note.
            f"{'compliance ' + tier:<28}{value:>14}  {base('compliance_falconsai_thr0.35', tier)}"
        )
    for key in ("mean", "median", "p10"):
        lines.append(
            f"{'sharpness x1e-3 ' + key:<28}{sharp[key]:>14.2f}"
            f"{base('sharpness_x1e-3_laplacian_var', key):>16}"
        )
    lines.append("dropped tags (missed/asked, asked >= 5):")
    for tag, counts in recall["dropped"].items():
        lines.append(
            f"  {tag:<26}{counts['missed']:>3}/{counts['asked']:<4}"
            f"{base('tag_recall_whitelist', 'top_dropped', tag):>12}"
        )
    lines.append(report["note"])
    return "\n".join(lines)


def _generate_images(
    args: argparse.Namespace, rows: Sequence[EvalRow], out_dir: Path
) -> list[Path]:
    """Generate one image per row through the fixed-eval sampler, no reward endpoint."""

    from vrl.scripts.eval.anima_fixed_eval import _generate

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    # The fixed eval slices ``[:limit]``; 0 there means no prompts, so pass the
    # resolved row count instead of forwarding this script's 0-means-all knob.
    generated = _generate(argparse.Namespace(**{**vars(args), "limit": len(rows)}), out_dir)
    return [Path(row["image_path"]) for row in generated]


if __name__ == "__main__":
    main()
