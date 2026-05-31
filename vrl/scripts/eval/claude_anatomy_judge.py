"""One-shot script: score a directory of generated images with Claude as the
anatomy judge. Reads <output_dir>/metadata.jsonl (rows with image_path + prompt),
calls Claude with a rubric-driven anatomy-judge prompt, writes scores.jsonl +
prints a markdown table.

The scoring criteria live in an external markdown rubric file
(default: configs/reward/rubrics/anatomy_v1.md). To change the reward's
behavior, edit the rubric — no code changes needed. Pass a different
rubric file via --rubric-file to A/B different scoring criteria.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m vrl.scripts.eval.claude_anatomy_judge \
        --dir outputs/anima_anatomy_eval16_896x1152_30step \
        --model claude-sonnet-4-6 --concurrency 8

    # Use a different rubric:
    python -m vrl.scripts.eval.claude_anatomy_judge \
        --dir outputs/... \
        --rubric-file configs/reward/rubrics/anatomy_v2.md
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUBRIC_PATH = REPO_ROOT / "configs" / "reward" / "rubrics" / "anatomy_v1.md"

# Fallback prompt used only when --rubric-file is missing or empty.
# Kept identical to the v0 hardcoded prompt for backward compatibility.
FALLBACK_JUDGE_PROMPT = (
    "Evaluate the anatomy of the character(s) in this anime image.\n"
    "\n"
    "The text prompt that produced this image is:\n"
    '"""{prompt}"""\n'
    "\n"
    "Score anatomical plausibility from 0 (severe anatomy failure) to 10 (clean,"
    " anatomically correct). Important: account for legitimate occlusion (hand"
    " behind body / props), gestures (fist, point, peace sign — not all 5 fingers"
    " need to be visible), viewing angles (side view, back view), and intentional"
    " stylization (chibi / deformed character / non-realistic proportions are OK"
    " if internally consistent). Only penalize FAILURES: extra/missing fingers"
    " inconsistent with the gesture, joints bending in impossible directions,"
    " limbs attached at wrong locations, mismatched left/right anatomy where it"
    " should match, fused limbs/digits.\n"
    "\n"
    "Output ONLY this JSON (no prose, no markdown fences):\n"
    '{{"score": <int 0..10>, "reasons": ["<short string>", ...]}}'
)


def build_prompt(rubric_text: str | None, user_prompt: str) -> str:
    """Compose the final judge prompt from rubric + the image's text prompt.

    When a rubric file is provided, the rubric becomes the system-of-record
    for scoring criteria and is injected verbatim. When absent, fall back to
    the legacy hardcoded prompt so existing callers keep working.
    """

    if not rubric_text or not rubric_text.strip():
        return FALLBACK_JUDGE_PROMPT.format(prompt=user_prompt)
    return (
        f"{rubric_text.rstrip()}\n\n"
        "---\n\n"
        "The text prompt that produced this image is:\n"
        f'"""{user_prompt}"""\n\n'
        "Apply the rubric above to score this image. Output ONLY the JSON "
        "object specified in the rubric — no prose, no markdown fences."
    )


def load_rubric(rubric_file: str | os.PathLike | None) -> str | None:
    """Read a rubric markdown file. Returns None when the path is empty
    or the file is missing/empty — caller falls back to the legacy prompt."""

    if not rubric_file:
        return None
    path = Path(rubric_file).expanduser()
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


async def score_one(
    client: anthropic.AsyncAnthropic,
    model: str,
    row: dict,
    sem: asyncio.Semaphore,
    rubric_text: str | None,
) -> dict:
    """Call Claude on a single (image, prompt) pair. Always returns a dict."""

    image_path = Path(row["image_path"])
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt_text = build_prompt(rubric_text, row["prompt"])
    async with sem:
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=400,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt_text,
                            },
                        ],
                    },
                ],
            )
            text = resp.content[0].text.strip()
            # Tolerate fenced or trailing-prose responses by extracting the first JSON object.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(match.group(0) if match else text)
            axes, score, reasons = _parse_score_payload(data)
            return {
                "prompt_index": row.get("prompt_index"),
                "image_path": str(image_path),
                "prompt": row["prompt"],
                "axes": axes,
                "score": score,
                "reasons": reasons,
                "raw": text,
                "error": None,
            }
        except Exception as exc:
            return {
                "prompt_index": row.get("prompt_index"),
                "image_path": str(image_path),
                "prompt": row["prompt"],
                "axes": None,
                "score": None,
                "reasons": [],
                "raw": "",
                "error": f"{type(exc).__name__}: {exc}",
            }


def _parse_score_payload(
    data: dict,
) -> tuple[dict | None, int, list[str]]:
    """Parse the judge's JSON output into (axes, aggregate_score, reasons).

    Supports two schemas:
      - **multi-axis** (preferred, rubric v1+):
            {"axes": {...}, "defects": [...]}
        Aggregate is computed by geometric mean over the non-null axes,
        rounded to the nearest int and clamped to [0, 10].
      - **single-score** (legacy / fallback prompt):
            {"score": int, "reasons": [...]}
        Pass through unchanged; axes returned as None.

    Raises if neither shape resolves to a usable score.
    """

    if "axes" in data and isinstance(data["axes"], dict):
        raw_axes = data["axes"]
        axes: dict[str, int | None] = {}
        for name, value in raw_axes.items():
            if value is None:
                axes[name] = None
            else:
                axes[name] = max(0, min(10, int(value)))
        score = _geometric_mean_score(axes)
        defects = list(data.get("defects") or [])
        return axes, score, defects

    if "score" in data:
        score = max(0, min(10, int(data["score"])))
        reasons = list(data.get("reasons") or [])
        return None, score, reasons

    raise ValueError(f"unrecognised score payload: {data!r}")


def _geometric_mean_score(axes: dict[str, int | None]) -> int:
    """Aggregate per-axis scores into a single integer in [0, 10].

    Uses the average of the geometric mean and the minimum axis:

        aggregate = round((geometric_mean(axes) + min(axes)) / 2)

    Properties:

      * **Worst axis pulls hard**: a hand at 3 with everything else at 9
        yields aggregate ≈ 5, not 7 (which is what plain geometric mean
        gives). RL receives clear signal that the hand is the failure.
      * **Gradient on every axis**: the geometric-mean term keeps every
        axis contributing — improving an already-good axis still moves
        the aggregate, just by less than improving the worst axis.
      * **Zero on any axis collapses to ≤ 0**: catastrophic local defect
        cannot be averaged away.

    The other natural choices and why we reject them:

      * Plain mean — too forgiving; clean torso silently masks broken hand.
      * Plain geometric mean — better than mean but still too soft; gives
        0000 (T=5 A=8 L=9 H=4 F=9) a 7, when human sense says ≈ 5.
      * Plain min — zero gradient on every axis except the worst, so
        RL stalls once one axis is dragging.
    """

    import math

    values = [v for v in axes.values() if v is not None]
    if not values:
        return 0
    if any(v <= 0 for v in values):
        return 0
    log_mean = sum(math.log(v) for v in values) / len(values)
    geom = math.exp(log_mean)
    worst = min(values)
    aggregate = (geom + worst) / 2.0
    return max(0, min(10, round(aggregate)))


async def run(args: argparse.Namespace) -> None:
    manifest = Path(args.dir) / "metadata.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    rubric_text = load_rubric(args.rubric_file)
    rubric_source = (
        str(Path(args.rubric_file).expanduser()) if rubric_text else "fallback (hardcoded)"
    )
    print(
        f"Scoring {len(rows)} images from {manifest} with model={args.model} "
        f"rubric={rubric_source} ...",
    )

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *[score_one(client, args.model, r, sem, rubric_text) for r in rows],
    )
    results.sort(key=lambda r: r.get("prompt_index") or 0)

    out_path = Path(args.dir) / "claude_anatomy_scores.jsonl"
    with out_path.open("w") as f:
        # First line: metadata header so callers can audit which rubric/model produced scores.
        f.write(
            json.dumps(
                {"_meta": True, "rubric": rubric_source, "model": args.model},
                sort_keys=True,
            )
            + "\n",
        )
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {out_path}\n")

    valid = [r for r in results if r["score"] is not None]
    failed = [r for r in results if r["score"] is None]
    if valid:
        avg = sum(r["score"] for r in valid) / len(valid)
        print(f"Mean score: {avg:.2f}  (n={len(valid)}, errors={len(failed)})\n")

    has_axes = any(r.get("axes") for r in results)
    if has_axes:
        print("| idx | score | torso | arms | legs | hands | feet | defects |")
        print("|---|---|---|---|---|---|---|---|")
        for r in results:
            idx = r.get("prompt_index", "?")
            score = r["score"] if r["score"] is not None else "ERR"
            axes = r.get("axes") or {}

            def cell(key: str, axes: dict[str, object] = axes) -> str:
                v = axes.get(key)
                return "-" if v is None else str(v)

            defects = "; ".join(r["reasons"][:2]).replace("|", "/") or (r["error"] or "")
            if len(defects) > 80:
                defects = defects[:80] + "..."
            print(
                f"| {idx} | {score} | {cell('torso')} | {cell('arms')} | "
                f"{cell('legs')} | {cell('hands')} | {cell('feet')} | {defects} |",
            )
    else:
        print("| idx | score | prompt (<=60ch) | reasons |")
        print("|---|---|---|---|")
        for r in results:
            idx = r.get("prompt_index", "?")
            score = r["score"] if r["score"] is not None else "ERR"
            prompt_short = (r["prompt"][:60] + "…") if len(r["prompt"]) > 60 else r["prompt"]
            prompt_short = prompt_short.replace("|", "/")
            reasons = "; ".join(r["reasons"][:3]).replace("|", "/") or (r["error"] or "")
            if len(reasons) > 90:
                reasons = reasons[:90] + "…"
            print(f"| {idx} | {score} | {prompt_short} | {reasons} |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing metadata.jsonl (image_path + prompt per row).",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model id.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent API requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score only the first N rows (0 = all).",
    )
    parser.add_argument(
        "--rubric-file",
        default=str(DEFAULT_RUBRIC_PATH),
        help=(
            "Path to a markdown rubric file that defines the scoring criteria. "
            "Pass an empty string to use the legacy hardcoded prompt. "
            f"Default: {DEFAULT_RUBRIC_PATH}"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in env — export it first.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
