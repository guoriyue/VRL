"""One-shot script: score a directory of generated images with Claude as the
anatomy judge. Reads <output_dir>/metadata.jsonl (rows with image_path + prompt),
calls Claude with a structured anatomy-judge prompt, writes scores.jsonl +
prints a markdown table.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/eval/claude_anatomy_judge.py \
        --dir outputs/anima_anatomy_eval16_parity10step \
        --model claude-sonnet-4-6 --concurrency 8
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


ANATOMY_JUDGE_PROMPT = (
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


async def score_one(
    client: anthropic.AsyncAnthropic,
    model: str,
    row: dict,
    sem: asyncio.Semaphore,
) -> dict:
    """Call Claude on a single (image, prompt) pair. Always returns a dict."""

    image_path = Path(row["image_path"])
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
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
                                "text": ANATOMY_JUDGE_PROMPT.format(prompt=row["prompt"]),
                            },
                        ],
                    },
                ],
            )
            text = resp.content[0].text.strip()
            # Tolerate fenced or trailing-prose responses by extracting the first JSON object.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(match.group(0) if match else text)
            score = int(data["score"])
            reasons = list(data.get("reasons") or [])
            return {
                "prompt_index": row.get("prompt_index"),
                "image_path": str(image_path),
                "prompt": row["prompt"],
                "score": score,
                "reasons": reasons,
                "raw": text,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "prompt_index": row.get("prompt_index"),
                "image_path": str(image_path),
                "prompt": row["prompt"],
                "score": None,
                "reasons": [],
                "raw": "",
                "error": f"{type(exc).__name__}: {exc}",
            }


async def run(args: argparse.Namespace) -> None:
    manifest = Path(args.dir) / "metadata.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Scoring {len(rows)} images from {manifest} with model={args.model} ...")

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[score_one(client, args.model, r, sem) for r in rows])
    results.sort(key=lambda r: r.get("prompt_index") or 0)

    out_path = Path(args.dir) / "claude_anatomy_scores.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {out_path}\n")

    valid = [r for r in results if r["score"] is not None]
    failed = [r for r in results if r["score"] is None]
    if valid:
        avg = sum(r["score"] for r in valid) / len(valid)
        print(f"Mean score: {avg:.2f}  (n={len(valid)}, errors={len(failed)})\n")

    print("| idx | score | prompt (≤60ch) | reasons |")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in env — export it first.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
