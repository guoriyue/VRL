"""Mine detector-reliable Anima anatomy prompts for GRPO training."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from vrl.config.loading import load_config
from vrl.models.diffusion.cosmos.anima.runtime import (
    build_anima_runtime_bundle,
    extract_anima_runtime_spec,
)
from vrl.rewards.functions.anime_anatomy import AnimeAnatomyStructureReward
from vrl.rewards.types import RewardRollout, RewardTrajectory
from vrl.scripts.diffusion.cosmos.anima.generate import (
    DEFAULT_NEGATIVE_PROMPT,
    _configure_lora_for_inference,
    _generate_images,
    _resolve_device,
    _resolve_dtype,
    _resolve_sampling,
)
from vrl.trainers.data import PromptExample, load_prompt_manifest

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate base Anima samples and keep prompts with useful anatomy reward variance.",
    )
    parser.add_argument("--config", default="experiment/diffusion/anima_preview3/online_grpo_anatomy")
    parser.add_argument("--manifest", default="", help="Prompt manifest; defaults to cfg.data.manifest.")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--candidate-limit", type=int, default=128)
    parser.add_argument("--min-accepted", type=int, default=64)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--min-reward-std", type=float, default=0.05)
    parser.add_argument("--min-nonzero", type=int, default=2)
    parser.add_argument("--min-detected", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides passed to config loading.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.candidate_limit < 1:
        raise ValueError("--candidate-limit must be >= 1")
    if args.min_accepted < 1:
        raise ValueError("--min-accepted must be >= 1")
    if args.samples_per_prompt < 2:
        raise ValueError("--samples-per-prompt must be >= 2")

    cfg = load_config(args.config, overrides=args.overrides)
    manifest_path = args.manifest or str(cfg.data.manifest)
    examples = load_prompt_manifest(manifest_path)
    if not examples:
        raise ValueError(f"prompt manifest is empty: {manifest_path}")

    selected = _bucket_balanced_indices(examples, seed=args.seed, limit=args.candidate_limit)
    _configure_lora_for_inference(cfg, lora_path="", use_config_lora=False)

    import torch

    device = _resolve_device(args.device, torch)
    dtype = _resolve_dtype(args.dtype, cfg, device=device, torch=torch)
    sampling = _resolve_sampling(args, cfg)
    logger.info(
        "Mining %d candidates from %s on device=%s dtype=%s samples_per_prompt=%d",
        len(selected),
        manifest_path,
        device,
        dtype,
        args.samples_per_prompt,
    )

    bundle = build_anima_runtime_bundle(extract_anima_runtime_spec(cfg, device, dtype))
    model = bundle.model.eval()
    reward = AnimeAnatomyStructureReward(device=str(device))

    accepted_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for ordinal, example_index in enumerate(selected):
            example = examples[example_index]
            prompt_seed = int(args.seed) + ordinal * int(args.samples_per_prompt)
            logger.info(
                "Mining candidate %d/%d accepted=%d bucket=%s seed=%d",
                ordinal + 1,
                len(selected),
                len(accepted_rows),
                _bucket(example),
                prompt_seed,
            )
            images = _generate_images(
                model,
                prompt=example.prompt,
                negative_prompt=args.negative_prompt,
                seed=prompt_seed,
                samples_per_prompt=args.samples_per_prompt,
                sampling=sampling,
                device=device,
                dtype=dtype,
                torch=torch,
            )
            scores, diagnostics = asyncio.run(
                _score_images(reward, example, prompt_seed, images),
            )
            record = _record_for_candidate(
                example=example,
                example_index=example_index,
                prompt_seed=prompt_seed,
                scores=scores,
                diagnostics=diagnostics,
                thresholds={
                    "min_reward_std": float(args.min_reward_std),
                    "min_nonzero": int(args.min_nonzero),
                    "min_detected": int(args.min_detected),
                },
            )
            records.append(record)
            if record["accepted"]:
                accepted_rows.append(_manifest_row(example, record))
                logger.info(
                    "Accepted prompt mean=%.4f std=%.4f detected=%d nonzero=%d",
                    record["reward_mean"],
                    record["reward_std"],
                    record["detected_count"],
                    record["nonzero_count"],
                )
            if len(accepted_rows) >= args.min_accepted:
                break

    _write_outputs(
        output_manifest=Path(args.output_manifest),
        report_json=Path(args.report_json),
        manifest_rows=accepted_rows,
        records=records,
        source_manifest=manifest_path,
        args=args,
    )


async def _score_images(
    reward: AnimeAnatomyStructureReward,
    example: PromptExample,
    prompt_seed: int,
    images: list[Image.Image],
) -> tuple[list[float], list[dict[str, Any]]]:
    rollouts = [
        RewardRollout(
            request=SimpleNamespace(),
            trajectory=RewardTrajectory(
                prompt=example.prompt,
                seed=prompt_seed + sample_index,
                steps=[],
                output=image,
            ),
            metadata={"manifest_row": _manifest_source_row(example)},
        )
        for sample_index, image in enumerate(images)
    ]
    scores = await reward.score_batch(rollouts)
    return [float(score) for score in scores], list(reward.last_diagnostics)


def _record_for_candidate(
    *,
    example: PromptExample,
    example_index: int,
    prompt_seed: int,
    scores: list[float],
    diagnostics: list[dict[str, Any]],
    thresholds: dict[str, float | int],
) -> dict[str, Any]:
    score_arr = np.asarray(scores, dtype=np.float64)
    people_counts = [int(diagnostic.get("num_people", 0)) for diagnostic in diagnostics]
    nonzero_count = int(np.count_nonzero(score_arr > 0.0))
    detected_count = sum(1 for count in people_counts if count > 0)
    reward_std = float(score_arr.std())
    accepted = (
        reward_std >= float(thresholds["min_reward_std"])
        and nonzero_count >= int(thresholds["min_nonzero"])
        and detected_count >= int(thresholds["min_detected"])
    )
    return {
        "accepted": bool(accepted),
        "example_index": int(example_index),
        "prompt_seed": int(prompt_seed),
        "prompt": example.prompt,
        "bucket": _bucket(example),
        "constraints": list(_constraints(example)),
        "scores": [float(score) for score in scores],
        "people_counts": people_counts,
        "reward_mean": float(score_arr.mean()),
        "reward_std": reward_std,
        "reward_min": float(score_arr.min()),
        "reward_max": float(score_arr.max()),
        "nonzero_count": nonzero_count,
        "detected_count": int(detected_count),
        "thresholds": dict(thresholds),
    }


def _manifest_row(example: PromptExample, record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(example.metadata or {})
    metadata["mining"] = {
        "source": "base_anima_anatomy_reward_variance",
        "prompt_seed": record["prompt_seed"],
        "scores": record["scores"],
        "reward_mean": record["reward_mean"],
        "reward_std": record["reward_std"],
        "detected_count": record["detected_count"],
        "nonzero_count": record["nonzero_count"],
    }
    return {"prompt": example.prompt, "metadata": metadata}


def _write_outputs(
    *,
    output_manifest: Path,
    report_json: Path,
    manifest_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    source_manifest: str,
    args: argparse.Namespace,
) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "accepted_count": len(manifest_rows),
        "candidate_count": len(records),
        "source_manifest": source_manifest,
        "output_manifest": str(output_manifest),
        "settings": {
            "candidate_limit": int(args.candidate_limit),
            "min_accepted": int(args.min_accepted),
            "samples_per_prompt": int(args.samples_per_prompt),
            "seed": int(args.seed),
            "steps": int(args.steps),
            "cfg_scale": float(args.cfg_scale),
            "min_reward_std": float(args.min_reward_std),
            "min_nonzero": int(args.min_nonzero),
            "min_detected": int(args.min_detected),
        },
        "records": records,
    }
    report_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %d accepted prompts to %s; report=%s",
        len(manifest_rows),
        output_manifest,
        report_json,
    )


def _bucket_balanced_indices(
    examples: list[PromptExample],
    *,
    seed: int,
    limit: int,
) -> list[int]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        by_bucket[_bucket(example)].append(idx)
    for values in by_bucket.values():
        rng.shuffle(values)
    bucket_names = sorted(by_bucket)
    rng.shuffle(bucket_names)
    out: list[int] = []
    while len(out) < limit and bucket_names:
        next_bucket_names = []
        for name in bucket_names:
            values = by_bucket[name]
            if values and len(out) < limit:
                out.append(values.pop())
            if values:
                next_bucket_names.append(name)
        bucket_names = next_bucket_names
    return out


def _manifest_source_row(example: PromptExample) -> dict[str, Any]:
    return {"prompt": example.prompt, "metadata": dict(example.metadata or {})}


def _bucket(example: PromptExample) -> str:
    return str((example.metadata or {}).get("bucket") or "unknown")


def _constraints(example: PromptExample) -> tuple[str, ...]:
    metadata = example.metadata or {}
    raw = metadata.get("constraints") or []
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw if str(item).strip())


if __name__ == "__main__":
    main()
