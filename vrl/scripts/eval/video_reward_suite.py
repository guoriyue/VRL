"""Fixed video-reward eval suite: Kling components + VBench metrics to one CSV.

Reads a directory of ``.mp4`` files plus a prompt manifest and emits
``eval_video_metrics.csv`` with one row per video. Two metric sources:

* **Kling VideoReward** (always available — VRL owns the model): per-video
  ``visual_quality`` / ``motion_quality`` / ``text_alignment`` / ``overall_reward``.
  ``motion_quality`` is the headline motion signal the sprint asks for.
* **VBench** (optional, lazily imported): the decomposable subject/temporal/motion
  metrics. VBench ships its own models and only runs the four sprint dimensions
  when its package + weights are installed; if it is missing or errors, the
  ``vbench_*`` columns stay present but empty and a warning is logged. The exact
  per-video JSON schema is confirmed by the Phase A availability gate.

The repo-owned VideoScore2 judge plugs in behind ``--videoscore2``, emitting
``vs2_<key>`` columns. Other external benchmarks (VMBench, VBench-2.0,
DynamicEval) are run via their own CLIs and folded in with ``--merge-json
prefix=path`` — a per-video ``{filename: {metric: score}}`` JSON — so the suite
never guesses an unverified Python API.

This is a fixed *eval* surface, not an online reward — VBench's multi-model
latency/VRAM is unbounded, so the sprint keeps it out of the training loop until
a dimension proves it can separate good from bad samples.

Usage::

    python -m vrl.scripts.eval.video_reward_suite \
        --video-dir outputs/dance_eval/videos \
        --manifest data/prompts/dance_spin.jsonl \
        --output outputs/dance_eval/eval_video_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.trainers.data.prompts import load_prompt_manifest

logger = logging.getLogger(__name__)

# Kling public score keys recorded per video (see rewards/models/kling_video_reward.py).
_KLING_SCORE_KEYS = ("visual_quality", "motion_quality", "text_alignment", "overall_reward")
# Sprint dimensions. NOTE: VBench custom-input mode supports subject_consistency,
# background_consistency, motion_smoothness, dynamic_degree, aesthetic_quality,
# imaging_quality. temporal_flickering may require non-custom mode — the Phase A
# gate confirms which dims actually run; unsupported dims simply stay empty.
_DEFAULT_VBENCH_DIMS = (
    "subject_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True, help="Directory of .mp4 files to score.")
    parser.add_argument(
        "--manifest",
        default="",
        help="Prompt manifest (JSONL). Paired to videos by reference_video filename, else by order.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Defaults to <video-dir>/eval_video_metrics.csv.",
    )
    parser.add_argument(
        "--kling-config",
        default="configs/reward/kling_video_reward.yaml",
        help="Reward YAML providing reward.kwargs.kling_video_reward.worker_config.",
    )
    parser.add_argument(
        "--vbench-dims",
        default=",".join(_DEFAULT_VBENCH_DIMS),
        help="Comma-separated VBench dimensions, or empty to skip VBench.",
    )
    parser.add_argument(
        "--vbench-full-info",
        default="",
        help="Path to VBench_full_info.json (required by VBench when --vbench-dims is set).",
    )
    parser.add_argument("--no-kling", action="store_true", help="Skip Kling scoring.")
    parser.add_argument(
        "--videoscore2",
        action="store_true",
        help="Also score with VideoScore2 (downloads/loads the 7B judge).",
    )
    parser.add_argument(
        "--merge-json",
        action="append",
        default=[],
        help=(
            "prefix=path to a per-video metric JSON ({filename: {metric: score}}) to merge "
            "as <prefix>_<metric> columns. Run external benchmarks (VMBench, VBench-2.0, "
            "DynamicEval) via their own CLIs and fold their results in this way."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Max videos to score (0 = all).")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Defaults to cuda:0 when CUDA is available, else cpu.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    video_dir = Path(args.video_dir).expanduser().resolve()
    if not video_dir.is_dir():
        raise NotADirectoryError(f"--video-dir is not a directory: {video_dir}")
    videos = sorted(video_dir.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"no .mp4 files found in {video_dir}")

    prompts = _pair_prompts(videos, args.manifest)
    output = Path(args.output).expanduser().resolve() if args.output else (
        video_dir / "eval_video_metrics.csv"
    )

    rows: list[dict[str, Any]] = [
        {"video_path": str(video), "prompt": prompts[video]} for video in videos
    ]

    if not args.no_kling:
        device = _resolve_device(args.device)
        kling_scores = _score_kling(videos, prompts, args.kling_config, device=device)
        for row, video in zip(rows, videos, strict=True):
            for key in _KLING_SCORE_KEYS:
                row[f"kling_{key}"] = kling_scores[str(video)].get(key)

    vbench_dims = [dim.strip() for dim in str(args.vbench_dims).split(",") if dim.strip()]
    if vbench_dims:
        vbench_scores = _score_vbench(
            video_dir,
            videos,
            vbench_dims,
            full_info=args.vbench_full_info,
            device=args.device,
        )
        for row, video in zip(rows, videos, strict=True):
            for dim in vbench_dims:
                row[f"vbench_{dim}"] = vbench_scores.get(str(video), {}).get(dim)

    for prefix, model in _optional_reward_models(args).items():
        scores = _score_reward_model(videos, prompts, model)
        for row, video in zip(rows, videos, strict=True):
            for key, value in scores.get(str(video), {}).items():
                row[f"{prefix}_{key}"] = value

    for spec in args.merge_json:
        prefix, merged = _load_merge_json(spec)
        for row, video in zip(rows, videos, strict=True):
            for metric, value in merged.get(video.name, {}).items():
                row[f"{prefix}_{metric}"] = value

    _write_csv(rows, output)
    logger.info("wrote %d rows to %s", len(rows), output)
    print(json.dumps({"output": str(output), "videos": len(rows)}, indent=2))


def _pair_prompts(videos: list[Path], manifest: str) -> dict[Path, str]:
    """Map each video to a prompt: by reference_video filename, else by order."""

    if not manifest:
        return {video: "" for video in videos}
    examples = load_prompt_manifest(manifest)
    by_name = {
        Path(example.reference_video).name: example.prompt
        for example in examples
        if str(example.reference_video).strip()
    }
    paired: dict[Path, str] = {}
    for index, video in enumerate(videos):
        if video.name in by_name:
            paired[video] = by_name[video.name]
        elif index < len(examples):
            paired[video] = examples[index].prompt
        else:
            paired[video] = ""
    return paired


def _score_kling(
    videos: list[Path],
    prompts: dict[Path, str],
    kling_config: str,
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    worker_config = _kling_worker_config(kling_config)
    worker_config.setdefault("device", str(device))
    logger.info("loading Kling VideoReward for %d videos", len(videos))
    model = KlingVideoRewardModel(worker_config)
    scores: dict[str, dict[str, float]] = {}
    try:
        for video in videos:
            artifact = RewardInferenceArtifact(
                artifact_id=video.stem,
                path=str(video),
                media_type="video",
                prompt=prompts[video],
            )
            request = RewardInferenceRequest(
                request_id="video-reward-suite",
                artifacts=(artifact,),
                reward_name="kling_video_reward",
                score_key="overall_reward",
            )
            scores[str(video)] = dict(model(artifact=artifact, request=request))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores


def _kling_worker_config(kling_config: str) -> dict[str, Any]:
    cfg = OmegaConf.load(Path(kling_config).expanduser().resolve())
    reward_cfg = OmegaConf.select(cfg, "reward.kwargs.kling_video_reward", default={})
    reward_cfg = OmegaConf.to_container(reward_cfg, resolve=False) or {}
    if not isinstance(reward_cfg, dict):
        raise ValueError(f"{kling_config} has no reward.kwargs.kling_video_reward mapping")
    worker_config = dict(reward_cfg.get("worker_config") or {})
    worker_config.setdefault(
        "reward_model_name",
        str(reward_cfg.get("reward_name") or "KlingTeam/VideoReward@main"),
    )
    return worker_config


def _score_vbench(
    video_dir: Path,
    videos: list[Path],
    dimensions: list[str],
    *,
    full_info: str,
    device: str,
) -> dict[str, dict[str, float]]:
    """Run VBench over the video dir (custom-input) and parse per-video scores.

    Best-effort: VBench is an optional, heavyweight dependency with its own model
    weights. Any failure (missing package, missing full_info, runtime error)
    logs a warning and returns empty scores so the CSV schema is preserved.
    """

    try:
        from vbench import VBench  # type: ignore
    except ImportError:
        logger.warning("VBench is not installed; leaving vbench_* columns empty")
        return {}
    if not full_info:
        logger.warning(
            "--vbench-full-info not set (path to VBench_full_info.json); skipping VBench",
        )
        return {}

    output_dir = video_dir / "vbench_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    try:
        evaluator = VBench(torch_device, str(Path(full_info).expanduser()), str(output_dir))
        evaluator.evaluate(
            videos_path=str(video_dir),
            name="video_reward_suite",
            dimension_list=list(dimensions),
            mode="custom_input",
        )
    except Exception as exc:  # optional tool; never fail the suite on its errors
        logger.warning("VBench evaluation failed (%s); leaving vbench_* columns empty", exc)
        return {}
    return _parse_vbench_results(output_dir, videos)


def _parse_vbench_results(
    output_dir: Path,
    videos: list[Path],
) -> dict[str, dict[str, float]]:
    """Parse VBench ``<name>_eval_results.json`` into ``{video_path: {dim: score}}``.

    VBench writes ``{dimension: [aggregate, [per_video_entry, ...]]}``; each
    per-video entry is a ``{video_path, video_results}`` dict. We match entries to
    our videos by filename so absolute-path differences don't drop rows.
    """

    candidates = sorted(output_dir.glob("*_eval_results.json"))
    if not candidates:
        logger.warning("no VBench *_eval_results.json found in %s", output_dir)
        return {}
    payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    by_name = {video.name: str(video) for video in videos}
    scores: dict[str, dict[str, float]] = {}
    for dimension, entry in payload.items():
        per_video = entry[1] if isinstance(entry, list) and len(entry) > 1 else entry
        if not isinstance(per_video, list):
            continue
        for item in per_video:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("video_path") or item.get("video") or "")
            value = item.get("video_results", item.get("score"))
            target = by_name.get(Path(raw_path).name)
            if target is None or value is None:
                continue
            try:
                scores.setdefault(target, {})[str(dimension)] = float(value)
            except (TypeError, ValueError):
                continue
    return scores


def _optional_reward_models(args: argparse.Namespace) -> dict[str, Any]:
    """Instantiate the repo-owned video reward models enabled by CLI flags.

    Keyed by the CSV column prefix. Each is a ``RewardModel`` whose ``__call__``
    returns named scores we emit as ``<prefix>_<key>`` columns.
    """

    device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    models: dict[str, Any] = {}
    if args.videoscore2:
        from vrl.rewards.models.videoscore2 import VideoScore2Model

        models["vs2"] = VideoScore2Model({"device": device, "dtype": "bfloat16"})
    return models


def _score_reward_model(
    videos: list[Path],
    prompts: dict[Path, str],
    model: Any,
) -> dict[str, dict[str, float]]:
    """Run a RewardModel over each video, returning ``{video_path: {key: score}}``."""

    scores: dict[str, dict[str, float]] = {}
    try:
        for video in videos:
            artifact = RewardInferenceArtifact(
                artifact_id=video.stem,
                path=str(video),
                media_type="video",
                prompt=prompts[video],
            )
            request = RewardInferenceRequest(
                request_id="video-reward-suite",
                artifacts=(artifact,),
                reward_name="video_reward_suite",
                score_key="overall",
            )
            scores[str(video)] = {
                str(key): float(value)
                for key, value in model(artifact=artifact, request=request).items()
            }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores


def _load_merge_json(spec: str) -> tuple[str, dict[str, dict[str, float]]]:
    """Parse a ``prefix=path`` external-metric JSON into ``(prefix, {filename: {metric: score}})``."""

    if "=" not in spec:
        raise ValueError(f"--merge-json expects prefix=path, got {spec!r}")
    prefix, path = spec.split("=", 1)
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"--merge-json {path!r} must be a JSON object keyed by video filename")
    merged: dict[str, dict[str, float]] = {}
    for name, metrics in payload.items():
        if isinstance(metrics, dict):
            merged[Path(str(name)).name] = {
                str(metric): float(value) for metric, value in metrics.items()
            }
    return prefix.strip(), merged


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        device = torch.device(device_arg)
        if getattr(device, "type", str(device)) == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({device_arg}) but CUDA is unavailable")
        return device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    # Keep identity columns first for readability.
    for lead in ("prompt", "video_path"):
        if lead in fieldnames:
            fieldnames.remove(lead)
            fieldnames.insert(0, lead)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
