"""Fixed held-out evaluation for Wan robotics full-parameter checkpoints.

Generation and scoring are separate commands so checkpoint shards can use
different GPUs while every shard remains bound to one manifest, sampling, and
seed protocol. The strict evaluation pool excludes training prompts and source
video shards before deterministic identity hashing selects the requested rows.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from vrl.config.loading import load_config
from vrl.families.registry import get_model_family_entry
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    sha256_file,
)
from vrl.rewards.models.robotics_video_reward import RoboticsVideoRewardModel
from vrl.scripts.eval.denoise_video_generation import generate_one_video
from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    is_complete_checkpoint,
    load_trainable_state,
    load_training_checkpoint,
    read_checkpoint_meta,
)
from vrl.trainers.data import PromptExample, load_prompt_manifest
from vrl.utils.artifacts import resolve_artifact_path
from vrl.utils.media import write_mp4

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "vrl.wan_robotics_checkpoint_eval/v1"
REPORT_SCHEMA_VERSION = 1
DEFAULT_TARGETS = ("base", "5", "10", "15", "20")
DEFAULT_BASE_SEED = 2_026_072_200
DEFAULT_SEED_STRIDE = 1_000
BOOTSTRAP_RESAMPLES = 2_000
SCORE_KEYS = (
    "robotics_blend",
    "target_dino_similarity",
    "motion_dynamics",
    "kling_text_alignment",
    "kling_visual_quality",
    "kling_motion_quality",
    "kling_overall_reward",
)


@dataclass(frozen=True, slots=True)
class CheckpointTarget:
    """One base/checkpoint model selected from the evaluated run."""

    label: str
    epoch: int
    path: Path | None
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SelectedExample:
    """One held-out manifest row in the deterministic strict evaluation pool."""

    row_index: int
    identity_sha256: str
    prompt: str
    target_video: str
    source_episode: str
    source_video: str


@dataclass(frozen=True, slots=True)
class GeneratedVideo:
    """One materialized checkpoint/prompt/seed grid cell."""

    checkpoint_label: str
    epoch: int
    row_index: int
    sample_index: int
    seed: int
    prompt: str
    target_video: str
    source_episode: str
    path: str
    bytes: int
    sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate one base/checkpoint shard.")
    _add_common_args(generate)
    generate.add_argument(
        "--target",
        required=True,
        help="Exactly one target: base or a numbered checkpoint epoch such as 5.",
    )
    generate.add_argument(
        "--limit",
        type=int,
        default=16,
        help="Number of deterministically selected strict held-out rows.",
    )
    generate.add_argument("--samples-per-prompt", type=int, default=2)
    generate.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    generate.add_argument("--seed-stride", type=int, default=DEFAULT_SEED_STRIDE)
    generate.add_argument(
        "--device",
        default="auto",
        help="Generation device; auto selects cuda:0 when available.",
    )

    score = commands.add_parser("score", help="Score and aggregate complete shards.")
    _add_common_args(score)
    score.add_argument(
        "--target",
        action="append",
        default=[],
        help="Target label to require; repeat as needed. Defaults to base/5/10/15/20.",
    )
    score.add_argument(
        "--device",
        default="auto",
        help="Reward device; auto selects cuda:0 when available.",
    )
    score.add_argument(
        "--no-reference-targets",
        action="store_true",
        help="Do not score the exact held-out target clips as a reward calibration anchor.",
    )
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output containing resolved_config.yaml and checkpoint-N directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dedicated evaluation output directory.",
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    result = generate_shard(args) if args.command == "generate" else score_shards(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def generate_shard(args: argparse.Namespace) -> dict[str, Any]:
    """Generate and atomically publish one protocol-bound checkpoint shard."""

    _validate_generation_args(args)
    run_dir, cfg, config_path = _load_run(args.run_dir)
    target = _resolve_target(run_dir, str(args.target))
    protocol = _build_protocol(
        cfg,
        config_path=config_path,
        limit=int(args.limit),
        samples_per_prompt=int(args.samples_per_prompt),
        base_seed=int(args.base_seed),
        seed_stride=int(args.seed_stride),
    )
    protocol_sha256 = _json_sha256(protocol)

    output_dir = args.output_dir.expanduser().resolve()
    generation_root = output_dir / "generation"
    final_dir = generation_root / target.label
    if final_dir.exists():
        raise FileExistsError(f"evaluation shard already exists: {final_dir}")
    staging_dir = generation_root / f".{target.label}.tmp-{os.getpid()}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    device = _resolve_device(str(args.device))
    generated: list[GeneratedVideo] = []
    bundle: Any | None = None
    model: Any | None = None
    try:
        entry = get_model_family_entry(str(cfg.model.family))
        if entry.family != "wan_2_1":
            raise ValueError(f"Wan robotics evaluation requires wan_2_1; got {entry.family!r}")
        build = entry.resolve_model_build(cfg, device, for_rollout=True)
        bundle = entry.build_rollout(build)
        if target.path is not None:
            checkpoint = load_training_checkpoint(target.path)
            _validate_loaded_checkpoint(checkpoint, target)
            trainable_state = checkpoint.trainable_state
            load_trainable_state(bundle, trainable_state, strict=True)
            del trainable_state, checkpoint
            gc.collect()
        model = bundle.model.eval()

        selected = [SelectedExample(**row) for row in protocol["selected_examples"]]
        for selected_index, example in enumerate(selected):
            for sample_index in range(int(args.samples_per_prompt)):
                seed = _seed_for(
                    base_seed=int(args.base_seed),
                    row_index=example.row_index,
                    sample_index=sample_index,
                    seed_stride=int(args.seed_stride),
                )
                logger.info(
                    "Generating target=%s heldout=%d/%d row=%d sample=%d seed=%d",
                    target.label,
                    selected_index + 1,
                    len(selected),
                    example.row_index,
                    sample_index,
                    seed,
                )
                video = generate_one_video(
                    model,
                    prompt=example.prompt,
                    seed=seed,
                    sampling=dict(protocol["sampling"]),
                )
                if not bool(torch.isfinite(video).all()) or float(video.float().std()) < 1e-3:
                    raise RuntimeError(
                        f"generated video is non-finite or degenerate: target={target.label} "
                        f"row={example.row_index} sample={sample_index}",
                    )
                relative_path = Path("videos") / (
                    f"row{example.row_index:04d}_sample{sample_index:02d}.mp4"
                )
                staging_path = staging_dir / relative_path
                final_path = final_dir / relative_path
                write_mp4(video, staging_path, fps=float(protocol["sampling"]["fps"]))
                generated.append(
                    GeneratedVideo(
                        checkpoint_label=target.label,
                        epoch=target.epoch,
                        row_index=example.row_index,
                        sample_index=sample_index,
                        seed=seed,
                        prompt=example.prompt,
                        target_video=example.target_video,
                        source_episode=example.source_episode,
                        path=str(final_path),
                        bytes=staging_path.stat().st_size,
                        sha256=sha256_file(staging_path),
                    ),
                )
                del video
    finally:
        del model, bundle
        _release_cuda()

    provenance = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
        "target": {
            "label": target.label,
            "epoch": target.epoch,
            **target.provenance,
        },
        "execution": {"device": str(device)},
        "sample_count": len(generated),
    }
    _write_jsonl(staging_dir / "generated.jsonl", [asdict(video) for video in generated])
    _write_json(staging_dir / "provenance.json", provenance)
    _fsync_tree(staging_dir)
    generation_root.mkdir(parents=True, exist_ok=True)
    os.replace(staging_dir, final_dir)
    _fsync_directory(generation_root)
    return {
        "output_dir": str(final_dir),
        "protocol_sha256": protocol_sha256,
        "target": target.label,
        "videos": len(generated),
    }


def score_shards(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, score, and atomically publish one paired checkpoint report."""

    run_dir, cfg, config_path = _load_run(args.run_dir)
    del config_path
    target_values = list(args.target or DEFAULT_TARGETS)
    targets = [_resolve_target(run_dir, value) for value in target_values]
    labels = [target.label for target in targets]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate evaluation targets: {labels}")
    if "base" not in labels:
        raise ValueError("paired checkpoint scoring requires the base shard")

    output_dir = args.output_dir.expanduser().resolve()
    scoring_dir = output_dir / "scoring"
    if scoring_dir.exists():
        raise FileExistsError(f"evaluation scoring output already exists: {scoring_dir}")
    staging_dir = output_dir / f".scoring.tmp-{os.getpid()}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    provenance_rows, generated = _load_and_validate_shards(output_dir, targets)
    protocol = provenance_rows[0]["protocol"]
    artifacts, artifact_rows = _build_scoring_artifacts(
        generated,
        protocol,
        cfg,
        include_reference_targets=not bool(args.no_reference_targets),
    )
    request = RewardInferenceRequest(
        request_id="wan-robotics-fixed-checkpoint-eval",
        artifacts=tuple(artifacts),
        reward_name="robotics_video_reward",
        score_key="robotics_blend",
    )
    device = _resolve_device(str(args.device))
    worker_config = _reward_worker_config(cfg, device=device)
    logger.info("Loading robotics reward to score %d fixed artifacts", len(artifacts))
    model = RoboticsVideoRewardModel(worker_config)
    scored: list[dict[str, Any]] = []
    try:
        model.prepare_for_inference()
        for index, (artifact, row) in enumerate(
            zip(request.artifacts, artifact_rows, strict=True),
        ):
            scores = {
                key: float(value)
                for key, value in model(artifact=artifact, request=request).items()
            }
            missing = sorted(set(SCORE_KEYS) - set(scores))
            if missing or any(not math.isfinite(scores[key]) for key in SCORE_KEYS):
                raise ValueError(
                    f"invalid robotics scores for {artifact.artifact_id}: missing={missing} "
                    f"scores={scores}",
                )
            scored.append({**row, **{f"r_{key}": scores[key] for key in SCORE_KEYS}})
            logger.info(
                "Scored artifact=%d/%d label=%s row=%s blend=%.4f",
                index + 1,
                len(artifacts),
                row["checkpoint_label"],
                row["row_index"],
                scores["robotics_blend"],
            )
    finally:
        del model
        _release_cuda()

    summary = summarize_scores(scored)
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_sha256": provenance_rows[0]["protocol_sha256"],
        "protocol": protocol,
        "targets": [row["target"] for row in provenance_rows],
        "execution": {"reward_device": str(device)},
        "sample_count": len(scored),
        "summary": summary,
    }
    _write_jsonl(staging_dir / "scores.jsonl", scored)
    _write_csv(staging_dir / "scores.csv", scored)
    _write_json(staging_dir / "report.json", report)
    _fsync_tree(staging_dir)
    os.replace(staging_dir, scoring_dir)
    _fsync_directory(output_dir)
    return {
        "output_dir": str(scoring_dir),
        "report": str(scoring_dir / "report.json"),
        "summary": summary,
    }


def _validate_generation_args(args: argparse.Namespace) -> None:
    if int(args.limit) < 1:
        raise ValueError("--limit must be >= 1")
    if int(args.samples_per_prompt) < 1:
        raise ValueError("--samples-per-prompt must be >= 1")
    if int(args.seed_stride) < 1:
        raise ValueError("--seed-stride must be >= 1")


def _load_run(run_dir_arg: Path) -> tuple[Path, DictConfig, Path]:
    run_dir = run_dir_arg.expanduser().resolve()
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"training run has no resolved config: {config_path}")
    cfg = load_config(config_path)
    entry = get_model_family_entry(str(cfg.model.family))
    if entry.family != "wan_2_1":
        raise ValueError(f"expected Wan 2.1 T2V run, got {entry.family!r}")
    if OmegaConf.select(cfg, "model.use_lora", default=None) is not False:
        raise ValueError("Wan robotics checkpoint evaluation accepts full-parameter runs only")
    return run_dir, cfg, config_path


def _resolve_target(run_dir: Path, raw_value: str) -> CheckpointTarget:
    value = str(raw_value).strip().lower()
    if value in {"base", "baseline", "0", "checkpoint-0"}:
        return CheckpointTarget(
            label="base",
            epoch=0,
            path=None,
            provenance={"source": "resolved_config_base_model", "checkpoint_loaded": False},
        )
    if value.startswith("checkpoint-"):
        value = value.removeprefix("checkpoint-")
    try:
        epoch = int(value)
    except ValueError as exc:
        raise ValueError(
            f"target must be base or a numbered checkpoint epoch: {raw_value!r}"
        ) from exc
    if epoch <= 0:
        raise ValueError(f"checkpoint epoch must be > 0, got {epoch}")
    path = (run_dir / f"checkpoint-{epoch}").resolve()
    if not is_complete_checkpoint(path):
        raise ValueError(f"checkpoint is missing or incomplete: {path}")
    meta = read_checkpoint_meta(path)
    if str(meta.get("family", "")) != "wan_2_1":
        raise ValueError(f"checkpoint family is not wan_2_1: {path}")
    if meta.get("uses_lora") is not False:
        raise ValueError(f"checkpoint must declare uses_lora=false: {path}")
    if int(meta.get("completed_epoch", -1)) != epoch:
        raise ValueError(
            f"checkpoint epoch disagrees with directory: {path} "
            f"completed_epoch={meta.get('completed_epoch')!r}",
        )
    checkpoint_file = path / TRAINING_CHECKPOINT_NAME
    return CheckpointTarget(
        label=f"checkpoint-{epoch}",
        epoch=epoch,
        path=path,
        provenance={
            "path": str(path),
            "checkpoint_file": str(checkpoint_file),
            "checkpoint_bytes": checkpoint_file.stat().st_size,
            "checkpoint_meta": meta,
        },
    )


def _validate_loaded_checkpoint(checkpoint: Any, target: CheckpointTarget) -> None:
    if target.path is None:
        raise ValueError("base target must not load a checkpoint")
    if str(checkpoint.payload.get("family", "")) != "wan_2_1":
        raise ValueError(f"checkpoint payload family is not wan_2_1: {target.path}")
    if checkpoint.meta != target.provenance["checkpoint_meta"]:
        raise RuntimeError(f"checkpoint metadata changed during evaluation: {target.path}")
    if checkpoint.next_epoch != target.epoch:
        raise ValueError(
            f"checkpoint progress disagrees with target: {target.path} "
            f"next_epoch={checkpoint.next_epoch}",
        )


def _build_protocol(
    cfg: DictConfig,
    *,
    config_path: Path,
    limit: int,
    samples_per_prompt: int,
    base_seed: int,
    seed_stride: int,
) -> dict[str, Any]:
    train_path = Path(str(cfg.data.manifest)).expanduser().resolve()
    eval_path = Path(str(cfg.data.eval_manifest)).expanduser().resolve()
    train_examples = load_prompt_manifest(train_path)
    eval_examples = load_prompt_manifest(eval_path)
    selected = select_strict_examples(train_examples, eval_examples, limit=limit)
    return {
        "resolved_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "model": {
            "family": "wan_2_1",
            "path": str(cfg.model.path),
            "revision": str(OmegaConf.select(cfg, "model.revision", default="")),
        },
        "training_manifest": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "rows": len(train_examples),
        },
        "eval_manifest": {
            "path": str(eval_path),
            "sha256": sha256_file(eval_path),
            "rows": len(eval_examples),
        },
        "selection": {
            "name": "strict_source_shard_and_prompt_disjoint_sha256_v1",
            "limit": limit,
        },
        "selected_examples": [asdict(example) for example in selected],
        "seed_grid": {
            "base_seed": base_seed,
            "seed_stride": seed_stride,
            "samples_per_prompt": samples_per_prompt,
            "formula": "base_seed + sample_index * seed_stride + eval_row_index",
        },
        "sampling": _resolve_sampling(cfg),
    }


def select_strict_examples(
    train_examples: list[PromptExample],
    eval_examples: list[PromptExample],
    *,
    limit: int,
) -> list[SelectedExample]:
    """Select deterministic eval rows with prompt, episode, target, and shard isolation."""

    train_prompts = {_normalized_prompt(example.prompt) for example in train_examples}
    train_source_videos = {_metadata_text(example, "source_video") for example in train_examples}
    train_episodes = {_metadata_text(example, "source_episode") for example in train_examples}
    train_targets = {str(example.target_video or "") for example in train_examples}
    eval_episodes = {_metadata_text(example, "source_episode") for example in eval_examples}
    eval_targets = {str(example.target_video or "") for example in eval_examples}
    if train_episodes & eval_episodes:
        raise ValueError("training and evaluation source episodes overlap")
    if train_targets & eval_targets:
        raise ValueError("training and evaluation target videos overlap")

    eligible: list[SelectedExample] = []
    for row_index, example in enumerate(eval_examples):
        source_video = _metadata_text(example, "source_video")
        source_episode = _metadata_text(example, "source_episode")
        target_video = str(example.target_video or "").strip()
        if not source_video or not source_episode or not target_video:
            raise ValueError(f"evaluation row {row_index} lacks source/target identity")
        if _normalized_prompt(example.prompt) in train_prompts:
            continue
        if source_video in train_source_videos:
            continue
        identity = "\0".join(
            (
                "vrl-wan-droid-fixed-eval-v1",
                _metadata_text(example, "source_repo"),
                source_episode,
                _metadata_text(example, "source_frame_index"),
                example.prompt,
                target_video,
            ),
        )
        eligible.append(
            SelectedExample(
                row_index=row_index,
                identity_sha256=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                prompt=example.prompt,
                target_video=target_video,
                source_episode=source_episode,
                source_video=source_video,
            ),
        )
    eligible.sort(key=lambda example: (example.identity_sha256, example.row_index))
    if len(eligible) < limit:
        raise ValueError(
            f"strict evaluation pool has {len(eligible)} rows, fewer than requested {limit}",
        )
    return eligible[:limit]


def _metadata_text(example: PromptExample, key: str) -> str:
    value = example.metadata.get(key)
    return "" if value is None else str(value).strip()


def _normalized_prompt(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _seed_for(*, base_seed: int, row_index: int, sample_index: int, seed_stride: int) -> int:
    return int(base_seed) + int(sample_index) * int(seed_stride) + int(row_index)


def _resolve_sampling(cfg: DictConfig) -> dict[str, Any]:
    return {
        "width": int(cfg.sampling.width),
        "height": int(cfg.sampling.height),
        "num_frames": int(cfg.sampling.num_frames),
        "num_steps": int(cfg.sampling.num_steps),
        "fps": int(cfg.sampling.fps),
        "max_sequence_length": int(cfg.sampling.max_sequence_length),
        "guidance_scale": float(cfg.sampling.guidance_scale),
        "denoise_mode": str(cfg.rollout.denoise_mode),
        "noise_level": float(cfg.rollout.noise_level),
        "sde_type": str(cfg.rollout.sde.type),
    }


def _load_and_validate_shards(
    output_dir: Path,
    targets: list[CheckpointTarget],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provenance_rows: list[dict[str, Any]] = []
    generated_by_label: dict[str, list[dict[str, Any]]] = {}
    protocol_sha256: str | None = None
    expected_cells: set[tuple[Any, ...]] | None = None
    for target in targets:
        shard_dir = output_dir / "generation" / target.label
        provenance_path = shard_dir / "provenance.json"
        generated_path = shard_dir / "generated.jsonl"
        if not provenance_path.is_file() or not generated_path.is_file():
            raise FileNotFoundError(f"incomplete evaluation shard: {shard_dir}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("schema") != REPORT_SCHEMA:
            raise ValueError(f"evaluation shard has an unknown schema: {shard_dir}")
        if provenance.get("target", {}).get("label") != target.label:
            raise ValueError(f"evaluation shard target mismatch: {shard_dir}")
        observed_protocol_sha256 = str(provenance.get("protocol_sha256", ""))
        if _json_sha256(provenance.get("protocol")) != observed_protocol_sha256:
            raise ValueError(f"evaluation shard protocol digest is invalid: {shard_dir}")
        if protocol_sha256 is None:
            protocol_sha256 = observed_protocol_sha256
        elif observed_protocol_sha256 != protocol_sha256:
            raise ValueError("evaluation shards use different prompt/seed/sampling protocols")
        rows = _read_jsonl(generated_path)
        if len(rows) != int(provenance.get("sample_count", -1)):
            raise ValueError(f"evaluation shard sample count changed: {shard_dir}")
        for row in rows:
            path = Path(str(row["path"]))
            if not path.is_file():
                raise FileNotFoundError(f"generated evaluation video is missing: {path}")
            if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                raise ValueError(f"generated evaluation video failed integrity check: {path}")
        cells = {
            (
                int(row["row_index"]),
                int(row["sample_index"]),
                int(row["seed"]),
                str(row["prompt"]),
                str(row["target_video"]),
            )
            for row in rows
        }
        if expected_cells is None:
            expected_cells = cells
        elif cells != expected_cells:
            raise ValueError(f"evaluation shard grid differs for target {target.label}")
        provenance_rows.append(provenance)
        generated_by_label[target.label] = rows
    generated = [row for target in targets for row in generated_by_label[target.label]]
    return provenance_rows, generated


def _build_scoring_artifacts(
    generated: list[dict[str, Any]],
    protocol: dict[str, Any],
    cfg: DictConfig,
    *,
    include_reference_targets: bool,
) -> tuple[list[RewardInferenceArtifact], list[dict[str, Any]]]:
    artifacts: list[RewardInferenceArtifact] = []
    rows: list[dict[str, Any]] = []
    for row in generated:
        path = Path(str(row["path"]))
        artifact = RewardInferenceArtifact(
            artifact_id=(
                f"{row['checkpoint_label']}-r{int(row['row_index']):04d}"
                f"-s{int(row['sample_index']):02d}"
            ),
            path=str(path),
            media_type="video",
            prompt=str(row["prompt"]),
            sample_id=f"r{row['row_index']}-s{row['sample_index']}",
            size_bytes=int(row["bytes"]),
            sha256=str(row["sha256"]),
            metadata={
                "target_video": str(row["target_video"]),
                "checkpoint_label": str(row["checkpoint_label"]),
                "seed": int(row["seed"]),
            },
        )
        artifacts.append(artifact)
        rows.append(dict(row))

    if include_reference_targets:
        data_root = Path(str(cfg.data.artifact_data_root)).expanduser().resolve()
        for selected in protocol["selected_examples"]:
            target_video = str(selected["target_video"])
            target_path = resolve_artifact_path(
                target_video,
                data_root=data_root,
                allow_absolute=True,
            )
            digest = sha256_file(target_path)
            artifacts.append(
                RewardInferenceArtifact(
                    artifact_id=f"reference-r{int(selected['row_index']):04d}",
                    path=str(target_path),
                    media_type="video",
                    prompt=str(selected["prompt"]),
                    sample_id=f"reference-r{selected['row_index']}",
                    size_bytes=target_path.stat().st_size,
                    sha256=digest,
                    metadata={"target_video": target_video, "checkpoint_label": "reference"},
                ),
            )
            rows.append(
                {
                    "checkpoint_label": "reference",
                    "epoch": -1,
                    "row_index": int(selected["row_index"]),
                    "sample_index": -1,
                    "seed": None,
                    "prompt": str(selected["prompt"]),
                    "target_video": target_video,
                    "source_episode": str(selected["source_episode"]),
                    "path": str(target_path),
                    "bytes": target_path.stat().st_size,
                    "sha256": digest,
                },
            )
    return artifacts, rows


def _reward_worker_config(cfg: DictConfig, *, device: torch.device) -> dict[str, Any]:
    reward_cfg = OmegaConf.select(cfg, "reward.kwargs.robotics_video_reward", default={})
    reward_cfg = OmegaConf.to_container(reward_cfg, resolve=True) or {}
    if not isinstance(reward_cfg, dict):
        raise ValueError("run config has no robotics_video_reward mapping")
    worker_config = dict(reward_cfg.get("worker_config") or {})
    worker_config["device"] = str(device)
    worker_config["data_root"] = str(
        Path(str(worker_config.get("data_root") or cfg.data.artifact_data_root))
        .expanduser()
        .resolve(),
    )
    return worker_config


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize absolute scores and paired checkpoint deltas from the base grid."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint_label"]), []).append(row)
    absolute: dict[str, Any] = {}
    for label, group in grouped.items():
        absolute[label] = {
            key: _distribution([float(row[f"r_{key}"]) for row in group]) for key in SCORE_KEYS
        }

    base = {
        (int(row["row_index"]), int(row["sample_index"])): row for row in grouped.get("base", [])
    }
    if not base:
        raise ValueError("paired score summary requires base rows")
    paired: dict[str, Any] = {}
    for label, group in grouped.items():
        if label in {"base", "reference"}:
            continue
        cells = {(int(row["row_index"]), int(row["sample_index"])): row for row in group}
        if set(cells) != set(base):
            raise ValueError(f"paired score grid differs for {label}")
        paired[label] = {}
        for key in SCORE_KEYS:
            deltas = [
                float(cells[cell][f"r_{key}"]) - float(base[cell][f"r_{key}"])
                for cell in sorted(base)
            ]
            lower, upper = _bootstrap_mean_interval(deltas, label=label, score_key=key)
            paired[label][key] = {
                **_distribution(deltas),
                "win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
                "tie_rate": sum(delta == 0 for delta in deltas) / len(deltas),
                "bootstrap_95ci": [lower, upper],
                "clear_improvement": lower > 0.0,
            }

    checkpoint_labels = [label for label in grouped if label not in {"base", "reference"}]
    best_label = max(
        ["base", *checkpoint_labels],
        key=lambda label: float(absolute[label]["robotics_blend"]["mean"]),
    )
    return {
        "absolute": absolute,
        "paired_delta_from_base": paired,
        "best_mean_robotics_blend": best_label,
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty score distribution")
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": std,
        "stderr": std / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _bootstrap_mean_interval(
    values: list[float],
    *,
    label: str,
    score_key: str,
) -> tuple[float, float]:
    seed_bytes = hashlib.sha256(f"{REPORT_SCHEMA}\0{label}\0{score_key}".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        means.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    means.sort()
    lower_index = int(0.025 * (len(means) - 1))
    upper_index = int(0.975 * (len(means) - 1))
    return means[lower_index], means[upper_index]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"JSONL row must be an object: {path}")
                rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fsync_tree(path: Path) -> None:
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        with file_path.open("rb") as handle:
            os.fsync(handle.fileno())
    for directory in sorted(
        (item for item in path.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(path)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
