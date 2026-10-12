"""Evaluate a SANA training run on a configurable aesthetic curve.

The training process owns checkpoints only. This standalone process loads the
run's resolved config and complete ``checkpoint-N`` directories, evaluates the
base model and selected checkpoints on one configured prompt/seed grid,
then writes a provenance-bound report for ``sana_aesthetic_curve_verdict``.

The run config supplies model, manifests and rewards. CLI options select the
paired seed/sample grid and checkpoint interval; the report records those choices.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import RootConfig, parse_config
from vrl.scripts.eval import sana_aesthetic_report as sana_report
from vrl.scripts.eval._device import resolve_eval_device
from vrl.scripts.eval.sana_inference import (
    SANA_EVAL_SAMPLING_CONFIG,
    SANA_EVAL_SCHEDULER_CONFIG,
    generate_prompt_images,
    load_official_scheduler,
)
from vrl.trainers.checkpointing import (
    RESOLVED_CONFIG_NAME,
    TRAINING_CHECKPOINT_NAME,
    CheckpointTarget,
    TrainingCheckpoint,
    load_resolved_run_config,
)
from vrl.trainers.data.prompts import load_prompt_dataset_index
from vrl.utils.artifacts import sha256_file, user_path
from vrl.utils.cuda_memory import release_cuda_memory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """One materialized cell in the checkpoint/prompt/sample grid."""

    checkpoint_label: str
    epoch: int
    prompt_index: int
    sample_index: int
    group_seed: int
    prompt: str
    path: Path
    # Provenance-only: binds the qualitative artifact to the scored sample row.
    image_sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output directory containing resolved_config.yaml and checkpoint-N dirs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device; auto selects cuda:0 when available, otherwise cpu.",
    )
    defaults = sana_report.EvaluationSettings()
    parser.add_argument("--seed", type=int, default=defaults.base_seed)
    parser.add_argument("--samples-per-prompt", type=int, default=defaults.samples_per_prompt)
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = sana_report.EvaluationSettings(
        base_seed=args.seed,
        samples_per_prompt=args.samples_per_prompt,
        checkpoint_interval=args.checkpoint_interval,
    )
    run_dir = user_path(args.run_dir)
    config_path = run_dir / RESOLVED_CONFIG_NAME
    cfg, _ = load_resolved_run_config(run_dir)
    training_metrics_path = run_dir / "metrics.csv"
    if not training_metrics_path.is_file():
        raise FileNotFoundError(f"training run has no metrics CSV: {training_metrics_path}")

    root = sana_report.validate_run_config(cfg)
    sana_report.validate_training_metrics(training_metrics_path, root)
    training_manifest_path, eval_manifest_path, prompts = sana_report.resolve_protocol_manifests(
        root,
    )
    if not prompts:
        raise ValueError(f"evaluation manifest has no prompts: {eval_manifest_path}")
    training_log = sana_report.require_training_log_provenance(run_dir, root)

    targets = _discover_checkpoint_targets(run_dir, root, settings)
    device = resolve_eval_device(args.device)
    sampling = dict(SANA_EVAL_SAMPLING_CONFIG)
    if root.model is None:
        raise ValueError("SANA checkpoint evaluation requires model configuration")
    identity_precision = PrecisionPolicy.from_section(root.precision)
    from vrl.models.families.registry import get_model_family_entry

    identity_entry = get_model_family_entry(str(root.model.family))
    model_identity = identity_entry.resolve_model_build(
        root, device, precision=identity_precision, for_rollout=True
    ).checkpoint_identity()
    for target in targets:
        if target.path is not None:
            target.validate_compatibility(family="sana", expected_model_identity=model_identity)
    build_root = _materialize_model_snapshot(cfg)
    build_precision = PrecisionPolicy.from_section(build_root.precision)
    reward_models = _materialize_reward_model_snapshots(
        sana_report.build_reward_model_definitions(root, generation_device=str(device)),
    )
    eval_dir = run_dir / sana_report.REPORT_RELATIVE_PATH.parent
    eval_dir.mkdir(parents=True, exist_ok=True)

    generated = _generate_images(
        build_root,
        build_precision,
        targets,
        prompts,
        output_dir=eval_dir,
        sampling=sampling,
        device=device,
        expected_model_identity=model_identity,
        settings=settings,
    )
    sample_scores = _score_images(generated, reward_models)
    sample_path = run_dir / sana_report.SAMPLES_RELATIVE_PATH
    sana_report.write_sample_manifest(sample_path, sample_scores, base_dir=run_dir)
    metrics = sana_report.summarize_scores(sample_scores)
    if not metrics:
        raise RuntimeError("SANA checkpoint evaluation produced no summary rows")

    provenance = {
        "run": {
            "path": str(run_dir),
            "training_metrics_path": training_metrics_path.name,
            "training_metrics_sha256": sha256_file(training_metrics_path),
        },
        "training_log": training_log,
        "resolved_config": {
            "path": config_path.name,
            "sha256": sha256_file(config_path),
        },
        "model": {
            "family": str(cfg.model.family),
            "repo": str(cfg.model.path),
            "revision": str(cfg.model.revision),
        },
        "training_manifest": {
            "path": str(training_manifest_path),
            "sha256": sha256_file(training_manifest_path),
            "prompt_count": len(load_prompt_dataset_index(training_manifest_path)),
        },
        "eval_manifest": {
            "path": str(eval_manifest_path),
            "sha256": sha256_file(eval_manifest_path),
            "prompt_count": len(prompts),
        },
        "seed_grid": settings.seed_grid_record(),
        "evaluation_curve": {"checkpoint_interval": settings.checkpoint_interval},
        "sampling": sampling,
        "scheduler_protocol": dict(SANA_EVAL_SCHEDULER_CONFIG),
        "execution": {"generation_device": str(device)},
        "rewards": [reward_model.to_report_record() for reward_model in reward_models],
        "checkpoints": [_checkpoint_record(target, run_dir) for target in targets],
        "samples": {
            "path": str(sana_report.SAMPLES_RELATIVE_PATH),
            "sha256": sha256_file(sample_path),
            "count": len(sample_scores),
        },
    }
    report_path = sana_report.publish_report(
        run_dir,
        provenance=provenance,
        metrics=metrics,
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "curve_points": len(metrics),
                "scored_images": len(sample_scores),
            },
            indent=2,
            sort_keys=True,
        ),
    )


def _discover_checkpoint_targets(
    run_dir: Path,
    root: RootConfig,
    settings: sana_report.EvaluationSettings | None = None,
) -> list[CheckpointTarget]:
    settings = settings or sana_report.EvaluationSettings()
    numbered: list[tuple[int, CheckpointTarget]] = []
    for candidate in run_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        target = CheckpointTarget.load(candidate, label=candidate.name, digest=True)
        if str(target.meta.get("family", "")) != "sana":
            raise ValueError(f"checkpoint family is not sana: {candidate}")
        if target.epoch != epoch:
            raise ValueError(
                f"checkpoint epoch disagrees with directory name: {candidate} "
                f"completed_epoch={target.epoch!r}",
            )
        numbered.append((epoch, target))
    if not numbered:
        raise ValueError(f"training run has no complete checkpoint-N directories: {run_dir}")
    numbered.sort(key=lambda item: item[0])
    expected = sana_report.checkpoint_curve_epochs(root, settings)
    checkpoint_targets = [
        target for epoch, target in numbered if epoch % settings.checkpoint_interval == 0
    ]
    epochs = [target.epoch for target in checkpoint_targets]
    if epochs != expected:
        raise ValueError(
            f"checkpoint curve is incomplete or has gaps: found={epochs}, expected={expected}",
        )
    baseline = CheckpointTarget.base("baseline")
    return [baseline, *checkpoint_targets]


def _materialize_model_snapshot(cfg: DictConfig) -> RootConfig:
    """Pin SANA architecture/frozen modules through the official Hub API."""

    from vrl.utils.hf_hub import HuggingFaceRepoRevision

    local_path = str(
        HuggingFaceRepoRevision(str(cfg.model.path), str(cfg.model.revision)).download()
    )
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    plain["model"]["path"] = local_path
    materialized = OmegaConf.create(plain)
    OmegaConf.resolve(materialized)
    assert isinstance(materialized, DictConfig)
    return parse_config(materialized)


def _materialize_reward_model_snapshots(
    reward_models: list[sana_report.RewardModelDefinition],
) -> list[sana_report.RewardModelDefinition]:
    """Resolve immutable reward snapshots without exposing local paths in reports."""

    from vrl.utils.hf_hub import HuggingFaceRepoRevision

    materialized: list[sana_report.RewardModelDefinition] = []
    for reward_model in reward_models:
        model_config = dict(reward_model.model_config)
        if reward_model.name == "aesthetic":
            identity = reward_model.provenance["model"]
            model_config["model_name"] = str(
                HuggingFaceRepoRevision(
                    str(identity["repo"]), str(identity["revision"])
                ).download(),
            )
        elif reward_model.name == "pickscore":
            processor = reward_model.provenance["processor"]
            model = reward_model.provenance["model"]
            model_config["processor_name"] = str(
                HuggingFaceRepoRevision(
                    str(processor["repo"]), str(processor["revision"])
                ).download(),
            )
            model_config["model_name"] = str(
                HuggingFaceRepoRevision(str(model["repo"]), str(model["revision"])).download(),
            )
        else:
            raise ValueError(
                f"unsupported SANA evaluation reward: {reward_model.name!r}",
            )
        materialized.append(
            sana_report.RewardModelDefinition(
                name=reward_model.name,
                score_key=reward_model.score_key,
                model_factory=reward_model.model_factory,
                model_config=model_config,
                provenance=reward_model.provenance,
            ),
        )
    return materialized


def _generate_images(
    root: RootConfig,
    precision: PrecisionPolicy,
    targets: list[CheckpointTarget],
    prompts: list[str],
    *,
    output_dir: Path,
    sampling: dict[str, Any],
    device: Any,
    expected_model_identity: dict[str, Any],
    settings: sana_report.EvaluationSettings | None = None,
) -> list[GeneratedImage]:
    from vrl.models.families.registry import get_model_family_entry
    from vrl.utils.media import write_png

    settings = settings or sana_report.EvaluationSettings()
    if root.model is None:
        raise ValueError("SANA checkpoint evaluation requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    # Checkpoint compatibility stays bound to the configured Hub repo+commit.
    # The downloaded tree has a different, content-based identity, so retain it
    # separately only to prove the local snapshot did not change while loading.
    build = entry.resolve_model_build(root, device, precision=precision, for_rollout=True)
    bundle = entry.build_rollout(
        build,
        expected_identity=build.checkpoint_identity(),
    )
    model = bundle.model.eval()
    generated: list[GeneratedImage] = []
    checkpoint_read = False
    try:
        for target in targets:
            if target.path is None:
                if checkpoint_read:
                    raise RuntimeError("SANA baseline was scheduled after checkpoint loading")
            else:
                checkpoint = TrainingCheckpoint.load(target.path)
                if checkpoint.meta.get("uses_lora") is not False:
                    raise ValueError(
                        f"full-parameter checkpoint must declare uses_lora=false: {target.path}",
                    )
                match = re.fullmatch(r"checkpoint-(\d+)", target.path.name)
                if match is None:
                    raise ValueError(f"invalid numbered checkpoint path: {target.path}")
                source_epoch = int(match.group(1))
                if checkpoint.next_epoch != source_epoch:
                    raise ValueError(
                        f"checkpoint progress changed during evaluation: {target.path} "
                        f"next_epoch={checkpoint.next_epoch}, expected={source_epoch}",
                    )
                checkpoint.restore_model(
                    bundle=bundle,
                    family="sana",
                    expected_model_identity=expected_model_identity,
                )
                del checkpoint
                checkpoint_read = True
            for prompt_index, prompt in enumerate(prompts):
                group_seed = settings.group_seed(prompt_index)
                logger.info(
                    "Generating checkpoint=%s prompt=%d samples=%d group_seed=%d",
                    target.label,
                    prompt_index,
                    settings.samples_per_prompt,
                    group_seed,
                )
                # A fresh scheduler per prompt group makes each grid cell
                # independent of mutable scheduler state from earlier prompts.
                decoded = generate_prompt_images(
                    model,
                    scheduler=load_official_scheduler(build),
                    prompt=prompt,
                    seed=group_seed,
                    num_images=settings.samples_per_prompt,
                    device=device,
                    sampling=sampling,
                )
                for sample_index, image in enumerate(decoded):
                    path = (
                        output_dir
                        / "images"
                        / target.label
                        / f"prompt{prompt_index:04d}_sample{sample_index:02d}.png"
                    )
                    write_png(image, path)
                    generated.append(
                        GeneratedImage(
                            checkpoint_label=target.label,
                            # The persisted curve marks the baseline as epoch -1.
                            epoch=-1 if target.path is None else target.epoch,
                            prompt_index=prompt_index,
                            sample_index=sample_index,
                            group_seed=group_seed,
                            prompt=prompt,
                            path=path.resolve(),
                            image_sha256=sha256_file(path),
                        ),
                    )
                del decoded
    finally:
        del model, bundle
        release_cuda_memory()
    return generated


def _score_images(
    generated: list[GeneratedImage],
    reward_models: list[sana_report.RewardModelDefinition],
) -> list[dict[str, Any]]:
    from PIL import Image

    from vrl.rewards.inference import RewardInferenceArtifact
    from vrl.utils.config import import_from_path

    rows = [
        {
            "checkpoint_label": image.checkpoint_label,
            "epoch": image.epoch,
            "prompt_index": image.prompt_index,
            "sample_index": image.sample_index,
            "group_seed": image.group_seed,
            "prompt": image.prompt,
            "image_path": str(image.path),
            "image_sha256": image.image_sha256,
        }
        for image in generated
    ]
    for reward_model in reward_models:
        logger.info(
            "Loading %s reward for %d images",
            reward_model.name,
            len(generated),
        )
        model = import_from_path(reward_model.model_factory)(reward_model.model_config)
        try:
            for row, image in zip(rows, generated, strict=True):
                with Image.open(image.path) as source:
                    media = source.convert("RGB")
                artifact = RewardInferenceArtifact(
                    artifact_id=(
                        f"{image.checkpoint_label}-p{image.prompt_index:04d}"
                        f"-s{image.sample_index:02d}"
                    ),
                    sample_id=(
                        f"{image.checkpoint_label}-p{image.prompt_index:04d}"
                        f"-s{image.sample_index:02d}"
                    ),
                    path=str(image.path),
                    prompt=image.prompt,
                    metadata={
                        "checkpoint_label": image.checkpoint_label,
                        "group_seed": image.group_seed,
                    },
                    media=media,
                )
                scores = model(artifact)
                selected_keys = tuple(
                    key.strip() for key in reward_model.score_key.split("+") if key.strip()
                )
                row[f"r_{reward_model.name}"] = sum(float(scores[key]) for key in selected_keys)
        finally:
            del model
            release_cuda_memory()
    return rows


def _checkpoint_record(target: CheckpointTarget, run_dir: Path) -> dict[str, Any]:
    if target.path is None:
        return {
            "label": target.label,
            "epoch": -1,
            "source": "pinned_base_model_snapshot",
            "checkpoint_loaded": False,
        }
    return {
        "label": target.label,
        "epoch": target.epoch,
        "path": str(target.path.relative_to(run_dir)),
        "checkpoint_sha256": target.checkpoint_sha256,
        "checkpoint_bytes": (target.path / TRAINING_CHECKPOINT_NAME).stat().st_size,
    }


if __name__ == "__main__":
    main()
