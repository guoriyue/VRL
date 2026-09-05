"""Evaluate a SANA training run on the registered aesthetic curve protocol.

The training process owns checkpoints only. This standalone process loads the
run's resolved config and complete ``checkpoint-N`` directories, evaluates the
base model and every saved checkpoint on one fixed DrawBench prompt/seed grid,
then writes a provenance-bound report for ``sana_aesthetic_curve_verdict``.

Only ``--run-dir`` selects experiment inputs. Config, manifest, checkpoints,
sampling, seed, and rewards are deliberately not CLI overrides: allowing any of
them to drift would make the resulting curve incomparable with the registered
training run.
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

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy, resolve_precision_policy
from vrl.config.schema import RootConfig, parse_config
from vrl.models import checkpoint_identity
from vrl.scripts.eval import sana_aesthetic_report as sana_report
from vrl.scripts.eval._device import resolve_eval_device
from vrl.scripts.eval.sana_inference import (
    SCHEDULER_PROTOCOL,
    generate_prompt_images,
    load_official_scheduler,
)
from vrl.trainers.checkpointing import (
    RESOLVED_CONFIG_NAME,
    TRAINING_CHECKPOINT_NAME,
    is_complete_checkpoint,
    load_training_checkpoint,
    read_checkpoint_meta,
    restore_model_checkpoint,
    validate_checkpoint_meta_compatibility,
)
from vrl.trainers.data import load_prompt_manifest
from vrl.utils.artifacts import sha256_file
from vrl.utils.cuda_memory import release_cuda_memory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckpointTarget:
    """One curve point and its immutable checkpoint provenance."""

    label: str
    epoch: int
    path: Path | None
    # Provenance-only: emitted into the report and revalidated by its reader.
    checkpoint_sha256: str | None
    checkpoint_bytes: int | None


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
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / RESOLVED_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"training run has no resolved config: {config_path}")
    training_metrics_path = run_dir / "metrics.csv"
    if not training_metrics_path.is_file():
        raise FileNotFoundError(f"training run has no metrics CSV: {training_metrics_path}")

    cfg = sana_report.normalize_run_config(load_config(config_path))
    root = parse_config(cfg)
    sana_report.validate_training_metrics(training_metrics_path, root)
    training_manifest_path, eval_manifest_path, prompts = sana_report.resolve_protocol_manifests(
        root,
    )
    if not prompts:
        raise ValueError(f"evaluation manifest has no prompts: {eval_manifest_path}")
    training_log = sana_report.validate_training_log_provenance(run_dir, root)

    targets = _discover_checkpoint_targets(run_dir, root)
    device = resolve_eval_device(args.device)
    sampling = sana_report.resolve_sampling()
    if root.model is None:
        raise ValueError("SANA checkpoint evaluation requires model configuration")
    identity_precision = resolve_precision_policy(root.precision)
    from vrl.models.families.registry import get_model_family_entry
    from vrl.run import resolve_model

    identity_entry = get_model_family_entry(str(root.model.family))
    model_identity = resolve_model(
        identity_entry,
        root,
        device,
        precision=identity_precision,
        for_rollout=True,
    ).identity
    for target in targets:
        if target.path is not None:
            validate_checkpoint_meta_compatibility(
                read_checkpoint_meta(target.path),
                family="sana",
                expected_model_identity=model_identity,
                strict=True,
            )
    build_root = _materialize_model_snapshot(cfg)
    build_precision = resolve_precision_policy(build_root.precision)
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
    )
    sample_scores = _score_images(generated, reward_models)
    sample_path = run_dir / sana_report.SAMPLES_RELATIVE_PATH
    sana_report.write_sample_manifest(sample_path, sample_scores, base_dir=run_dir)
    metrics = sana_report.summarize_scores(sample_scores)
    if not metrics:
        raise RuntimeError("SANA checkpoint evaluation produced no summary rows")
    # The supervisor may append its final shutdown line while a long evaluation
    # is running. Reparse at publication so the report binds the final log bytes.
    training_log = sana_report.validate_training_log_provenance(run_dir, root)

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
            "canonical_protocol": sana_report.CANONICAL_CONFIG_NAME,
            "canonical_protocol_sha256": sana_report.CANONICAL_PROTOCOL_SHA256,
        },
        "model": {
            "family": str(cfg.model.family),
            "repo": str(cfg.model.path),
            "revision": str(cfg.model.revision),
        },
        "training_manifest": {
            "path": str(training_manifest_path),
            "sha256": sha256_file(training_manifest_path),
            "prompt_count": len(load_prompt_manifest(training_manifest_path)),
        },
        "eval_manifest": {
            "path": str(eval_manifest_path),
            "sha256": sha256_file(eval_manifest_path),
            "prompt_count": len(prompts),
        },
        "seed_grid": sana_report.seed_grid_record(),
        "evaluation_curve": sana_report.evaluation_curve_record(),
        "sampling": sampling,
        "scheduler_protocol": dict(SCHEDULER_PROTOCOL),
        "execution": {"generation_device": str(device)},
        "rewards": [
            sana_report.reward_model_record(reward_model) for reward_model in reward_models
        ],
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


def _discover_checkpoint_targets(run_dir: Path, root: RootConfig) -> list[CheckpointTarget]:
    numbered: list[tuple[int, Path]] = []
    for candidate in run_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
        if match is None:
            continue
        if not is_complete_checkpoint(candidate):
            raise ValueError(f"incomplete checkpoint in SANA curve: {candidate}")
        epoch = int(match.group(1))
        meta = read_checkpoint_meta(candidate)
        if str(meta.get("family", "")) != "sana":
            raise ValueError(f"checkpoint family is not sana: {candidate}")
        if int(meta.get("completed_epoch", -1)) != epoch:
            raise ValueError(
                f"checkpoint epoch disagrees with directory name: {candidate} "
                f"completed_epoch={meta.get('completed_epoch')!r}",
            )
        numbered.append((epoch, candidate.resolve()))
    if not numbered:
        raise ValueError(f"training run has no complete checkpoint-N directories: {run_dir}")
    numbered.sort()
    expected = sana_report.checkpoint_curve_epochs(root)
    eval_numbered = [
        (epoch, path)
        for epoch, path in numbered
        if epoch % sana_report.EVAL_CHECKPOINT_INTERVAL == 0
    ]
    epochs = [epoch for epoch, _ in eval_numbered]
    if epochs != expected:
        raise ValueError(
            f"checkpoint curve is incomplete or has gaps: found={epochs}, expected={expected}",
        )

    checkpoint_targets: list[CheckpointTarget] = []
    for epoch, path in eval_numbered:
        checkpoint_file = path / TRAINING_CHECKPOINT_NAME
        checkpoint_targets.append(
            CheckpointTarget(
                label=path.name,
                epoch=epoch,
                path=path,
                checkpoint_sha256=sha256_file(checkpoint_file),
                checkpoint_bytes=checkpoint_file.stat().st_size,
            ),
        )
    baseline = CheckpointTarget(
        label="baseline",
        epoch=-1,
        path=None,
        checkpoint_sha256=None,
        checkpoint_bytes=None,
    )
    return [baseline, *checkpoint_targets]


def _materialize_model_snapshot(cfg: DictConfig) -> RootConfig:
    """Pin SANA architecture/frozen modules through the official Hub API."""

    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=str(cfg.model.path),
        revision=str(cfg.model.revision),
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

    from huggingface_hub import snapshot_download

    materialized: list[sana_report.RewardModelDefinition] = []
    for reward_model in reward_models:
        model_config = dict(reward_model.model_config)
        if reward_model.name == "aesthetic":
            identity = reward_model.provenance["model"]
            model_config["model_name"] = snapshot_download(
                repo_id=str(identity["repo"]),
                revision=str(identity["revision"]),
            )
        elif reward_model.name == "pickscore":
            processor = reward_model.provenance["processor"]
            model = reward_model.provenance["model"]
            model_config["processor_name"] = snapshot_download(
                repo_id=str(processor["repo"]),
                revision=str(processor["revision"]),
            )
            model_config["model_name"] = snapshot_download(
                repo_id=str(model["repo"]),
                revision=str(model["revision"]),
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
) -> list[GeneratedImage]:
    from vrl.models.families.registry import get_model_family_entry
    from vrl.run import resolve_model
    from vrl.utils.media import write_png

    if root.model is None:
        raise ValueError("SANA checkpoint evaluation requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    # Checkpoint compatibility stays bound to the configured Hub repo+commit.
    # The downloaded tree has a different, content-based identity, so retain it
    # separately only to prove the local snapshot did not change while loading.
    resolved = resolve_model(
        entry,
        root,
        device,
        precision=precision,
        for_rollout=True,
    )
    build = resolved.build
    materialized_model_identity = resolved.identity
    bundle = entry.build_rollout(build)
    # The registered mismatch wording below predates run.materialize and is
    # pinned by this protocol's tests, so the recheck stays inline.
    loaded_materialized_identity = checkpoint_identity.resolve_checkpoint_model_identity(build)
    if loaded_materialized_identity != materialized_model_identity:
        raise RuntimeError(
            "materialized SANA model source changed during runtime construction: "
            f"before={materialized_model_identity!r}, after={loaded_materialized_identity!r}",
        )
    model = bundle.model.eval()
    generated: list[GeneratedImage] = []
    checkpoint_read = False
    try:
        for target in targets:
            if target.epoch == -1:
                if target.path is not None:
                    raise ValueError(
                        "full-parameter baseline must come from the pinned base model"
                    )
                if checkpoint_read:
                    raise RuntimeError("SANA baseline was scheduled after checkpoint loading")
            else:
                if target.path is None:
                    raise ValueError(f"checkpoint target has no path: {target.label}")
                checkpoint = load_training_checkpoint(target.path)
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
                restore_model_checkpoint(
                    checkpoint,
                    bundle=bundle,
                    family="sana",
                    expected_model_identity=expected_model_identity,
                    strict=True,
                )
                del checkpoint
                checkpoint_read = True
            for prompt_index, prompt in enumerate(prompts):
                group_seed = sana_report.group_seed(prompt_index)
                logger.info(
                    "Generating checkpoint=%s prompt=%d samples=%d group_seed=%d",
                    target.label,
                    prompt_index,
                    sana_report.EVAL_SAMPLES_PER_PROMPT,
                    group_seed,
                )
                # A fresh scheduler per prompt group makes each grid cell
                # independent of mutable scheduler state from earlier prompts.
                decoded = generate_prompt_images(
                    model,
                    scheduler=load_official_scheduler(build),
                    prompt=prompt,
                    seed=group_seed,
                    num_images=sana_report.EVAL_SAMPLES_PER_PROMPT,
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
                            epoch=target.epoch,
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
    if target.epoch == -1:
        return {
            "label": target.label,
            "epoch": target.epoch,
            "source": "pinned_base_model_snapshot",
            "checkpoint_loaded": False,
        }
    if target.path is None or target.checkpoint_sha256 is None or target.checkpoint_bytes is None:
        raise ValueError(f"checkpoint target provenance is incomplete: {target.label}")
    return {
        "label": target.label,
        "epoch": target.epoch,
        "path": str(target.path.relative_to(run_dir)),
        "checkpoint_sha256": target.checkpoint_sha256,
        "checkpoint_bytes": target.checkpoint_bytes,
    }


if __name__ == "__main__":
    main()
