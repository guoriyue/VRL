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
import csv
import gc
import hashlib
import json
import logging
import math
import os
import re
import statistics
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy, resolve_precision_policy
from vrl.config.schema import RootConfig, parse_config
from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
from vrl.scripts.eval._device import resolve_eval_device
from vrl.scripts.eval.sana_inference import (
    OFFICIAL_SAMPLING_PROTOCOL,
    SCHEDULER_PROTOCOL,
    generate_prompt_images,
    load_official_scheduler,
)
from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    is_complete_checkpoint,
    load_training_checkpoint,
    read_checkpoint_meta,
    restore_model_checkpoint,
    validate_checkpoint_meta_compatibility,
)
from vrl.trainers.data import load_prompt_manifest

logger = logging.getLogger(__name__)

# These are persisted protocol/file identities, not tunable experiment defaults.
# v4 records the family-neutral training entrypoint introduced by the physical
# taxonomy migration; sampling, reward, and optimization protocol values are unchanged.
REPORT_SCHEMA = "vrl.sana_aesthetic_checkpoint_eval/v4"
REPORT_SCHEMA_VERSION = 4
REPORT_RELATIVE_PATH = Path("sana_aesthetic_fullparam_native_fp16_eval/report.json")
SAMPLES_RELATIVE_PATH = Path("sana_aesthetic_fullparam_native_fp16_eval/samples.jsonl")
EVAL_BASE_SEED = 20260710
EVAL_SAMPLES_PER_PROMPT = 2
# Recovery checkpoints may be denser than the preregistered held-out curve.
# This is the fixed scientific comparison interval, not a training IO knob.
EVAL_CHECKPOINT_INTERVAL = 25
CANONICAL_CONFIG_NAME = "experiment/sana/online_grpo_aesthetic_fullparam_long"
# The entrypoint this protocol's runs were launched from, and its replacement.
# vrl/scripts/diffusion/ no longer exists; the module path survives only inside
# the resolved_config.yaml these historical run directories already wrote.
_RETIRED_ENTRYPOINT = "vrl.scripts.diffusion.train:train_diffusion_grpo"
_LIVE_ENTRYPOINT = "vrl.scripts.train:train_online"
# Semantic digest of the resolved canonical preset; recomputed against the merged
# config schema (see _normalize_run_config, which fails closed if the bundled
# preset drifts without updating this pin).
CANONICAL_PROTOCOL_SHA256 = "ec1baab564b3dc97b6f4a3287474965c9811dc3ed6bfdf9b8b8e41028ab32da6"
# Frozen protocol-asset identities. These hashes name two concrete datasets;
# they are not a duplicated prompt taxonomy or a user-facing config table.
TRAIN_MANIFEST_SHA256 = "86580c8136a4b6d9fc6bbcc6d8e8e172b15fca6b5c6c956cc770255d8011de56"
EVAL_MANIFEST_SHA256 = "10c70e8af2ae16b0d76eb9da0f53801485ab0a3bae83e605d310faa9b16bfcdd"
TRAIN_PROMPT_COUNT = 192
EVAL_PROMPT_COUNT = 64
AESTHETIC_ASSET_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
AESTHETIC_ASSET_BYTES = 3_714_759


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
class RewardModelDefinition:
    """One reward model used by the fixed evaluation."""

    name: str
    score_key: str
    model_factory: str
    model_config: dict[str, Any]
    # Provenance-only: immutable repo revisions/local asset digest for the report.
    provenance: dict[str, Any]


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
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"training run has no resolved config: {config_path}")
    training_metrics_path = run_dir / "metrics.csv"
    if not training_metrics_path.is_file():
        raise FileNotFoundError(f"training run has no metrics CSV: {training_metrics_path}")

    cfg = _normalize_run_config(load_config(config_path))
    _validate_training_metrics(training_metrics_path, cfg)
    training_manifest_path, eval_manifest_path, prompts = _resolve_protocol_manifests(cfg)
    if not prompts:
        raise ValueError(f"evaluation manifest has no prompts: {eval_manifest_path}")
    training_log = _validate_training_log_provenance(run_dir, cfg)

    targets = _discover_checkpoint_targets(run_dir, cfg)
    device = resolve_eval_device(args.device)
    sampling = _resolve_sampling()
    identity_root = parse_config(cfg)
    if identity_root.model is None:
        raise ValueError("SANA checkpoint evaluation requires model configuration")
    identity_precision = resolve_precision_policy(identity_root)
    from vrl.families.registry import get_model_family_entry

    identity_entry = get_model_family_entry(str(identity_root.model.family))
    identity_build = identity_entry.resolve_model_build(
        identity_root,
        device,
        precision=identity_precision,
        for_rollout=True,
    )
    model_identity = resolve_checkpoint_model_identity(identity_build)
    for target in targets:
        if target.path is not None:
            validate_checkpoint_meta_compatibility(
                read_checkpoint_meta(target.path),
                family="sana",
                expected_model_identity=model_identity,
                strict=True,
            )
    build_root = _materialize_model_snapshot(cfg)
    build_precision = resolve_precision_policy(build_root)
    reward_models = _materialize_reward_model_snapshots(
        _build_reward_model_definitions(cfg, generation_device=str(device)),
    )
    eval_dir = run_dir / REPORT_RELATIVE_PATH.parent
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
    sample_path = run_dir / SAMPLES_RELATIVE_PATH
    _write_jsonl_atomic(sample_path, sample_scores, base_dir=run_dir)
    metrics = _summarize_scores(sample_scores)
    if not metrics:
        raise RuntimeError("SANA checkpoint evaluation produced no summary rows")
    # The supervisor may append its final shutdown line while a long evaluation
    # is running. Reparse at publication so the report binds the final log bytes.
    training_log = _validate_training_log_provenance(run_dir, cfg)

    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "provenance": {
            "run": {
                "path": str(run_dir),
                "training_metrics_path": training_metrics_path.name,
                "training_metrics_sha256": _sha256(training_metrics_path),
            },
            "training_log": training_log,
            "resolved_config": {
                "path": config_path.name,
                "sha256": _sha256(config_path),
                "canonical_protocol": CANONICAL_CONFIG_NAME,
                "canonical_protocol_sha256": CANONICAL_PROTOCOL_SHA256,
            },
            "model": {
                "family": str(cfg.model.family),
                "repo": str(cfg.model.path),
                "revision": str(cfg.model.revision),
            },
            "training_manifest": {
                "path": str(training_manifest_path),
                "sha256": _sha256(training_manifest_path),
                "prompt_count": len(load_prompt_manifest(training_manifest_path)),
            },
            "eval_manifest": {
                "path": str(eval_manifest_path),
                "sha256": _sha256(eval_manifest_path),
                "prompt_count": len(prompts),
            },
            "seed_grid": {
                "base_seed": EVAL_BASE_SEED,
                "samples_per_prompt": EVAL_SAMPLES_PER_PROMPT,
                "formula": "base_seed + prompt_index * samples_per_prompt",
                "sample_stream": "one batched torch.Generator stream per prompt group",
            },
            "evaluation_curve": {
                "checkpoint_interval": EVAL_CHECKPOINT_INTERVAL,
            },
            "sampling": sampling,
            "scheduler_protocol": dict(SCHEDULER_PROTOCOL),
            "execution": {"generation_device": str(device)},
            "rewards": [_reward_model_record(reward_model) for reward_model in reward_models],
            "checkpoints": [_checkpoint_record(target, run_dir) for target in targets],
            "samples": {
                "path": str(SAMPLES_RELATIVE_PATH),
                "sha256": _sha256(sample_path),
                "count": len(sample_scores),
            },
        },
        "metrics": metrics,
    }
    report_path = run_dir / REPORT_RELATIVE_PATH
    _write_json_atomic(report_path, report)
    load_report_metrics(run_dir)
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


def load_report_metrics(run_dir: str | Path) -> list[dict[str, float]]:
    """Load and validate the canonical standalone report for one training run.

    This is the producer/consumer protocol boundary used by the verdict. It
    rejects empty reports and any config, manifest, checkpoint, sample-manifest,
    seed-grid, or reward provenance that no longer matches the evaluated run.
    """

    root = Path(run_dir).expanduser().resolve()
    report_path = root / REPORT_RELATIVE_PATH
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != REPORT_SCHEMA
        or raw.get("schema_version") != REPORT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported SANA evaluation report schema in {report_path}: "
            f"schema={raw.get('schema') if isinstance(raw, dict) else type(raw).__name__}, "
            f"version={raw.get('schema_version') if isinstance(raw, dict) else None}",
        )
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"SANA evaluation report has no provenance object: {report_path}")
    sample_rows = _validate_report_provenance(root, provenance)

    raw_metrics = raw.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError(f"SANA evaluation report has no metric rows: {report_path}")
    required = {
        "epoch",
        "eval_reward_stderr",
        "r_aesthetic",
        "r_pickscore",
        "sample_count",
    }
    rows: list[dict[str, float]] = []
    for index, raw_row in enumerate(raw_metrics):
        if not isinstance(raw_row, dict):
            raise TypeError(f"SANA evaluation metric row {index} must be an object")
        missing = required - set(raw_row)
        if missing:
            raise ValueError(
                f"SANA evaluation metric row {index} missing fields: {sorted(missing)}",
            )
        row = {key: float(raw_row[key]) for key in required}
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"SANA evaluation metric row {index} contains non-finite values")
        rows.append(row)

    expected_metrics = _summarize_scores(sample_rows)
    if len(expected_metrics) != len(raw_metrics):
        raise ValueError(
            "SANA evaluation summary row count does not match its scored samples: "
            f"{len(raw_metrics)} != {len(expected_metrics)}",
        )
    for index, (raw_row, expected_row) in enumerate(
        zip(raw_metrics, expected_metrics, strict=True),
    ):
        if raw_row.get("checkpoint_label") != expected_row["checkpoint_label"]:
            raise ValueError(
                f"SANA evaluation summary row {index} has the wrong checkpoint label",
            )
        for key in required:
            if not math.isclose(
                float(raw_row[key]),
                float(expected_row[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"SANA evaluation summary row {index} field {key!r} "
                    "does not match its scored samples",
                )

    checkpoint_records = provenance["checkpoints"]
    expected_epochs = [int(record["epoch"]) for record in checkpoint_records]
    actual_epochs = [int(row["epoch"]) for row in rows]
    if actual_epochs != expected_epochs:
        raise ValueError(
            "SANA evaluation metric/checkpoint epochs disagree: "
            f"metrics={actual_epochs}, checkpoints={expected_epochs}",
        )
    expected_samples = int(provenance["eval_manifest"]["prompt_count"]) * int(
        provenance["seed_grid"]["samples_per_prompt"],
    )
    if any(int(row["sample_count"]) != expected_samples for row in rows):
        raise ValueError(
            "SANA evaluation metric sample_count does not match the fixed prompt/seed grid",
        )
    return rows


def _section(config: Any, *path: str) -> dict[str, Any] | None:
    """The nested mapping at ``path``, or None when any hop is absent/not a dict."""

    for key in path:
        if not isinstance(config, dict):
            return None
        config = config.get(key)
    return config if isinstance(config, dict) else None


def _drop_default_key(
    section: dict[str, Any] | None,
    key: str,
    *,
    default: Any,
    resolve: Callable[[Any], Any] | None = None,
) -> None:
    """Erase a key whose value means the same thing as leaving it unwritten.

    ``resolve`` is for keys whose public spelling is wider than their meaning
    (``kl_reward_coef: 0.0`` and an absent key both resolve to 0.0); an invalid
    value is left in place so the caller still reports it as real drift.
    """

    if section is None or key not in section:
        return
    value = section[key]
    if resolve is not None:
        try:
            value = resolve(value)
        except ValueError:
            return
    if value == default:
        section.pop(key)


def _erase_meaningless_spelling(
    actual: dict[str, Any],
    canonical: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Erase spelling differences that carry no meaning; leave real drift visible.

    Two rules, not one case per key:

    1. A key whose value equals its default says the same thing as an absent
       key, so it is dropped from BOTH configs. Applying it symmetrically is
       what keeps this a rule instead of a list of which-side-is-older patches:
       new preset keys that default to their current value stop needing an
       entry here at all.
    2. The retired ``vrl.scripts.diffusion.train`` entrypoint took its precision
       protocol from the family registry rather than YAML, so only a run from
       that entrypoint is granted the injection. The live entrypoint must spell
       precision out, and ``PrecisionConfig`` has no defaults to fall back on.
    """

    from dataclasses import fields as dataclass_fields

    from vrl.config.algorithm import resolve_kl_reward_coef
    from vrl.config.schema import RolloutWorkerSection
    from vrl.trajectory import (
        TrajectoryStoragePolicy,
        trajectory_storage_policy_from_cfg,
    )

    def storage_policy(value: Any) -> Any:
        """Resolve a storage block, refusing one that carries unknown keys.

        ``trajectory_storage_policy_from_cfg`` ignores extra keys, so resolving
        directly would let an unrecognized knob ride along inside an otherwise
        default block. Rejecting keeps it visible as real drift.
        """

        policy_fields = {item.name for item in dataclass_fields(TrajectoryStoragePolicy)}
        if isinstance(value, dict) and set(value) != policy_fields:
            raise ValueError(f"unexpected trajectory_storage keys: {sorted(set(value))}")
        return trajectory_storage_policy_from_cfg(value)

    trainer = actual.get("trainer")
    uses_retired_entrypoint = (
        isinstance(trainer, dict)
        and trainer.get("entrypoint") == _RETIRED_ENTRYPOINT
        and _section(canonical, "trainer") is not None
        and canonical["trainer"].get("entrypoint") == _LIVE_ENTRYPOINT
    )

    # Rule 1. Defaults come from their live owner, never a copied literal, so a
    # changed default cannot silently keep validating stale runs.
    default_equivalent: list[tuple[tuple[str, ...], str, Any, Any]] = [
        (
            ("algorithm",),
            "kl_reward_coef",
            resolve_kl_reward_coef(None),
            resolve_kl_reward_coef,
        ),
        (("data", "preprocessing"), "target_text", "none", None),
        (
            ("rollout",),
            "trajectory_storage",
            trajectory_storage_policy_from_cfg(None),
            storage_policy,
        ),
        (("reward", "kwargs", "aesthetic"), "device", None, None),
        *(
            (("distributed", "rollout"), name, default, None)
            for name, default in RolloutWorkerSection().model_dump().items()
        ),
    ]
    for path, key, default, resolve in default_equivalent:
        for side in (actual, canonical):
            _drop_default_key(_section(side, *path), key, default=default, resolve=resolve)

    # Rule 2. Same erasure, but only a retired-entrypoint SANA fp16 run earns it.
    precision = _section(actual, "precision")
    canonical_precision = _section(canonical, "precision")
    if (
        uses_retired_entrypoint
        and precision is not None
        and canonical_precision is not None
        and _section(actual, "model") is not None
        and actual["model"].get("family") == "sana"
        and _section(canonical, "model") is not None
        and canonical["model"].get("family") == "sana"
    ):
        stages = [
            (_section(precision, stage), _section(canonical_precision, stage))
            for stage in ("training", "rollout")
        ]
        if all(
            stage is not None
            and canonical_stage is not None
            and stage.get("dtype") == canonical_stage.get("dtype") == "fp16"
            for stage, canonical_stage in stages
        ):
            _drop_default_key(canonical_precision, "float32_precision", default="ieee")
            for _, canonical_stage in stages:
                _drop_default_key(canonical_stage, "outer_autocast", default=False)

    if uses_retired_entrypoint:
        trainer["entrypoint"] = _LIVE_ENTRYPOINT

    # Both are returned because both were edited: rule 1 erases defaults on
    # whichever side spells them, so the caller must compare the pair it gets
    # back rather than its own copy of the canonical config.
    return actual, canonical


def _normalize_run_config(cfg: DictConfig) -> DictConfig:
    """Require the exact pre-registered full-parameter long-run config."""

    actual = OmegaConf.to_container(cfg, resolve=True)
    expected = OmegaConf.to_container(load_config(CANONICAL_CONFIG_NAME), resolve=True)
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise TypeError("SANA evaluation configs must resolve to mappings")
    canonical_digest = _semantic_digest(expected)
    if canonical_digest != CANONICAL_PROTOCOL_SHA256:
        raise ValueError(
            "bundled SANA aesthetic preset changed without a protocol schema update: "
            f"{canonical_digest} != {CANONICAL_PROTOCOL_SHA256}",
        )
    # Compare on copies: the erasure edits both sides, and the canonical config
    # is also this function's return value once the run is accepted.
    normalized_actual, normalized_expected = _erase_meaningless_spelling(
        deepcopy(actual),
        deepcopy(expected),
    )
    if normalized_actual != normalized_expected:
        mismatch = _first_config_difference(normalized_actual, normalized_expected)
        raise ValueError(
            "resolved config does not match the registered SANA full-parameter protocol"
            + (f": {mismatch}" if mismatch else ""),
        )
    # A run that passes IS the registered protocol, so hand downstream the
    # canonical spelling rather than whichever historical shape wrote it. That
    # keeps every consumer reading fully-spelled keys (reward kwargs, precision)
    # instead of the omissions the comparison just proved irrelevant.
    normalized = OmegaConf.create(expected)
    OmegaConf.resolve(normalized)
    assert isinstance(normalized, DictConfig)
    parse_config(normalized)
    return normalized


def _first_config_difference(actual: Any, expected: Any, path: str = "") -> str:
    """Describe the first canonical-protocol mismatch without dumping the config."""

    if isinstance(actual, dict) and isinstance(expected, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            return (
                f"{path or '<root>'} keys differ: missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(actual):
            child = f"{path}.{key}" if path else str(key)
            mismatch = _first_config_difference(actual[key], expected[key], child)
            if mismatch:
                return mismatch
        return ""
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path} length differs: {len(actual)} != {len(expected)}"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            mismatch = _first_config_difference(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
            if mismatch:
                return mismatch
        return ""
    return "" if actual == expected else f"{path}: {actual!r} != {expected!r}"


def _semantic_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_protocol_manifests(cfg: DictConfig) -> tuple[Path, Path, list[str]]:
    training_path = (
        Path(
            str(OmegaConf.select(cfg, "data.manifest", default="") or ""),
        )
        .expanduser()
        .resolve()
    )
    eval_path = (
        Path(
            str(OmegaConf.select(cfg, "data.eval_manifest", default="") or ""),
        )
        .expanduser()
        .resolve()
    )
    for label, path in (("training", training_path), ("evaluation", eval_path)):
        if not path.is_file():
            raise FileNotFoundError(f"SANA {label} manifest does not exist: {path}")
    if _sha256(training_path) != TRAIN_MANIFEST_SHA256:
        raise ValueError(
            f"SANA training manifest does not match the registered asset: {training_path}"
        )
    if _sha256(eval_path) != EVAL_MANIFEST_SHA256:
        raise ValueError(
            f"SANA evaluation manifest does not match the registered asset: {eval_path}"
        )

    training_prompts = [example.prompt for example in load_prompt_manifest(training_path)]
    eval_prompts = [example.prompt for example in load_prompt_manifest(eval_path)]
    if len(training_prompts) != TRAIN_PROMPT_COUNT:
        raise ValueError(
            f"SANA training manifest has {len(training_prompts)} prompts, "
            f"expected {TRAIN_PROMPT_COUNT}",
        )
    if len(eval_prompts) != EVAL_PROMPT_COUNT or len(set(eval_prompts)) != EVAL_PROMPT_COUNT:
        raise ValueError("SANA evaluation manifest must contain exactly 64 unique prompts")
    overlap = set(training_prompts) & set(eval_prompts)
    if overlap:
        raise ValueError(
            f"SANA training/evaluation manifests overlap on {len(overlap)} prompts",
        )
    return training_path, eval_path, eval_prompts


def _discover_checkpoint_targets(run_dir: Path, cfg: DictConfig) -> list[CheckpointTarget]:
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
    save_freq = int(OmegaConf.select(cfg, "trainer.save_freq", default=0))
    if save_freq <= 0:
        raise ValueError("SANA aesthetic curve requires trainer.save_freq > 0")
    total_epochs = int(OmegaConf.select(cfg, "trainer.total_epochs", default=0))
    if (
        total_epochs <= 0
        or total_epochs % save_freq != 0
        or total_epochs % EVAL_CHECKPOINT_INTERVAL != 0
        or EVAL_CHECKPOINT_INTERVAL % save_freq != 0
    ):
        raise ValueError(
            "SANA aesthetic curve requires total_epochs divisible by both the "
            "checkpoint and evaluation intervals, with save_freq dividing the "
            "evaluation interval",
        )
    eval_numbered = [
        (epoch, path) for epoch, path in numbered if epoch % EVAL_CHECKPOINT_INTERVAL == 0
    ]
    epochs = [epoch for epoch, _ in eval_numbered]
    expected = list(
        range(EVAL_CHECKPOINT_INTERVAL, total_epochs + 1, EVAL_CHECKPOINT_INTERVAL),
    )
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
                checkpoint_sha256=_sha256(checkpoint_file),
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


def _validate_training_metrics(path: Path, cfg: DictConfig) -> None:
    """Fail before model load when the registered 300-update run is incomplete."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "epoch" not in reader.fieldnames:
            raise ValueError(f"training metrics CSV has no epoch column: {path}")
        rows = list(reader)
    total_epochs = int(OmegaConf.select(cfg, "trainer.total_epochs", default=0))
    actual_epochs = [int(float(row["epoch"])) for row in rows]
    expected_epochs = list(range(total_epochs))
    if actual_epochs != expected_epochs:
        raise ValueError(
            "training metrics are incomplete or out of order: "
            f"found {len(actual_epochs)} rows ending at "
            f"{actual_epochs[-1] if actual_epochs else None}, expected epochs 0..{total_epochs - 1}",
        )


def _validate_training_log_provenance(run_dir: Path, cfg: DictConfig) -> dict[str, Any]:
    """Bind the supervisor log to revisions pinned in the resolved config."""

    path = run_dir / "supervisor.log"
    if not path.is_file():
        raise FileNotFoundError(
            "SANA run has no supervisor.log launch evidence",
        )
    reward_kwargs = OmegaConf.to_container(
        OmegaConf.select(cfg, "reward.kwargs", default={}),
        resolve=True,
    )
    reward_kwargs = dict(reward_kwargs or {})
    aesthetic = dict(reward_kwargs.get("aesthetic") or {})
    pickscore = dict(reward_kwargs.get("pickscore") or {})
    configured = {
        str(cfg.model.path): str(cfg.model.revision or ""),
        str(aesthetic.get("model_name") or ""): str(
            aesthetic.get("model_revision") or "",
        ),
        str(pickscore.get("processor_name") or ""): str(
            pickscore.get("processor_revision") or "",
        ),
        str(pickscore.get("model_name") or ""): str(
            pickscore.get("model_revision") or "",
        ),
    }
    if "" in configured or any(not revision for revision in configured.values()):
        raise ValueError("SANA training provenance requires pinned model and reward revisions")
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "configured_model_revisions": configured,
    }


def _resolve_sampling() -> dict[str, Any]:
    """Return the quality protocol, intentionally independent of training SDE."""

    return dict(OFFICIAL_SAMPLING_PROTOCOL)


def _build_reward_model_definitions(
    cfg: DictConfig,
    *,
    generation_device: str,
) -> list[RewardModelDefinition]:
    raw_kwargs = OmegaConf.to_container(
        OmegaConf.select(cfg, "reward.kwargs", default={}),
        resolve=True,
    )
    reward_kwargs = dict(raw_kwargs or {})
    reward_models: list[RewardModelDefinition] = []
    for name, score_key, model_factory, identity_keys in (
        (
            "aesthetic",
            "aesthetic",
            "vrl.rewards.models.aesthetic:aesthetic_reward_model",
            ("model_name", "model_revision"),
        ),
        (
            "pickscore",
            "pickscore",
            "vrl.rewards.models.pickscore:pickscore_reward_model",
            ("processor_name", "processor_revision", "model_name", "model_revision"),
        ),
    ):
        model_config = dict(reward_kwargs.get(name) or {})
        missing_identity = [
            key for key in identity_keys if not str(model_config.get(key) or "").strip()
        ]
        if missing_identity:
            raise ValueError(
                f"SANA evaluation requires explicit {name} reward identity fields in the "
                f"resolved config: {missing_identity}",
            )
        # Reward presets use null to mean "inherit the resource-resolved device".
        # A standalone evaluator has no distributed resource resolver, so inherit
        # its explicit execution device here and persist the effective value.
        if not str(model_config.get("device") or "").strip():
            model_config["device"] = generation_device
        if not str(model_config.get("dtype") or "").strip():
            model_config["dtype"] = "float32"
        if name == "aesthetic":
            provenance = {
                "model": {
                    "repo": str(model_config["model_name"]),
                    "revision": str(model_config["model_revision"]),
                },
                "mlp_asset": _aesthetic_asset_record(),
            }
        else:
            provenance = {
                "processor": {
                    "repo": str(model_config["processor_name"]),
                    "revision": str(model_config["processor_revision"]),
                },
                "model": {
                    "repo": str(model_config["model_name"]),
                    "revision": str(model_config["model_revision"]),
                },
            }
        reward_models.append(
            RewardModelDefinition(
                name=name,
                score_key=score_key,
                model_factory=model_factory,
                model_config=model_config,
                provenance=provenance,
            ),
        )
    return reward_models


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
    reward_models: list[RewardModelDefinition],
) -> list[RewardModelDefinition]:
    """Resolve immutable reward snapshots without exposing local paths in reports."""

    from huggingface_hub import snapshot_download

    materialized: list[RewardModelDefinition] = []
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
            RewardModelDefinition(
                name=reward_model.name,
                score_key=reward_model.score_key,
                model_factory=reward_model.model_factory,
                model_config=model_config,
                provenance=reward_model.provenance,
            ),
        )
    return materialized


def _reward_model_record(reward_model: RewardModelDefinition) -> dict[str, Any]:
    return {
        "name": reward_model.name,
        "score_key": reward_model.score_key,
        "model_factory": reward_model.model_factory,
        "device": str(reward_model.model_config["device"]),
        "dtype": str(reward_model.model_config["dtype"]),
        "identity": reward_model.provenance,
    }


def _aesthetic_asset_record() -> dict[str, Any]:
    from importlib import resources

    asset = resources.files("vrl.rewards.assets").joinpath(
        "sac+logos+ava1-l14-linearMSE.pth",
    )
    digest = hashlib.sha256()
    size = 0
    with asset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    sha256 = digest.hexdigest()
    if sha256 != AESTHETIC_ASSET_SHA256 or size != AESTHETIC_ASSET_BYTES:
        raise ValueError(
            "packaged aesthetic MLP asset does not match the registered protocol: "
            f"sha256={sha256}, bytes={size}",
        )
    return {
        "package": "vrl.rewards.assets",
        "name": "sac+logos+ava1-l14-linearMSE.pth",
        "sha256": sha256,
        "bytes": size,
    }


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
    from vrl.families.registry import get_model_family_entry
    from vrl.utils.media import write_png

    if root.model is None:
        raise ValueError("SANA checkpoint evaluation requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    build = entry.resolve_model_build(
        root,
        device,
        precision=precision,
        for_rollout=True,
    )
    # Checkpoint compatibility stays bound to the configured Hub repo+commit.
    # The downloaded tree has a different, content-based identity, so retain it
    # separately only to prove the local snapshot did not change while loading.
    materialized_model_identity = resolve_checkpoint_model_identity(build)
    bundle = entry.build_rollout(build)
    loaded_materialized_identity = resolve_checkpoint_model_identity(build)
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
                group_seed = _group_seed(prompt_index)
                logger.info(
                    "Generating checkpoint=%s prompt=%d samples=%d group_seed=%d",
                    target.label,
                    prompt_index,
                    EVAL_SAMPLES_PER_PROMPT,
                    group_seed,
                )
                # A fresh scheduler per prompt group makes each grid cell
                # independent of mutable scheduler state from earlier prompts.
                decoded = generate_prompt_images(
                    model,
                    scheduler=load_official_scheduler(build),
                    prompt=prompt,
                    seed=group_seed,
                    num_images=EVAL_SAMPLES_PER_PROMPT,
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
                            image_sha256=_sha256(path),
                        ),
                    )
                del decoded
    finally:
        del model, bundle
        _release_cuda()
    return generated


def _score_images(
    generated: list[GeneratedImage],
    reward_models: list[RewardModelDefinition],
) -> list[dict[str, Any]]:
    from PIL import Image

    from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
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
                    path=str(image.path),
                    media_type="image",
                    prompt=image.prompt,
                    sample_id=f"p{image.prompt_index}-s{image.sample_index}",
                    metadata={
                        "checkpoint_label": image.checkpoint_label,
                        "group_seed": image.group_seed,
                    },
                    media=media,
                )
                request = RewardInferenceRequest(
                    request_id=(f"sana-aesthetic-eval-{artifact.artifact_id}-{reward_model.name}"),
                    artifacts=(artifact,),
                    reward_name=reward_model.name,
                    score_key=reward_model.score_key,
                )
                scores = model(artifact=artifact, request=request)
                row[f"r_{reward_model.name}"] = request.select_score(scores)
        finally:
            del model
            _release_cuda()
    return rows


def _summarize_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["epoch"]), str(row["checkpoint_label"])), []).append(row)
    metrics: list[dict[str, Any]] = []
    for (epoch, label), group in sorted(grouped.items()):
        aesthetic = [float(row["r_aesthetic"]) for row in group]
        pickscore = [float(row["r_pickscore"]) for row in group]
        if not aesthetic:
            continue
        aesthetic_std = statistics.pstdev(aesthetic) if len(aesthetic) > 1 else 0.0
        metrics.append(
            {
                "checkpoint_label": label,
                "epoch": epoch,
                "sample_count": len(group),
                "eval_reward_stderr": aesthetic_std / math.sqrt(len(aesthetic)),
                "r_aesthetic": statistics.fmean(aesthetic),
                "r_pickscore": statistics.fmean(pickscore),
            },
        )
    return metrics


def _group_seed(prompt_index: int) -> int:
    return EVAL_BASE_SEED + prompt_index * EVAL_SAMPLES_PER_PROMPT


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


def _validate_report_provenance(
    run_dir: Path,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    for key in (
        "resolved_config",
        "run",
        "training_log",
        "model",
        "training_manifest",
        "eval_manifest",
        "seed_grid",
        "evaluation_curve",
        "sampling",
        "scheduler_protocol",
        "execution",
        "rewards",
        "checkpoints",
        "samples",
    ):
        if key not in provenance:
            raise ValueError(f"SANA evaluation report provenance missing {key!r}")
    for key in (
        "run",
        "training_log",
        "resolved_config",
        "model",
        "training_manifest",
        "eval_manifest",
        "seed_grid",
        "evaluation_curve",
        "sampling",
        "scheduler_protocol",
        "execution",
        "samples",
    ):
        if not isinstance(provenance[key], dict):
            raise TypeError(f"SANA evaluation report provenance {key!r} must be an object")

    run_record = provenance["run"]
    if Path(str(run_record.get("path", ""))).expanduser().resolve() != run_dir:
        raise ValueError("SANA evaluation report belongs to a different training run")
    training_metrics_path = run_dir / str(run_record.get("training_metrics_path", ""))
    _require_matching_file(
        training_metrics_path,
        {"sha256": run_record.get("training_metrics_sha256")},
        label="training metrics",
    )

    config_record = provenance["resolved_config"]
    config_path = run_dir / str(config_record.get("path", ""))
    _require_matching_file(config_path, config_record, label="resolved config")
    if config_record.get("canonical_protocol") != CANONICAL_CONFIG_NAME:
        raise ValueError("SANA evaluation report names the wrong canonical config protocol")
    if config_record.get("canonical_protocol_sha256") != CANONICAL_PROTOCOL_SHA256:
        raise ValueError("SANA evaluation report names the wrong canonical protocol digest")
    cfg = _normalize_run_config(load_config(config_path))
    _validate_training_metrics(training_metrics_path, cfg)
    if provenance["training_log"] != _validate_training_log_provenance(run_dir, cfg):
        raise ValueError("SANA evaluation training-log provenance changed")
    expected_model = {
        "family": str(cfg.model.family),
        "repo": str(cfg.model.path),
        "revision": str(cfg.model.revision),
    }
    if provenance["model"] != expected_model:
        raise ValueError("SANA evaluation model provenance disagrees with resolved_config.yaml")

    training_manifest_path, eval_manifest_path, prompts = _resolve_protocol_manifests(cfg)
    for label, path, expected_count in (
        (
            "training_manifest",
            training_manifest_path,
            len(load_prompt_manifest(training_manifest_path)),
        ),
        ("eval_manifest", eval_manifest_path, len(prompts)),
    ):
        record = provenance[label]
        if Path(str(record.get("path", ""))).expanduser().resolve() != path:
            raise ValueError(
                f"SANA {label} provenance disagrees with resolved_config.yaml",
            )
        _require_matching_file(path, record, label=label.replace("_", " "))
        if int(record.get("prompt_count", -1)) != expected_count:
            raise ValueError(f"SANA {label} prompt count changed")

    if provenance["sampling"] != _resolve_sampling():
        raise ValueError("SANA evaluation sampling provenance changed")
    if provenance["scheduler_protocol"] != SCHEDULER_PROTOCOL:
        raise ValueError("SANA evaluation scheduler protocol changed")

    seed_grid = provenance["seed_grid"]
    expected_seed = {
        "base_seed": EVAL_BASE_SEED,
        "samples_per_prompt": EVAL_SAMPLES_PER_PROMPT,
        "formula": "base_seed + prompt_index * samples_per_prompt",
        "sample_stream": "one batched torch.Generator stream per prompt group",
    }
    if seed_grid != expected_seed:
        raise ValueError(
            f"SANA evaluation report seed protocol changed: {seed_grid!r} != {expected_seed!r}",
        )

    if provenance["evaluation_curve"] != {
        "checkpoint_interval": EVAL_CHECKPOINT_INTERVAL,
    }:
        raise ValueError("SANA evaluation checkpoint interval changed")

    execution = provenance["execution"]
    if not isinstance(execution, dict) or not str(execution.get("generation_device", "")):
        raise ValueError("SANA evaluation report execution provenance is incomplete")
    expected_rewards = [
        _reward_model_record(reward_model)
        for reward_model in _build_reward_model_definitions(
            cfg,
            generation_device=str(execution["generation_device"]),
        )
    ]
    if provenance["rewards"] != expected_rewards:
        raise ValueError("SANA evaluation reward provenance disagrees with resolved_config.yaml")

    checkpoints = provenance["checkpoints"]
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or not all(isinstance(record, dict) for record in checkpoints)
    ):
        raise ValueError("SANA evaluation report has no checkpoint provenance")
    expected_checkpoints = [
        _checkpoint_record(target, run_dir)
        for target in _discover_checkpoint_targets(run_dir, cfg)
    ]
    if checkpoints != expected_checkpoints:
        raise ValueError(
            "SANA evaluation checkpoint provenance no longer matches the training run",
        )

    sample_record = provenance["samples"]
    sample_path = run_dir / str(sample_record.get("path", ""))
    _require_matching_file(sample_path, sample_record, label="evaluation samples")
    sample_rows: list[dict[str, Any]] = []
    with sample_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"evaluation sample row {line_number} must be an object")
            sample_rows.append(row)
    if not sample_rows or len(sample_rows) != int(sample_record.get("count", -1)):
        raise ValueError(
            f"SANA evaluation sample manifest count changed: {len(sample_rows)} != "
            f"{sample_record.get('count')!r}",
        )
    expected_cells = {
        (record["label"], int(record["epoch"]), prompt_index, sample_index)
        for record in checkpoints
        for prompt_index in range(len(prompts))
        for sample_index in range(EVAL_SAMPLES_PER_PROMPT)
    }
    actual_cells: set[tuple[str, int, int, int]] = set()
    for index, row in enumerate(sample_rows):
        required_sample = {
            "checkpoint_label",
            "epoch",
            "prompt_index",
            "sample_index",
            "group_seed",
            "prompt",
            "image_path",
            "image_sha256",
            "r_aesthetic",
            "r_pickscore",
        }
        missing = required_sample - set(row)
        if missing:
            raise ValueError(f"evaluation sample row {index} missing fields: {sorted(missing)}")
        cell = (
            str(row["checkpoint_label"]),
            int(row["epoch"]),
            int(row["prompt_index"]),
            int(row["sample_index"]),
        )
        actual_cells.add(cell)
        prompt_index = cell[2]
        if not (0 <= prompt_index < len(prompts)) or str(row["prompt"]) != prompts[prompt_index]:
            raise ValueError(f"evaluation sample row {index} has the wrong prompt identity")
        if int(row["group_seed"]) != _group_seed(prompt_index):
            raise ValueError(f"evaluation sample row {index} has the wrong fixed-grid seed")
        image_path = run_dir / str(row["image_path"])
        _require_matching_file(
            image_path,
            {"sha256": row["image_sha256"]},
            label=f"evaluation image row {index}",
        )
        if not all(math.isfinite(float(row[key])) for key in ("r_aesthetic", "r_pickscore")):
            raise ValueError(f"evaluation sample row {index} contains non-finite reward scores")
    if actual_cells != expected_cells or len(actual_cells) != len(sample_rows):
        raise ValueError("evaluation sample rows do not exactly cover the fixed checkpoint grid")
    return sample_rows


def _require_matching_file(path: Path, record: dict[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} provenance target does not exist: {path}")
    expected = str(record.get("sha256") or record.get("checkpoint_sha256") or "")
    if not expected or _sha256(path) != expected:
        raise ValueError(f"{label} provenance hash changed: {path}")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]], *, base_dir: Path) -> None:
    if not rows:
        raise ValueError("refusing to write an empty SANA evaluation sample manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                portable = dict(row)
                portable["image_path"] = str(Path(str(row["image_path"])).relative_to(base_dir))
                handle.write(json.dumps(portable, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
