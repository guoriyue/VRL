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
import copy
import csv
import gc
import hashlib
import json
import logging
import math
import os
import re
import statistics
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.config.loading import load_config
from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    is_complete_checkpoint,
    load_trainable_state,
    load_training_checkpoint,
    read_checkpoint_meta,
)
from vrl.trainers.data import load_prompt_manifest

logger = logging.getLogger(__name__)

# These are persisted protocol/file identities, not tunable experiment defaults.
REPORT_SCHEMA = "vrl.sana_aesthetic_checkpoint_eval/v1"
REPORT_SCHEMA_VERSION = 1
REPORT_RELATIVE_PATH = Path("sana_aesthetic_eval/report.json")
SAMPLES_RELATIVE_PATH = Path("sana_aesthetic_eval/samples.jsonl")
EVAL_BASE_SEED = 20260710
EVAL_SAMPLES_PER_PROMPT = 2
CANONICAL_CONFIG_NAME = "experiment/diffusion/sana/online_grpo_aesthetic"
CANONICAL_PROTOCOL_SHA256 = "c1d3814771ce756e9ce0e412f09e6486e72edf26bb66e67c09f8556d1995beea"
# Frozen protocol-asset identities. These hashes name two concrete datasets;
# they are not a duplicated prompt taxonomy or a user-facing config table.
TRAIN_MANIFEST_SHA256 = "86580c8136a4b6d9fc6bbcc6d8e8e172b15fca6b5c6c956cc770255d8011de56"
EVAL_MANIFEST_SHA256 = "10c70e8af2ae16b0d76eb9da0f53801485ab0a3bae83e605d310faa9b16bfcdd"
TRAIN_PROMPT_COUNT = 192
EVAL_PROMPT_COUNT = 64
SANA_MODEL_REVISION = "d1b54936033cd7d45410ecadd692c5c502a19a38"
AESTHETIC_MODEL_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
PICKSCORE_PROCESSOR_REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
PICKSCORE_MODEL_REVISION = "a4e4367c6dfa7288a00c550414478f865b875800"
AESTHETIC_ASSET_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
AESTHETIC_ASSET_BYTES = 3_714_759


@dataclass(frozen=True, slots=True)
class CheckpointTarget:
    """One curve point and its immutable checkpoint provenance."""

    label: str
    epoch: int
    path: Path
    # Provenance-only: emitted into the report and revalidated by its reader.
    checkpoint_sha256: str
    checkpoint_bytes: int


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
    device = _resolve_device(args.device)
    sampling = _resolve_sampling(cfg)
    build_cfg = _materialize_model_snapshot(cfg)
    reward_models = _materialize_reward_model_snapshots(
        _build_reward_model_definitions(cfg, generation_device=str(device)),
    )
    eval_dir = run_dir / REPORT_RELATIVE_PATH.parent
    eval_dir.mkdir(parents=True, exist_ok=True)

    generated = _generate_images(
        build_cfg,
        targets,
        prompts,
        output_dir=eval_dir,
        sampling=sampling,
        device=device,
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
                "revision": SANA_MODEL_REVISION,
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
            "sampling": sampling,
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


def _normalize_run_config(cfg: DictConfig) -> DictConfig:
    """Return a canonical build copy while preserving archived run provenance.

    The active run predates removal of ``model.dtype`` and inline ``trainer.eval``.
    Those two legacy fields are accepted only at their exact SANA protocol values
    and removed from this in-memory copy. PickScore's later CPU placement default
    is execution topology rather than scientific protocol, so comparison ignores
    that one key while the returned config preserves the run's effective choice.
    Every other resolved field must equal the canonical bundled preset.
    """

    canonical = load_config(CANONICAL_CONFIG_NAME)
    actual = OmegaConf.to_container(cfg, resolve=True)
    expected = OmegaConf.to_container(canonical, resolve=True)
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise TypeError("SANA evaluation configs must resolve to mappings")
    actual = copy.deepcopy(actual)
    expected = copy.deepcopy(expected)

    model = actual.get("model")
    if not isinstance(model, dict):
        raise ValueError("SANA evaluation config has no model mapping")
    legacy_dtype = model.pop("dtype", None)
    if legacy_dtype is not None:
        from vrl.models.dtypes import dtype_to_precision_token

        try:
            legacy_token = dtype_to_precision_token(legacy_dtype)
        except ValueError as error:
            raise ValueError(
                f"legacy SANA model.dtype must be fp16, got {legacy_dtype!r}",
            ) from error
        if legacy_token != "fp16":
            raise ValueError(f"legacy SANA model.dtype must be fp16, got {legacy_dtype!r}")

    legacy_precision = actual.get("precision")
    if isinstance(legacy_precision, str):
        from vrl.models.dtypes import dtype_to_precision_token

        if dtype_to_precision_token(legacy_precision) != "bf16":
            raise ValueError(
                f"legacy SANA precision must be bf16, got {legacy_precision!r}",
            )
        # The precision cleanup replaced the legacy scalar with a typed training
        # dtype whose aligned rollout value is inherited. This build copy adopts
        # the equivalent canonical representation; the archived file/hash stays
        # untouched.
        actual["precision"] = copy.deepcopy(expected["precision"])

    trainer = actual.get("trainer")
    expected_trainer = expected.get("trainer")
    if not isinstance(trainer, dict) or not isinstance(expected_trainer, dict):
        raise ValueError("SANA evaluation config has no trainer mapping")
    legacy_eval = trainer.pop("eval", None)
    if legacy_eval is not None:
        eval_manifest = Path(str(expected["data"]["eval_manifest"])).expanduser().resolve()
        expected_legacy_eval = {
            "enabled": True,
            "freq": int(expected_trainer["save_freq"]),
            "samples_per_prompt": EVAL_SAMPLES_PER_PROMPT,
            "max_prompts": len(load_prompt_manifest(eval_manifest)),
            "seed": EVAL_BASE_SEED,
        }
        if legacy_eval != expected_legacy_eval:
            raise ValueError(
                "legacy trainer.eval does not match the registered SANA protocol: "
                f"{legacy_eval!r} != {expected_legacy_eval!r}",
            )

    # Device placement changed after the active run began (missing meant CUDA;
    # the current preset pins PickScore to CPU). Both execute the same float32
    # scorer and the effective device is persisted separately in report provenance.
    actual_for_comparison = copy.deepcopy(actual)
    expected_for_comparison = copy.deepcopy(expected)
    for candidate in (actual_for_comparison, expected_for_comparison):
        reward_kwargs = candidate.get("reward", {}).get("kwargs", {})
        pickscore = reward_kwargs.get("pickscore", {})
        if isinstance(pickscore, dict):
            pickscore.pop("device", None)
    canonical_digest = _semantic_digest(expected_for_comparison)
    if canonical_digest != CANONICAL_PROTOCOL_SHA256:
        raise ValueError(
            "bundled SANA aesthetic preset changed without a protocol schema update: "
            f"{canonical_digest} != {CANONICAL_PROTOCOL_SHA256}",
        )
    if actual_for_comparison != expected_for_comparison:
        mismatch = _first_config_difference(actual_for_comparison, expected_for_comparison)
        raise ValueError(
            "resolved config does not match the registered SANA aesthetic protocol"
            + (f": {mismatch}" if mismatch else ""),
        )

    normalized = OmegaConf.create(actual)
    OmegaConf.resolve(normalized)
    assert isinstance(normalized, DictConfig)
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
    if total_epochs <= 0 or total_epochs % save_freq != 0:
        raise ValueError(
            "SANA aesthetic curve requires total_epochs > 0 and divisible by save_freq",
        )
    epochs = [epoch for epoch, _ in numbered]
    expected = list(range(save_freq, total_epochs + 1, save_freq))
    if epochs != expected:
        raise ValueError(
            f"checkpoint curve is incomplete or has gaps: found={epochs}, expected={expected}",
        )

    checkpoint_targets: list[CheckpointTarget] = []
    for epoch, path in numbered:
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
    first = checkpoint_targets[0]
    baseline = CheckpointTarget(
        label="baseline",
        epoch=-1,
        path=first.path,
        checkpoint_sha256=first.checkpoint_sha256,
        checkpoint_bytes=first.checkpoint_bytes,
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
    """Bind this legacy run to the HF revisions observed in its startup log."""

    path = run_dir / "supervisor.log"
    if not path.is_file():
        raise FileNotFoundError(
            "SANA run has no supervisor.log revision evidence; legacy cache-hit runs "
            "without machine-readable model revisions cannot be evaluated trustworthily",
        )
    reward_kwargs = OmegaConf.to_container(
        OmegaConf.select(cfg, "reward.kwargs", default={}),
        resolve=True,
    )
    reward_kwargs = dict(reward_kwargs or {})
    aesthetic = dict(reward_kwargs.get("aesthetic") or {})
    pickscore = dict(reward_kwargs.get("pickscore") or {})
    expected = {
        str(cfg.model.path): SANA_MODEL_REVISION,
        str(aesthetic.get("model_name") or ""): AESTHETIC_MODEL_REVISION,
        str(pickscore.get("processor_name") or ""): PICKSCORE_PROCESSOR_REVISION,
        str(pickscore.get("model_name") or ""): PICKSCORE_MODEL_REVISION,
    }
    if "" in expected:
        raise ValueError("SANA training log provenance is missing a configured model identity")
    text = path.read_text(encoding="utf-8", errors="replace")
    resolved: dict[str, str] = {}
    for repo, revision in expected.items():
        pattern = re.compile(
            rf"/api/resolve-cache/models/{re.escape(repo)}/([0-9a-f]{{40}})/",
        )
        observed = set(pattern.findall(text))
        if observed != {revision}:
            raise ValueError(
                f"training log revisions for {repo!r} do not match the registered run: "
                f"observed={sorted(observed)}, expected={[revision]}",
            )
        resolved[repo] = revision
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "resolved_model_revisions": resolved,
    }


def _resolve_device(value: str) -> Any:
    import torch

    if value != "auto":
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device was requested ({value}), but CUDA is unavailable")
        return device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_sampling(cfg: DictConfig) -> dict[str, Any]:
    num_steps = int(OmegaConf.select(cfg, "sampling.num_steps", default=0))
    window_size = int(OmegaConf.select(cfg, "rollout.sde.window_size", default=0))
    if window_size != 0:
        raise ValueError(
            "SANA trustworthy-curve evaluation requires rollout.sde.window_size=0; "
            "a random SDE window is not a fixed evaluation protocol",
        )
    return {
        "width": int(OmegaConf.select(cfg, "sampling.width", default=0)),
        "height": int(OmegaConf.select(cfg, "sampling.height", default=0)),
        "num_steps": num_steps,
        "guidance_scale": float(
            OmegaConf.select(cfg, "sampling.guidance_scale", default=0.0),
        ),
        "max_sequence_length": int(
            OmegaConf.select(cfg, "sampling.max_sequence_length", default=0),
        ),
        "denoise_mode": str(OmegaConf.select(cfg, "rollout.denoise_mode", default="sde")),
        "noise_level": float(OmegaConf.select(cfg, "rollout.noise_level", default=1.0)),
        "sde_type": str(OmegaConf.select(cfg, "rollout.sde.type", default="flow_grpo")),
        "sde_window": [0, num_steps],
    }


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
            ("model_name",),
        ),
        (
            "pickscore",
            "pickscore",
            "vrl.rewards.models.pickscore:pickscore_reward_model",
            ("processor_name", "model_name"),
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
                    "revision": AESTHETIC_MODEL_REVISION,
                },
                "mlp_asset": _aesthetic_asset_record(),
            }
        else:
            provenance = {
                "processor": {
                    "repo": str(model_config["processor_name"]),
                    "revision": PICKSCORE_PROCESSOR_REVISION,
                },
                "model": {
                    "repo": str(model_config["model_name"]),
                    "revision": PICKSCORE_MODEL_REVISION,
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


def _materialize_model_snapshot(cfg: DictConfig) -> DictConfig:
    """Pin SANA architecture/frozen modules through the official Hub API."""

    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=str(cfg.model.path),
        revision=SANA_MODEL_REVISION,
    )
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    plain["model"]["path"] = local_path
    materialized = OmegaConf.create(plain)
    OmegaConf.resolve(materialized)
    assert isinstance(materialized, DictConfig)
    return materialized


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
    cfg: DictConfig,
    targets: list[CheckpointTarget],
    prompts: list[str],
    *,
    output_dir: Path,
    sampling: dict[str, Any],
    device: Any,
) -> list[GeneratedImage]:
    from vrl.models.diffusion.build import (
        build_family_runtime_bundle,
        resolve_family_model_build,
    )
    from vrl.utils.media import write_png

    build = resolve_family_model_build(cfg, device, for_rollout=True)
    bundle = build_family_runtime_bundle(build)
    model = bundle.model.eval()
    generated: list[GeneratedImage] = []
    loaded_checkpoint: Path | None = None
    try:
        for target in targets:
            if target.path != loaded_checkpoint:
                checkpoint = load_training_checkpoint(target.path)
                if str(checkpoint.payload.get("family", "")) != "sana":
                    raise ValueError(f"checkpoint family is not sana: {target.path}")
                match = re.fullmatch(r"checkpoint-(\d+)", target.path.name)
                if match is None:
                    raise ValueError(f"invalid numbered checkpoint path: {target.path}")
                source_epoch = int(match.group(1))
                if checkpoint.next_epoch != source_epoch:
                    raise ValueError(
                        f"checkpoint progress changed during evaluation: {target.path} "
                        f"next_epoch={checkpoint.next_epoch}, expected={source_epoch}",
                    )
                trainable_state = checkpoint.trainable_state
                load_trainable_state(bundle, trainable_state, strict=True)
                # The checkpoint payload also owns optimizer/RNG state that
                # evaluation never reads. Drop it before generating 128 images.
                del checkpoint, trainable_state
                loaded_checkpoint = target.path
            # The registered baseline is the adapter-off base policy, not an
            # assumption about PEFT's current LoRA initializer. Checkpoints use
            # their loaded adapters normally.
            adapter_context = model.disable_adapter() if target.epoch == -1 else nullcontext()
            with adapter_context:
                for prompt_index, prompt in enumerate(prompts):
                    group_seed = _group_seed(prompt_index)
                    logger.info(
                        "Generating checkpoint=%s prompt=%d samples=%d group_seed=%d",
                        target.label,
                        prompt_index,
                        EVAL_SAMPLES_PER_PROMPT,
                        group_seed,
                    )
                    decoded = _generate_prompt_images(
                        model,
                        prompt=prompt,
                        group_seed=group_seed,
                        sampling=sampling,
                        device=device,
                    )
                    if getattr(decoded, "shape", (0,))[0] != EVAL_SAMPLES_PER_PROMPT:
                        raise ValueError(
                            "SANA fixed eval generated the wrong prompt-group batch size: "
                            f"{getattr(decoded, 'shape', None)}",
                        )
                    for sample_index in range(EVAL_SAMPLES_PER_PROMPT):
                        path = (
                            output_dir
                            / "images"
                            / target.label
                            / f"prompt{prompt_index:04d}_sample{sample_index:02d}.png"
                        )
                        write_png(decoded[sample_index], path)
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


def _generate_prompt_images(
    model: Any,
    *,
    prompt: str,
    group_seed: int,
    sampling: dict[str, Any],
    device: Any,
) -> Any:
    import torch

    from vrl.generation.diffusion.layout import DiffusionRequestLayout, VideoGenerationRequest
    from vrl.math.diffusion.flow_matching import sde_step_with_logprob

    encoded = model.encode_prompt(
        prompt,
        None,
        max_sequence_length=int(sampling["max_sequence_length"]),
        guidance_scale=float(sampling["guidance_scale"]),
    )
    encoded = DiffusionRequestLayout().repeat_encoded_batch(
        encoded,
        EVAL_SAMPLES_PER_PROMPT,
    )
    request = VideoGenerationRequest(
        prompt=prompt,
        width=int(sampling["width"]),
        height=int(sampling["height"]),
        frame_count=1,
        num_steps=int(sampling["num_steps"]),
        guidance_scale=float(sampling["guidance_scale"]),
        seed=group_seed,
        extra={"max_sequence_length": int(sampling["max_sequence_length"])},
    )
    state = model.prepare_sampling(request, encoded)
    generator = torch.Generator(device=state.latents.device)
    generator.manual_seed(group_seed)
    autocast_dtype = getattr(model, "autocast_dtype", torch.float32)
    autocast = (
        torch.amp.autocast("cuda", dtype=autocast_dtype)
        if getattr(device, "type", None) == "cuda"
        and autocast_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast:
            for step_index, timestep in enumerate(state.timesteps):
                noise_pred = model.forward_step(state, step_index)["noise_pred"]
                if sampling["denoise_mode"] == "native":
                    state.latents = state.scheduler.step(
                        noise_pred.float(),
                        timestep,
                        state.latents.float(),
                        return_dict=False,
                    )[0]
                else:
                    state.latents = sde_step_with_logprob(
                        state.scheduler,
                        noise_pred.float(),
                        timestep.unsqueeze(0),
                        state.latents.float(),
                        generator=generator,
                        deterministic=False,
                        return_dt=False,
                        noise_level=float(sampling["noise_level"]),
                        sde_type=str(sampling["sde_type"]),
                        step_index=step_index,
                    ).prev_sample
        # SANA's DC-AE is an explicit fp32 fidelity boundary. The production
        # executor decodes after leaving transformer autocast; keep eval equal.
        return model.decode_latents(state.latents)


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
            "source_checkpoint_path": str(target.path.relative_to(run_dir)),
            "checkpoint_sha256": target.checkpoint_sha256,
            "checkpoint_bytes": target.checkpoint_bytes,
            "adapter": "disabled",
        }
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
        "sampling",
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
        "sampling",
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
        "revision": SANA_MODEL_REVISION,
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

    if provenance["sampling"] != _resolve_sampling(cfg):
        raise ValueError("SANA evaluation sampling provenance disagrees with resolved_config.yaml")

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
