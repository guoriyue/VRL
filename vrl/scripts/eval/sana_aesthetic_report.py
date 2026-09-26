"""Own the persisted SANA aesthetic-curve report protocol.

The checkpoint evaluator produces images and scores. This module owns the
report and sample paths, publication helpers, and
the fail-closed reader consumed by the curve verdict. Keeping both sides of the
persisted contract here prevents the producer CLI from becoming an accidental
schema owner.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from vrl.config.loading import load_config
from vrl.config.schema import RootConfig, parse_config
from vrl.scripts.eval.sana_inference import SANA_EVAL_SAMPLING_CONFIG, SANA_EVAL_SCHEDULER_CONFIG
from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    is_complete_checkpoint,
    read_checkpoint_meta,
)
from vrl.trainers.data.prompts import load_prompt_dataset_index
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import read_jsonl, write_json, write_jsonl

# Report format and file names are shared by the producer and reader.
REPORT_SCHEMA = "vrl.sana_aesthetic_checkpoint_eval/v6"
REPORT_RELATIVE_PATH = Path("aesthetic_eval/report.json")
SAMPLES_RELATIVE_PATH = REPORT_RELATIVE_PATH.with_name("samples.jsonl")


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Configurable evaluation grid, persisted with each report."""

    base_seed: int = 0
    samples_per_prompt: int = 2
    checkpoint_interval: int = 25

    def __post_init__(self) -> None:
        for name in ("base_seed", "samples_per_prompt", "checkpoint_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name == "base_seed" else 1):
                raise ValueError(f"invalid evaluation {name}: {value}")

    def group_seed(self, prompt_index: int) -> int:
        return self.base_seed + prompt_index * self.samples_per_prompt

    def seed_grid_record(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "samples_per_prompt": self.samples_per_prompt,
            "formula": "base_seed + prompt_index * samples_per_prompt",
            "sample_stream": "one batched torch.Generator stream per prompt group",
        }


@dataclass(frozen=True, slots=True)
class RewardModelDefinition:
    """One reward implementation and its immutable report provenance."""

    name: str
    score_key: str
    model_factory: str
    model_config: dict[str, Any]
    # Provenance-only: immutable repo revisions/local asset digest for the report.
    provenance: dict[str, Any]

    def to_report_record(self) -> dict[str, Any]:
        """Project one runtime reward definition into persisted provenance."""

        return {
            "name": self.name,
            "score_key": self.score_key,
            "model_factory": self.model_factory,
            "device": str(self.model_config["device"]),
            "dtype": str(self.model_config["dtype"]),
            "identity": self.provenance,
        }


def validate_run_config(cfg: DictConfig) -> RootConfig:
    """Validate the run's own config, without comparing it to a preset."""

    root = parse_config(cfg)
    if root.model is None or root.model.family != "sana":
        raise ValueError("SANA evaluation requires model.family=sana")
    return root


def resolve_protocol_manifests(root: RootConfig) -> tuple[Path, Path, list[str]]:
    """Resolve the configured train/evaluation split and reject leakage."""

    data = root.data
    training_path = (
        Path(str((data.manifest if data is not None else None) or "")).expanduser().resolve()
    )
    eval_path = (
        Path(str((data.eval_manifest if data is not None else None) or "")).expanduser().resolve()
    )
    for label, path in (("training", training_path), ("evaluation", eval_path)):
        if not path.is_file():
            raise FileNotFoundError(f"SANA {label} manifest does not exist: {path}")
    training_prompts = [example.prompt for example in load_prompt_dataset_index(training_path)]
    eval_prompts = [example.prompt for example in load_prompt_dataset_index(eval_path)]
    if not training_prompts or not eval_prompts:
        raise ValueError("SANA training/evaluation manifests must be nonempty")
    if len(set(eval_prompts)) != len(eval_prompts):
        raise ValueError("SANA evaluation manifest must contain unique prompts")
    overlap = set(training_prompts) & set(eval_prompts)
    if overlap:
        raise ValueError(
            f"SANA training/evaluation manifests overlap on {len(overlap)} prompts",
        )
    return training_path, eval_path, eval_prompts


def validate_training_metrics(path: Path, root: RootConfig) -> None:
    """Fail when the training CSV does not cover every configured update."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "epoch" not in reader.fieldnames:
            raise ValueError(f"training metrics CSV has no epoch column: {path}")
        rows = list(reader)
    total_epochs = int((root.trainer.total_epochs if root.trainer is not None else None) or 0)
    actual_epochs = [int(float(row["epoch"])) for row in rows]
    expected_epochs = list(range(total_epochs))
    if actual_epochs != expected_epochs:
        raise ValueError(
            "training metrics are incomplete or out of order: "
            f"found {len(actual_epochs)} rows ending at "
            f"{actual_epochs[-1] if actual_epochs else None}, expected epochs 0..{total_epochs - 1}",
        )


def require_training_log_provenance(run_dir: Path, root: RootConfig) -> dict[str, Any]:
    """Bind the supervisor log to revisions pinned in the resolved config."""

    path = run_dir / "supervisor.log"
    if not path.is_file():
        raise FileNotFoundError("SANA run has no supervisor.log launch evidence")
    reward_kwargs = dict(root.reward.kwargs) if root.reward is not None else {}
    aesthetic = dict(reward_kwargs.get("aesthetic") or {})
    pickscore = dict(reward_kwargs.get("pickscore") or {})
    if root.model is None:
        raise ValueError("SANA training provenance requires model configuration")
    configured = {
        str(root.model.path): str(root.model.revision or ""),
        str(aesthetic.get("model_name") or ""): str(aesthetic.get("model_revision") or ""),
        str(pickscore.get("processor_name") or ""): str(
            pickscore.get("processor_revision") or "",
        ),
        str(pickscore.get("model_name") or ""): str(pickscore.get("model_revision") or ""),
    }
    if "" in configured or any(not revision for revision in configured.values()):
        raise ValueError("SANA training provenance requires pinned model and reward revisions")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "configured_model_revisions": configured,
    }


def build_reward_model_definitions(
    root: RootConfig,
    *,
    generation_device: str,
) -> list[RewardModelDefinition]:
    """Resolve the fixed reward implementations and persisted identities."""

    reward_kwargs = dict(root.reward.kwargs) if root.reward is not None else {}
    reward_models: list[RewardModelDefinition] = []
    for name, score_key, model_factory, identity_keys in (
        (
            "aesthetic",
            "aesthetic",
            "vrl.rewards.models.aesthetic:AestheticRewardModel",
            ("model_name", "model_revision"),
        ),
        (
            "pickscore",
            "pickscore",
            "vrl.rewards.models.pickscore:PickScoreRewardModel",
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
        # Standalone evaluation has no distributed device resolver.
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


def summarize_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive the report summary from its scored sample rows."""

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["epoch"]), str(row["checkpoint_label"])), []).append(row)
    metrics: list[dict[str, Any]] = []
    for (epoch, label), group in sorted(grouped.items()):
        aesthetic = [float(row["r_aesthetic"]) for row in group]
        pickscore = [float(row["r_pickscore"]) for row in group]
        aesthetic_std = statistics.pstdev(aesthetic)
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


def checkpoint_curve_epochs(
    root: RootConfig,
    settings: EvaluationSettings,
) -> list[int]:
    """Return the configured curve epochs after validating save cadence."""

    trainer = root.trainer
    save_freq = int((trainer.save_freq if trainer is not None else None) or 0)
    if save_freq <= 0:
        raise ValueError("SANA aesthetic curve requires trainer.save_freq > 0")
    total_epochs = int((trainer.total_epochs if trainer is not None else None) or 0)
    if (
        total_epochs <= 0
        or total_epochs % save_freq != 0
        or total_epochs % settings.checkpoint_interval != 0
        or settings.checkpoint_interval % save_freq != 0
    ):
        raise ValueError(
            "SANA aesthetic curve requires total_epochs divisible by both the "
            "checkpoint and evaluation intervals, with save_freq dividing the "
            "evaluation interval",
        )
    return list(
        range(settings.checkpoint_interval, total_epochs + 1, settings.checkpoint_interval)
    )


def load_report_metrics(run_dir: str | Path) -> list[dict[str, float]]:
    """Load and fully validate one provenance-bound standalone evaluation report."""

    root = Path(run_dir).expanduser().resolve()
    report_path = root / REPORT_RELATIVE_PATH
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != REPORT_SCHEMA:
        raise ValueError(
            f"unsupported SANA evaluation report schema in {report_path}: "
            f"schema={raw.get('schema') if isinstance(raw, dict) else type(raw).__name__}",
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

    expected_metrics = summarize_scores(sample_rows)
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


def write_sample_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    base_dir: Path,
) -> None:
    """Atomically persist scored samples with run-relative image paths."""

    if not rows:
        raise ValueError("refusing to write an empty SANA evaluation sample manifest")
    portable_rows = []
    for row in rows:
        portable = dict(row)
        portable["image_path"] = str(Path(str(row["image_path"])).relative_to(base_dir))
        portable_rows.append(portable)
    write_jsonl(path, portable_rows)


def publish_report(
    run_dir: Path,
    *,
    provenance: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> Path:
    """Publish and re-read a report through the same persisted contract."""

    if not metrics:
        raise ValueError("refusing to write a SANA evaluation report without metrics")
    path = run_dir / REPORT_RELATIVE_PATH
    payload = {
        "schema": REPORT_SCHEMA,
        "provenance": provenance,
        "metrics": metrics,
    }
    write_json(path, payload)
    load_report_metrics(run_dir)
    return path


def _aesthetic_asset_record() -> dict[str, Any]:
    from importlib import resources

    asset = resources.files("vrl.rewards.assets").joinpath(
        "aesthetic_predictor_v2_5.pth",
    )
    with resources.as_file(asset) as asset_path:
        sha256 = sha256_file(asset_path)
        size = asset_path.stat().st_size
    return {
        "package": "vrl.rewards.assets",
        "name": "aesthetic_predictor_v2_5.pth",
        "sha256": sha256,
        "bytes": size,
    }


def _validate_report_provenance(
    run_dir: Path,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    required = (
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
    )
    for key in required:
        if key not in provenance:
            raise ValueError(f"SANA evaluation report provenance missing {key!r}")
    mapping_sections = (*required[:-2], "samples")
    for key in mapping_sections:
        if key in {"rewards", "checkpoints"}:
            continue
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
    root = validate_run_config(load_config(config_path))
    validate_training_metrics(training_metrics_path, root)
    if provenance["training_log"] != require_training_log_provenance(run_dir, root):
        raise ValueError("SANA evaluation training-log provenance changed")
    if root.model is None:
        raise ValueError("SANA evaluation report requires model configuration")
    expected_model = {
        "family": str(root.model.family),
        "repo": str(root.model.path),
        "revision": str(root.model.revision),
    }
    if provenance["model"] != expected_model:
        raise ValueError("SANA evaluation model provenance disagrees with resolved_config.yaml")

    training_manifest_path, eval_manifest_path, prompts = resolve_protocol_manifests(root)
    for label, path, expected_count in (
        (
            "training_manifest",
            training_manifest_path,
            len(load_prompt_dataset_index(training_manifest_path)),
        ),
        ("eval_manifest", eval_manifest_path, len(prompts)),
    ):
        record = provenance[label]
        if Path(str(record.get("path", ""))).expanduser().resolve() != path:
            raise ValueError(f"SANA {label} provenance disagrees with resolved_config.yaml")
        _require_matching_file(path, record, label=label.replace("_", " "))
        if int(record.get("prompt_count", -1)) != expected_count:
            raise ValueError(f"SANA {label} prompt count changed")

    if provenance["sampling"] != SANA_EVAL_SAMPLING_CONFIG:
        raise ValueError("SANA evaluation sampling provenance changed")
    if provenance["scheduler_protocol"] != SANA_EVAL_SCHEDULER_CONFIG:
        raise ValueError("SANA evaluation scheduler protocol changed")
    settings = EvaluationSettings(
        base_seed=provenance["seed_grid"].get("base_seed"),
        samples_per_prompt=provenance["seed_grid"].get("samples_per_prompt"),
        checkpoint_interval=provenance["evaluation_curve"].get("checkpoint_interval"),
    )
    if provenance["seed_grid"] != settings.seed_grid_record():
        raise ValueError("SANA evaluation seed protocol is invalid")
    if provenance["evaluation_curve"] != {"checkpoint_interval": settings.checkpoint_interval}:
        raise ValueError("SANA evaluation checkpoint interval is invalid")

    execution = provenance["execution"]
    if not str(execution.get("generation_device", "")):
        raise ValueError("SANA evaluation report execution provenance is incomplete")
    expected_rewards = [
        reward_model.to_report_record()
        for reward_model in build_reward_model_definitions(
            root,
            generation_device=str(execution["generation_device"]),
        )
    ]
    if provenance["rewards"] != expected_rewards:
        raise ValueError("SANA evaluation reward provenance disagrees with resolved_config.yaml")

    checkpoints = provenance["checkpoints"]
    _validate_checkpoint_records(run_dir, root, checkpoints, settings)

    sample_record = provenance["samples"]
    sample_path = run_dir / str(sample_record.get("path", ""))
    _require_matching_file(sample_path, sample_record, label="evaluation samples")
    sample_rows = read_jsonl(sample_path)
    if not sample_rows or len(sample_rows) != int(sample_record.get("count", -1)):
        raise ValueError(
            f"SANA evaluation sample manifest count changed: {len(sample_rows)} != "
            f"{sample_record.get('count')!r}",
        )
    expected_cells = {
        (record["label"], int(record["epoch"]), prompt_index, sample_index)
        for record in checkpoints
        for prompt_index in range(len(prompts))
        for sample_index in range(settings.samples_per_prompt)
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
        if int(row["group_seed"]) != settings.group_seed(prompt_index):
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


def _validate_checkpoint_records(
    run_dir: Path,
    root: RootConfig,
    checkpoints: Any,
    settings: EvaluationSettings,
) -> None:
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or not all(isinstance(record, dict) for record in checkpoints)
    ):
        raise ValueError("SANA evaluation report has no checkpoint provenance")
    baseline = {
        "label": "baseline",
        "epoch": -1,
        "source": "pinned_base_model_snapshot",
        "checkpoint_loaded": False,
    }
    if checkpoints[0] != baseline:
        raise ValueError(
            "SANA evaluation checkpoint provenance no longer matches the training run",
        )
    expected_epochs = checkpoint_curve_epochs(root, settings)
    if [int(record.get("epoch", -1)) for record in checkpoints[1:]] != expected_epochs:
        raise ValueError(
            "SANA evaluation checkpoint provenance no longer matches the training run",
        )
    for epoch, record in zip(expected_epochs, checkpoints[1:], strict=True):
        label = f"checkpoint-{epoch}"
        checkpoint_dir = run_dir / label
        if (
            record.get("label") != label
            or record.get("path") != label
            or not is_complete_checkpoint(checkpoint_dir)
        ):
            raise ValueError(
                "SANA evaluation checkpoint provenance no longer matches the training run",
            )
        meta = read_checkpoint_meta(checkpoint_dir)
        if str(meta.get("family", "")) != "sana" or int(meta.get("completed_epoch", -1)) != epoch:
            raise ValueError(
                "SANA evaluation checkpoint provenance no longer matches the training run",
            )
        checkpoint_path = checkpoint_dir / TRAINING_CHECKPOINT_NAME
        if int(record.get("checkpoint_bytes", -1)) != checkpoint_path.stat().st_size:
            raise ValueError(
                "SANA evaluation checkpoint provenance no longer matches the training run",
            )
        _require_matching_file(checkpoint_path, record, label="checkpoint provenance")


def _require_matching_file(path: Path, record: dict[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} provenance target does not exist: {path}")
    expected = str(record.get("sha256") or record.get("checkpoint_sha256") or "")
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"{label} provenance hash changed: {path}")
