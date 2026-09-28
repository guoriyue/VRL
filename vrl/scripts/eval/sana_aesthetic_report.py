"""Own the persisted SANA aesthetic-curve report protocol.

The checkpoint evaluator produces images and scores. This module owns the
report and sample paths, publication helpers, and
the metric-row reader consumed by the curve verdict. Keeping both sides of the
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

from vrl.config.schema import RootConfig, parse_config
from vrl.trainers.data.prompts import load_prompt_dataset_index
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import write_json, write_jsonl

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
    """Load the metric rows of one standalone evaluation report."""

    report_path = Path(run_dir).expanduser().resolve() / REPORT_RELATIVE_PATH
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != REPORT_SCHEMA:
        raise ValueError(
            f"unsupported SANA evaluation report schema in {report_path}: "
            f"schema={raw.get('schema') if isinstance(raw, dict) else type(raw).__name__}",
        )
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
    """Publish the report beside its sample manifest."""

    if not metrics:
        raise ValueError("refusing to write a SANA evaluation report without metrics")
    path = run_dir / REPORT_RELATIVE_PATH
    payload = {
        "schema": REPORT_SCHEMA,
        "provenance": provenance,
        "metrics": metrics,
    }
    write_json(path, payload)
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
