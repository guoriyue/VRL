"""Own the persisted SANA aesthetic-curve report protocol.

The checkpoint evaluator produces images and scores. This module owns the
frozen experiment identity, report and sample paths, publication helpers, and
the fail-closed reader consumed by the curve verdict. Keeping both sides of the
persisted contract here prevents the producer CLI from becoming an accidental
schema owner.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.config.loading import load_config
from vrl.config.schema import parse_config
from vrl.scripts.eval.sana_inference import OFFICIAL_SAMPLING_PROTOCOL, SCHEDULER_PROTOCOL
from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    is_complete_checkpoint,
    read_checkpoint_meta,
)
from vrl.trainers.data import load_prompt_manifest
from vrl.utils.artifacts import sha256_file

# Persisted protocol and asset identities. These constants are real schema
# boundaries, not tunable experiment defaults or duplicated typed structures.
REPORT_SCHEMA = "vrl.sana_aesthetic_checkpoint_eval/v4"
REPORT_SCHEMA_VERSION = 4
REPORT_RELATIVE_PATH = Path("sana_aesthetic_fullparam_native_fp16_eval/report.json")
SAMPLES_RELATIVE_PATH = Path("sana_aesthetic_fullparam_native_fp16_eval/samples.jsonl")
EVAL_BASE_SEED = 20260710
EVAL_SAMPLES_PER_PROMPT = 2
# Recovery checkpoints may be denser than the preregistered held-out curve.
EVAL_CHECKPOINT_INTERVAL = 25
CANONICAL_CONFIG_NAME = "experiment/sana/online_grpo_aesthetic_fullparam_long"
# Historical run configs persist the retired path, so it remains protocol data.
_RETIRED_ENTRYPOINT = "vrl.scripts.diffusion.train:train_diffusion_grpo"
_LIVE_ENTRYPOINT = "vrl.scripts.train:train_online"
# Digest of the bundled canonical preset after the 2026-08 sharing-grammar
# simplification (allow_overlap retired, default-valued resource spellings
# dropped from the preset chain; sharing derives from device-set intersections).
CANONICAL_PROTOCOL_SHA256 = "d7bc7f2848d5a443e5156bf8ffcaf693190ac24419b745e0a07511a540ae6c5a"
TRAIN_MANIFEST_SHA256 = "86580c8136a4b6d9fc6bbcc6d8e8e172b15fca6b5c6c956cc770255d8011de56"
EVAL_MANIFEST_SHA256 = "10c70e8af2ae16b0d76eb9da0f53801485ab0a3bae83e605d310faa9b16bfcdd"
TRAIN_PROMPT_COUNT = 192
EVAL_PROMPT_COUNT = 64
AESTHETIC_ASSET_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
AESTHETIC_ASSET_BYTES = 3_714_759


@dataclass(frozen=True, slots=True)
class RewardModelDefinition:
    """One reward implementation and its immutable report provenance."""

    name: str
    score_key: str
    model_factory: str
    model_config: dict[str, Any]
    # Provenance-only: immutable repo revisions/local asset digest for the report.
    provenance: dict[str, Any]


def normalize_run_config(cfg: DictConfig) -> DictConfig:
    """Require the exact preregistered full-parameter long-run config."""

    actual = OmegaConf.to_container(cfg, resolve=True)
    expected = OmegaConf.to_container(load_config(CANONICAL_CONFIG_NAME), resolve=True)
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise TypeError("SANA evaluation configs must resolve to mappings")
    # The frozen v4 identity predates the parity gate's move out of debug.
    # Project only that spelling back for hashing: threshold changes and any
    # additional parity settings must still invalidate the registered protocol.
    # Runtime validation below keeps the live shape and its mandatory gate.
    registered_shape = deepcopy(expected)
    registered_trainer = _section(registered_shape, "trainer")
    parity = _section(registered_shape, "trainer", "replay_parity")
    if registered_trainer is not None and parity is not None:
        debug = _section(registered_shape, "trainer", "debug")
        if debug is not None and "max_abs_logprob_diff" in parity:
            if "max_abs_logprob_diff" in debug:
                raise ValueError("ambiguous SANA parity threshold in debug and replay_parity")
            debug["max_abs_logprob_diff"] = parity.pop("max_abs_logprob_diff")
            if not parity:
                registered_trainer.pop("replay_parity")
    canonical_digest = _semantic_digest(registered_shape)
    if canonical_digest != CANONICAL_PROTOCOL_SHA256:
        raise ValueError(
            "bundled SANA aesthetic preset changed without a protocol schema update: "
            f"{canonical_digest} != {CANONICAL_PROTOCOL_SHA256}",
        )
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
    # Once accepted, downstream code reads the fully spelled canonical shape.
    normalized = OmegaConf.create(expected)
    OmegaConf.resolve(normalized)
    assert isinstance(normalized, DictConfig)
    parse_config(normalized)
    return normalized


def resolve_protocol_manifests(cfg: DictConfig) -> tuple[Path, Path, list[str]]:
    """Resolve and validate the frozen train/evaluation prompt split."""

    training_path = (
        Path(str(OmegaConf.select(cfg, "data.manifest", default="") or "")).expanduser().resolve()
    )
    eval_path = (
        Path(str(OmegaConf.select(cfg, "data.eval_manifest", default="") or ""))
        .expanduser()
        .resolve()
    )
    for label, path in (("training", training_path), ("evaluation", eval_path)):
        if not path.is_file():
            raise FileNotFoundError(f"SANA {label} manifest does not exist: {path}")
    if sha256_file(training_path) != TRAIN_MANIFEST_SHA256:
        raise ValueError(
            f"SANA training manifest does not match the registered asset: {training_path}",
        )
    if sha256_file(eval_path) != EVAL_MANIFEST_SHA256:
        raise ValueError(
            f"SANA evaluation manifest does not match the registered asset: {eval_path}",
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


def validate_training_metrics(path: Path, cfg: DictConfig) -> None:
    """Fail when the training CSV does not cover every registered update."""

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


def validate_training_log_provenance(run_dir: Path, cfg: DictConfig) -> dict[str, Any]:
    """Bind the supervisor log to revisions pinned in the resolved config."""

    path = run_dir / "supervisor.log"
    if not path.is_file():
        raise FileNotFoundError("SANA run has no supervisor.log launch evidence")
    reward_kwargs = OmegaConf.to_container(
        OmegaConf.select(cfg, "reward.kwargs", default={}),
        resolve=True,
    )
    reward_kwargs = dict(reward_kwargs or {})
    aesthetic = dict(reward_kwargs.get("aesthetic") or {})
    pickscore = dict(reward_kwargs.get("pickscore") or {})
    configured = {
        str(cfg.model.path): str(cfg.model.revision or ""),
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


def resolve_sampling() -> dict[str, Any]:
    """Return the quality protocol, intentionally independent of training SDE."""

    return dict(OFFICIAL_SAMPLING_PROTOCOL)


def build_reward_model_definitions(
    cfg: DictConfig,
    *,
    generation_device: str,
) -> list[RewardModelDefinition]:
    """Resolve the fixed reward implementations and persisted identities."""

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


def reward_model_record(reward_model: RewardModelDefinition) -> dict[str, Any]:
    """Project one runtime reward definition into persisted provenance."""

    return {
        "name": reward_model.name,
        "score_key": reward_model.score_key,
        "model_factory": reward_model.model_factory,
        "device": str(reward_model.model_config["device"]),
        "dtype": str(reward_model.model_config["dtype"]),
        "identity": reward_model.provenance,
    }


def summarize_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive the report summary from its scored sample rows."""

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


def group_seed(prompt_index: int) -> int:
    """Return the registered seed for one prompt group."""

    return EVAL_BASE_SEED + prompt_index * EVAL_SAMPLES_PER_PROMPT


def checkpoint_curve_epochs(cfg: DictConfig) -> list[int]:
    """Return the preregistered curve epochs after validating save cadence."""

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
    return list(range(EVAL_CHECKPOINT_INTERVAL, total_epochs + 1, EVAL_CHECKPOINT_INTERVAL))


def seed_grid_record() -> dict[str, Any]:
    """Return the persisted identity of the fixed prompt/sample grid."""

    return {
        "base_seed": EVAL_BASE_SEED,
        "samples_per_prompt": EVAL_SAMPLES_PER_PROMPT,
        "formula": "base_seed + prompt_index * samples_per_prompt",
        "sample_stream": "one batched torch.Generator stream per prompt group",
    }


def evaluation_curve_record() -> dict[str, int]:
    """Return the persisted checkpoint-selection protocol."""

    return {"checkpoint_interval": EVAL_CHECKPOINT_INTERVAL}


def load_report_metrics(run_dir: str | Path) -> list[dict[str, float]]:
    """Load and fully validate one canonical standalone evaluation report."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "provenance": provenance,
        "metrics": metrics,
    }
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    load_report_metrics(run_dir)
    return path


def _section(config: Any, *path: str) -> dict[str, Any] | None:
    """Return the nested mapping, or None when any hop is absent/not a dict."""

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

    ``resolve`` handles public spellings wider than their meaning. Invalid
    values stay visible so protocol drift still fails closed.
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

    Default-valued keys are erased symmetrically. The retired synchronous actor
    also spelled its single mailbox slot explicitly, and its entrypoint obtained
    the SANA fp16 precision defaults from the family registry instead of YAML.
    Only those exact historical shapes are accepted.
    """

    from dataclasses import fields as dataclass_fields

    from vrl.config.algorithm import resolve_kl_reward_coef
    from vrl.config.schema import RolloutRuntimeSection
    from vrl.trajectory import TrajectoryStoragePolicy, trajectory_storage_policy_from_cfg

    def storage_policy(value: Any) -> Any:
        """Resolve a storage block without hiding unknown keys."""

        policy_fields = {item.name for item in dataclass_fields(TrajectoryStoragePolicy)}
        if isinstance(value, dict) and set(value) != policy_fields:
            raise ValueError(f"unexpected trajectory_storage keys: {sorted(set(value))}")
        return trajectory_storage_policy_from_cfg(value)

    trainer = actual.get("trainer")
    # Historical reports may carry the threshold under debug. This protocol
    # adapter accepts that old spelling without teaching live training configs
    # an alias or discarding conflicting/unknown persisted settings.
    debug = _section(actual, "trainer", "debug")
    if isinstance(trainer, dict) and debug is not None and "max_abs_logprob_diff" in debug:
        parity = trainer.setdefault("replay_parity", {})
        if not isinstance(parity, dict) or "max_abs_logprob_diff" in parity:
            raise ValueError("ambiguous SANA parity threshold in debug and replay_parity")
        parity["max_abs_logprob_diff"] = debug.pop("max_abs_logprob_diff")
    uses_retired_entrypoint = (
        isinstance(trainer, dict)
        and trainer.get("entrypoint") == _RETIRED_ENTRYPOINT
        and _section(canonical, "trainer") is not None
        and canonical["trainer"].get("entrypoint") == _LIVE_ENTRYPOINT
    )
    # 2026-08 batch-vocabulary rename: historical resolved configs persist the
    # pre-rename keys; translate them so only real behavioral drift is visible.
    for path, old, new in (
        (("rollout",), "samples_per_chunk", "samples_per_generation_batch"),
        (("distributed", "rollout"), "chunk_placement_strategy", "batch_placement_strategy"),
        (("actor",), "replay_samples_per_chunk", "samples_per_replay_batch"),
    ):
        renamed_section = _section(actual, *path)
        if isinstance(renamed_section, dict) and old in renamed_section:
            renamed_section[new] = renamed_section.pop(old)

    # 2026-08 sharing-grammar simplification: allow_overlap was retired and the
    # canonical preset chain dropped its default-valued resource spellings
    # (visible_devices auto probe, one trainer GPU, one one-GPU rollout worker).
    # Only those exact historical shapes are meaningless spelling — any other
    # persisted value stays visible as drift.
    resources_section = _section(actual, "distributed", "resources")
    if isinstance(resources_section, dict):
        if resources_section.get("allow_overlap") is True:
            resources_section.pop("allow_overlap")
        _drop_default_key(resources_section, "visible_devices", default="auto")
        _drop_default_key(_section(resources_section, "trainer"), "num_gpus", default=1)
        rollout_resources = _section(resources_section, "rollout")
        # Historical spellings: gpus_per_worker was deleted and num_workers was
        # renamed to num_engines; persisted run configs keep the old keys.
        _drop_default_key(rollout_resources, "gpus_per_worker", default=1.0)
        _drop_default_key(rollout_resources, "num_workers", default=1)
        if resources_section.get("trainer") == {}:
            resources_section.pop("trainer")

    # Defaults come from their live owners so a changed default cannot silently
    # keep validating stale runs.
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
            for name, default in RolloutRuntimeSection().model_dump().items()
        ),
    ]
    for path, key, default, resolve in default_equivalent:
        for side in (actual, canonical):
            _drop_default_key(_section(side, *path), key, default=default, resolve=resolve)

    actual_rollout = _section(actual, "distributed", "rollout")
    if actual_rollout is not None and actual_rollout.get("max_inflight_chunks_per_worker") == 1:
        actual_rollout.pop("max_inflight_chunks_per_worker")

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

    # Both sides were edited, so callers must compare the normalized pair.
    return actual, canonical


def _first_config_difference(actual: Any, expected: Any, path: str = "") -> str:
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
            mismatch = _first_config_difference(actual_item, expected_item, f"{path}[{index}]")
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
    if config_record.get("canonical_protocol") != CANONICAL_CONFIG_NAME:
        raise ValueError("SANA evaluation report names the wrong canonical config protocol")
    if config_record.get("canonical_protocol_sha256") != CANONICAL_PROTOCOL_SHA256:
        raise ValueError("SANA evaluation report names the wrong canonical protocol digest")
    cfg = normalize_run_config(load_config(config_path))
    validate_training_metrics(training_metrics_path, cfg)
    if provenance["training_log"] != validate_training_log_provenance(run_dir, cfg):
        raise ValueError("SANA evaluation training-log provenance changed")
    expected_model = {
        "family": str(cfg.model.family),
        "repo": str(cfg.model.path),
        "revision": str(cfg.model.revision),
    }
    if provenance["model"] != expected_model:
        raise ValueError("SANA evaluation model provenance disagrees with resolved_config.yaml")

    training_manifest_path, eval_manifest_path, prompts = resolve_protocol_manifests(cfg)
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
            raise ValueError(f"SANA {label} provenance disagrees with resolved_config.yaml")
        _require_matching_file(path, record, label=label.replace("_", " "))
        if int(record.get("prompt_count", -1)) != expected_count:
            raise ValueError(f"SANA {label} prompt count changed")

    if provenance["sampling"] != resolve_sampling():
        raise ValueError("SANA evaluation sampling provenance changed")
    if provenance["scheduler_protocol"] != SCHEDULER_PROTOCOL:
        raise ValueError("SANA evaluation scheduler protocol changed")
    expected_seed = seed_grid_record()
    if provenance["seed_grid"] != expected_seed:
        raise ValueError(
            "SANA evaluation report seed protocol changed: "
            f"{provenance['seed_grid']!r} != {expected_seed!r}",
        )
    if provenance["evaluation_curve"] != evaluation_curve_record():
        raise ValueError("SANA evaluation checkpoint interval changed")

    execution = provenance["execution"]
    if not str(execution.get("generation_device", "")):
        raise ValueError("SANA evaluation report execution provenance is incomplete")
    expected_rewards = [
        reward_model_record(reward_model)
        for reward_model in build_reward_model_definitions(
            cfg,
            generation_device=str(execution["generation_device"]),
        )
    ]
    if provenance["rewards"] != expected_rewards:
        raise ValueError("SANA evaluation reward provenance disagrees with resolved_config.yaml")

    checkpoints = provenance["checkpoints"]
    _validate_checkpoint_records(run_dir, cfg, checkpoints)

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
        if int(row["group_seed"]) != group_seed(prompt_index):
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
    cfg: DictConfig,
    checkpoints: Any,
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
    expected_epochs = checkpoint_curve_epochs(cfg)
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
