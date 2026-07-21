"""Checkpoint helpers for resumable training."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from vrl.trainers.weight_sync import require_trainable_modules, to_cpu
from vrl.utils.config import cfg_get, cfg_path

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
TRAINING_CHECKPOINT_NAME = "checkpoint.pt"
LORA_WEIGHTS_NAME = "lora_weights"
CHECKPOINT_META_NAME = "checkpoint_meta.json"


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """Torch-style training checkpoint payload.

    ``checkpoint.pt`` is the source of truth for resume. Export directories
    such as ``lora_weights/`` are optional artifacts for warm-start/publishing.
    """

    checkpoint_dir: Path
    checkpoint_path: Path
    payload: dict[str, Any]
    meta: dict[str, Any]

    @property
    def trainer_state(self) -> dict[str, Any]:
        trainer = self.payload.get("trainer")
        if not isinstance(trainer, dict):
            raise TypeError("checkpoint payload missing dict field: trainer")
        return trainer

    @property
    def trainable_state(self) -> dict[str, Any]:
        model = self.payload.get("model")
        if not isinstance(model, dict):
            raise TypeError("checkpoint payload missing dict field: model")
        state = model.get("trainable_modules")
        if not isinstance(state, dict):
            raise TypeError("checkpoint payload missing dict field: model.trainable_modules")
        return state

    @property
    def model_identity(self) -> dict[str, Any] | None:
        model = self.payload.get("model")
        if not isinstance(model, dict):
            raise TypeError("checkpoint payload missing dict field: model")
        identity = model.get("identity")
        if identity is None:
            return None
        if not isinstance(identity, dict):
            raise TypeError("checkpoint payload field model.identity must be a dict")
        return identity

    @property
    def progress(self) -> dict[str, Any]:
        progress = self.payload.get("progress", {})
        if not isinstance(progress, dict):
            raise TypeError("checkpoint payload field progress must be a dict")
        return progress

    @property
    def rng_state(self) -> dict[str, Any]:
        rng = self.payload.get("rng", {})
        if not isinstance(rng, dict):
            raise TypeError("checkpoint payload field rng must be a dict")
        return rng

    @property
    def next_epoch(self) -> int:
        if "next_epoch" in self.progress:
            return _non_negative_int(self.progress["next_epoch"], "progress.next_epoch")
        return infer_next_epoch(self.checkpoint_dir, self.trainer_state, self.meta)

    @property
    def next_step(self) -> int:
        if "next_step" in self.progress:
            return _non_negative_int(self.progress["next_step"], "progress.next_step")
        if "global_step" in self.trainer_state:
            return _non_negative_int(self.trainer_state["global_step"], "trainer.global_step")
        if "step" in self.trainer_state:
            return _non_negative_int(self.trainer_state["step"], "trainer.step")
        return self.next_epoch


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    trainer: Any,
    bundle: Any,
    family: str,
    progress: dict[str, Any],
    rng_state: dict[str, Any] | None = None,
    export_modules: dict[str, Any] | None = None,
    export_ema: Any | None = None,
    model_identity: dict[str, Any] | None = None,
    strategy: Any | None = None,
    is_primary: bool = True,
) -> dict[str, Any]:
    """Save a generic Torch training checkpoint.

    Every model family participates through ``RuntimeBundle.trainable_modules``.
    No family-specific serialization code is needed for resume. ``checkpoint.pt``
    always stores the current/raw training state. When ``export_ema`` is passed,
    optional ``save_pretrained`` artifacts are written from EMA weights and then
    the raw training weights are restored.

    Trainable-state export goes through the training ``strategy`` when one is
    wired (the strategy seam owns state export; FSDP2 gathers only mutable
    trainable tensors). ``export_trainable_state`` below is the
    single_process implementation -- the strategy delegates to it -- and stays
    the default for callers without a strategy (e.g. the Wan DPO trainer).

    ``is_primary`` MUST be passed under a multi-rank strategy. The trainable-state
    export is a COLLECTIVE under FSDP2 (each trainable DTensor is all-gathered
    across every rank), so it runs on all ranks below; only the primary rank then writes
    the files. Gating the whole call to rank0 (the natural "only rank0 owns IO"
    instinct) deadlocks FSDP: rank0 waits at the all-gather for peers that never
    join. single_process / ddp gather locally, so for them this is a no-op.
    """

    export_trainable = (
        strategy.export_trainable_state if strategy is not None else export_trainable_state
    )
    # Collectives on ALL ranks (FSDP all-gathers) — before the is_primary gate:
    # the trainable-state gather, and trainer.state_dict() (whose optimizer
    # moments and EMA shadows are DTensor shards that gather to full tensors).
    trainable_modules = export_trainable(bundle)
    trainer_state = trainer.state_dict()
    if not is_primary:
        # Non-primary ranks joined the gathers above; only rank0 writes the files.
        return {}

    # Atomic publish: every artifact (checkpoint.pt, exports, meta) is written
    # into a same-filesystem staging directory, fsynced, and then renamed into
    # place in one os.replace. A crash at any point leaves either the previous
    # complete checkpoint or an ignorable *.tmp-* directory — never a torn
    # checkpoint.pt that a supervisor resume would trust.
    final_path = Path(checkpoint_dir)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    path = Path(
        tempfile.mkdtemp(prefix=f"{final_path.name}.tmp-", dir=final_path.parent),
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": family,
        "trainer": trainer_state,
        "model": {
            "trainable_modules": trainable_modules,
            **({"identity": dict(model_identity)} if model_identity is not None else {}),
        },
        "progress": dict(progress),
        "rng": rng_state or capture_rng_state(),
    }
    try:
        checkpoint_file = path / TRAINING_CHECKPOINT_NAME
        torch.save(payload, checkpoint_file)
        with checkpoint_file.open("rb") as handle:
            os.fsync(handle.fileno())
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise

    try:
        return _write_checkpoint_artifacts_and_publish(
            staging=path,
            final_path=final_path,
            checkpoint_file=checkpoint_file,
            bundle=bundle,
            family=family,
            progress=progress,
            trainer_state=trainer_state,
            trainable_modules=trainable_modules,
            export_modules=export_modules,
            export_ema=export_ema,
        )
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise


def _write_checkpoint_artifacts_and_publish(
    *,
    staging: Path,
    final_path: Path,
    checkpoint_file: Path,
    bundle: Any,
    family: str,
    progress: dict[str, Any],
    trainer_state: dict[str, Any],
    trainable_modules: dict[str, Any],
    export_modules: dict[str, Any] | None,
    export_ema: Any | None,
) -> dict[str, Any]:
    """Write exports + meta into ``staging``, then atomically rename into place."""

    path = staging
    export_modules = export_modules or {}
    trainable_parameters: list[Any] = []
    used_ema_export = False
    if export_ema is not None and export_modules:
        has_updates = getattr(export_ema, "has_updates", None)
        # A missing has_updates attribute means the EMA cannot report, so export.
        if has_updates is None:
            has_updates = True
        elif callable(has_updates):
            has_updates = has_updates()
        if has_updates:
            for module in bundle.trainable_modules.values():
                parameters = getattr(module, "parameters", None)
                if callable(parameters):
                    trainable_parameters.extend(p for p in parameters() if p.requires_grad)
            if not trainable_parameters:
                raise ValueError("export_ema was provided but bundle has no trainable parameters")
            export_ema.copy_ema_to(trainable_parameters, store_temp=True)
            used_ema_export = True
        else:
            logger.info(
                "Skipping EMA export weights because EMA has not updated; "
                "exporting raw trainable weights.",
            )

    try:
        from torch.distributed.tensor import DTensor

        for name, module in export_modules.items():
            save_pretrained = getattr(module, "save_pretrained", None)
            if not callable(save_pretrained):
                raise TypeError(f"export module {name!r} does not expose save_pretrained()")
            # A DTensor-sharded module (FSDP2) cannot drive the default
            # save_pretrained path: it reads module.state_dict(), which yields
            # shards, not a loadable artifact. The strategy already gathered
            # the full CPU state for checkpoint.pt above — hand that state to
            # save_pretrained (PEFT extracts the adapter keys from it) so the
            # artifact carries the same gathered tensors. Plain-tensor modules
            # keep the default path, which is what the EMA export relies on
            # (copy-EMA-into-params -> save_pretrained -> restore reads the
            # LIVE parameters); EMA+fsdp is rejected at strategy construction,
            # so the two paths never conflict.
            parameters = getattr(module, "parameters", None)
            if callable(parameters) and any(isinstance(p, DTensor) for p in parameters()):
                gathered_state = next(
                    (
                        trainable_modules[trainable_name]
                        for trainable_name, trainable in getattr(
                            bundle,
                            "trainable_modules",
                            {},
                        ).items()
                        if trainable is module and trainable_name in trainable_modules
                    ),
                    None,
                )
                if gathered_state is None:
                    raise ValueError(
                        f"export module {name!r} is DTensor-sharded but is not one "
                        "of the bundle's gathered trainable modules; a sharded "
                        "save_pretrained would write shard wrappers instead of a "
                        "loadable artifact",
                    )
                save_pretrained(path / name, state_dict=gathered_state)
            else:
                save_pretrained(path / name)
    finally:
        if used_ema_export:
            export_ema.copy_temp_to(trainable_parameters)

    meta = write_checkpoint_meta(
        path,
        family=family,
        trainer_state=trainer_state,
        completed_epoch=int(progress.get("completed_epoch", progress.get("next_epoch", 0))),
        next_epoch=int(progress.get("next_epoch", progress.get("next_step", 0))),
        uses_lora=any(
            name == LORA_WEIGHTS_NAME or name.startswith(f"{LORA_WEIGHTS_NAME}/")
            for name in export_modules
        ),
        checkpoint_file_bytes=checkpoint_file.stat().st_size,
    )
    _publish_checkpoint_dir(staging, final_path)
    return meta


def _publish_checkpoint_dir(staging: Path, final_path: Path) -> None:
    """Atomically rename the fully written staging directory into place.

    Re-saving to an existing directory (crash-loop overwriting the same
    ``checkpoint-N``) removes the stale directory first; the replaced window is
    not atomic, but the staging directory is complete before it opens, and
    discovery ignores ``*.tmp-*`` so no reader can observe a partial state.
    """

    if final_path.exists():
        shutil.rmtree(final_path)
    os.replace(staging, final_path)
    directory_fd = os.open(final_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_training_checkpoint(path: str | Path) -> TrainingCheckpoint:
    """Load ``checkpoint.pt`` from a checkpoint directory or direct file path."""

    raw_path = Path(path).expanduser().resolve()
    checkpoint_path = raw_path if raw_path.is_file() else raw_path / TRAINING_CHECKPOINT_NAME
    checkpoint_dir = checkpoint_path.parent
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"training checkpoint file not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{checkpoint_path} must contain a dict payload")
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version={schema_version}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}",
        )
    return TrainingCheckpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        payload=payload,
        meta=read_checkpoint_meta(checkpoint_dir),
    )


def load_training_checkpoint_from_config(cfg: Any) -> TrainingCheckpoint | None:
    """Return configured resume checkpoint, or ``None`` for a fresh run."""

    resume_from = str(cfg_path(cfg, "trainer.resume_from", "") or "").strip()
    if not resume_from:
        return None
    return load_training_checkpoint(resume_from)


def prepare_model_config_for_training_resume(
    cfg: Any,
    checkpoint: TrainingCheckpoint | None,
    *,
    strict: bool = True,
) -> None:
    """Remove warm-start adapter paths when doing full training resume.

    Full resume restores ``RuntimeBundle.trainable_modules`` from
    ``checkpoint.pt``. Loading an unrelated ``model.lora.path`` before that can
    silently alter adapter structure, so strict mode rejects the combination.
    """

    if checkpoint is None:
        return
    lora_path = cfg_path(cfg, "model.lora.path", None)
    if lora_path is None:
        return
    text = str(lora_path or "").strip()
    if text and strict:
        raise ValueError(
            "trainer.resume_from cannot be combined with model.lora.path; "
            "checkpoint.pt is the resume source of truth",
        )
    _set_cfg_path(cfg, "model.lora.path", "")


def restore_training_checkpoint(
    checkpoint: TrainingCheckpoint | None,
    *,
    trainer: Any,
    bundle: Any,
    family: str,
    expected_model_identity: dict[str, Any] | None = None,
    strict: bool = True,
) -> None:
    """Restore model trainable modules and trainer state from checkpoint."""

    if checkpoint is None:
        return
    if strict and not family:
        raise ValueError("strict checkpoint restore requires a non-empty family")
    checkpoint_family = checkpoint.payload.get("family")
    if strict and checkpoint_family and str(checkpoint_family) != str(family):
        raise ValueError(
            f"checkpoint family mismatch: checkpoint={checkpoint_family!r}, runtime={family!r}",
        )
    if strict and expected_model_identity is not None:
        saved_identity = checkpoint.model_identity
        if saved_identity is None:
            raise ValueError(
                "checkpoint is missing model identity required by this runtime",
            )
        if saved_identity != expected_model_identity:
            raise ValueError(
                "checkpoint model identity mismatch: "
                f"checkpoint={saved_identity!r}, runtime={expected_model_identity!r}",
            )
    strategy = getattr(trainer, "_strategy", None)
    strategy_loader = getattr(strategy, "load_trainable_state", None)
    if callable(strategy_loader):
        # Model state must use the same distributed seam as save: FSDP resumes
        # full LoRA tensors by scattering them back into local DTensor shards.
        strategy_loader(bundle, checkpoint.trainable_state, strict=strict)
    else:
        load_trainable_state(bundle, checkpoint.trainable_state, strict=strict)
    trainer.load_state_dict(checkpoint.trainer_state, strict=strict)
    if strict and expected_model_identity is not None:
        restored_trainable = (
            strategy.export_trainable_state(bundle)
            if callable(getattr(strategy, "export_trainable_state", None))
            else export_trainable_state(bundle)
        )
        _require_equal_tensor_tree(
            checkpoint.trainable_state,
            restored_trainable,
            label="restored dual-expert weights",
        )
        if "optimizer" in checkpoint.trainer_state:
            restored_trainer = trainer.state_dict()
            _require_equal_tensor_tree(
                checkpoint.trainer_state["optimizer"],
                restored_trainer.get("optimizer"),
                label="restored dual-expert optimizer state",
            )


def _require_equal_tensor_tree(expected: Any, actual: Any, *, label: str) -> None:
    """Fail closed unless two checkpoint state trees are tensor-exact."""

    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if not torch.equal(expected, actual):
            raise ValueError(f"{label} tensor mismatch")
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(
                f"{label} keys mismatch: expected={sorted(expected)}, actual={sorted(actual)}",
            )
        for key in expected:
            _require_equal_tensor_tree(expected[key], actual[key], label=f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            raise ValueError(
                f"{label} length mismatch: expected={len(expected)}, actual={len(actual)}",
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True),
        ):
            _require_equal_tensor_tree(
                expected_item,
                actual_item,
                label=f"{label}[{index}]",
            )
        return
    if expected != actual:
        raise ValueError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def export_trainable_state(bundle: Any) -> dict[str, dict[str, Any]]:
    """Export all trainable module state_dicts to CPU tensors."""

    modules = require_trainable_modules(bundle)
    out: dict[str, dict[str, Any]] = {}
    for name, module in modules.items():
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(f"trainable module {name!r} does not expose state_dict()")
        out[name] = to_cpu(state_dict())
    return out


def load_trainable_state(
    bundle: Any,
    state: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Load trainable module state_dicts into a runtime bundle."""

    modules = require_trainable_modules(bundle)
    missing = sorted(set(modules) - set(state))
    extra = sorted(set(state) - set(modules))
    if strict and (missing or extra):
        raise ValueError(
            f"checkpoint trainable module keys mismatch: missing={missing}, extra={extra}",
        )
    for name, module in modules.items():
        if name not in state:
            continue
        load_state_dict = getattr(module, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError(f"trainable module {name!r} does not expose load_state_dict()")
        load_state_dict(state[name], strict=strict)


def capture_rng_state(**generators: torch.Generator) -> dict[str, Any]:
    """Capture process RNG state plus named torch.Generator states."""

    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "generators": {name: gen.get_state() for name, gen in generators.items()},
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import random

        state["python_random"] = random.getstate()
    except Exception:
        pass
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def restore_rng_state(state: dict[str, Any] | None, **generators: torch.Generator) -> None:
    """Restore process RNG state and named torch.Generator states when present."""

    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    named = state.get("generators", {})
    if isinstance(named, dict):
        for name, gen in generators.items():
            if name in named:
                gen.set_state(named[name])
    if "python_random" in state:
        try:
            import random

            random.setstate(state["python_random"])
        except Exception:
            pass
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass


def save_resolved_config(cfg: Any, output_dir: str | Path, *, resumed: bool) -> None:
    """Save resolved config without overwriting the original on resume."""

    from omegaconf import OmegaConf

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    resolved_path = path / "resolved_config.yaml"
    if not resumed or not resolved_path.exists():
        OmegaConf.save(cfg, resolved_path)
        return
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    OmegaConf.save(cfg, path / f"resume_config_{stamp}.yaml")


def prepare_metrics_csv(
    csv_path: str | Path,
    header: str,
    *,
    resume_at: tuple[str, int] | None,
) -> None:
    """Create a metrics CSV or align it atomically with a resume checkpoint.

    A checkpoint at position N cannot support metrics already written for N or
    later: those updates were not captured in the checkpoint and will be
    recomputed. Keeping them would create duplicate positions after resume.
    """

    path = Path(csv_path)
    normalized_header = header.rstrip("\r\n") + "\n"
    if resume_at is None:
        path.write_text(normalized_header)
        return

    position_column, resume_position = resume_at
    if resume_position < 0:
        raise ValueError(f"metrics resume position must be >= 0, got {resume_position}")
    if not path.exists():
        logger.warning("Resume requested but metrics file does not exist; creating %s", path)
        path.write_text(normalized_header)
        return

    text = path.read_text(encoding="utf-8")
    complete_text = text if text.endswith("\n") else text.rpartition("\n")[0] + "\n"
    lines = complete_text.splitlines(keepends=True)
    existing_header = lines[0] if lines else ""
    if existing_header.rstrip("\r\n") != normalized_header.rstrip("\n"):
        raise ValueError(
            f"{path} was written by a different metrics schema; appending "
            "would silently misalign columns. Move the old file aside or "
            "start a fresh output_dir.",
        )

    columns = next(csv.reader([existing_header]))
    if position_column not in columns:
        raise ValueError(f"metrics CSV is missing resume column {position_column!r}: {path}")
    position_index = columns.index(position_column)
    retained_lines: list[str] = []
    previous_position = -1
    truncated_rows = 0
    for line_number, line in enumerate(lines[1:], start=2):
        values = next(csv.reader([line]))
        if len(values) != len(columns):
            raise ValueError(
                f"metrics CSV row {line_number} has {len(values)} columns; "
                f"expected {len(columns)}: {path}",
            )
        raw_position = values[position_index]
        try:
            position = int(raw_position)
        except ValueError as exc:
            raise ValueError(
                f"metrics CSV row {line_number} has a non-integer "
                f"{position_column}: {raw_position!r}",
            ) from exc
        if position < 0 or position <= previous_position:
            raise ValueError(
                f"metrics CSV {position_column} must be strictly increasing "
                f"non-negative integers; row {line_number} has {position}",
            )
        previous_position = position
        if position < resume_position:
            retained_lines.append(line)
        else:
            truncated_rows += 1

    aligned_text = normalized_header + "".join(retained_lines)
    if aligned_text != text:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(aligned_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    if truncated_rows:
        logger.warning(
            "Discarded %d metrics row(s) at %s >= %d before checkpoint resume",
            truncated_rows,
            position_column,
            resume_position,
        )
    elif previous_position + 1 < resume_position:
        logger.warning(
            "Metrics end at %s=%d before checkpoint resume position %d",
            position_column,
            previous_position,
            resume_position,
        )


def read_checkpoint_meta(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata if present."""

    meta_path = Path(checkpoint_dir) / CHECKPOINT_META_NAME
    if not meta_path.exists():
        return {}
    raw = json.loads(meta_path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{meta_path} must contain a JSON object")
    return raw


def infer_next_epoch(
    checkpoint_dir: str | Path,
    trainer_state: dict[str, Any],
    meta: dict[str, Any] | None,
) -> int:
    """Infer the epoch index to start from when resuming."""

    meta = meta or {}
    if "next_epoch" in meta:
        return _non_negative_int(meta["next_epoch"], "checkpoint_meta.next_epoch")

    if "step" in trainer_state:
        return _non_negative_int(trainer_state["step"], "trainer_state.step")

    checkpoint_name = Path(checkpoint_dir).name
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_name)
    if match:
        return _non_negative_int(match.group(1), "checkpoint directory suffix")

    raise ValueError(
        "cannot infer next_epoch: checkpoint_meta.next_epoch and trainer_state.step are missing",
    )


def write_checkpoint_meta(
    checkpoint_dir: str | Path,
    *,
    family: str,
    trainer_state: dict[str, Any],
    completed_epoch: int,
    next_epoch: int,
    uses_lora: bool,
    checkpoint_file_bytes: int | None = None,
) -> dict[str, Any]:
    """Write human-readable checkpoint metadata next to ``checkpoint.pt``.

    ``checkpoint_file_bytes`` records the published size of ``checkpoint.pt``
    so completeness checks (supervisor resume discovery) can reject truncated
    copies without loading the payload.
    """

    meta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": family,
        "checkpoint_file": TRAINING_CHECKPOINT_NAME,
        "checkpoint_file_bytes": (
            int(checkpoint_file_bytes) if checkpoint_file_bytes is not None else None
        ),
        "trainer_step": int(trainer_state.get("step", 0)),
        "global_step": int(trainer_state.get("global_step", 0)),
        "completed_epoch": int(completed_epoch),
        "next_epoch": int(next_epoch),
        "uses_lora": bool(uses_lora),
    }
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / CHECKPOINT_META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def is_complete_checkpoint(checkpoint_dir: str | Path) -> bool:
    """True when the directory holds a fully published checkpoint.

    Complete means: not a staging leftover, meta present, ``checkpoint.pt``
    present, and (when recorded) the published byte size matches. This is the
    supervisor's trust boundary — anything else is treated as absent.
    """

    path = Path(checkpoint_dir)
    if ".tmp-" in path.name:
        return False
    checkpoint_file = path / TRAINING_CHECKPOINT_NAME
    if not checkpoint_file.is_file():
        return False
    try:
        meta = read_checkpoint_meta(path)
    except (ValueError, TypeError, json.JSONDecodeError):
        return False
    if not meta:
        return False
    expected_bytes = meta.get("checkpoint_file_bytes")
    return not (
        expected_bytes is not None and checkpoint_file.stat().st_size != int(expected_bytes)
    )


def find_latest_complete_checkpoint(output_dir: str | Path) -> Path | None:
    """Latest fully published ``checkpoint-*`` directory under ``output_dir``.

    Ordered by meta ``global_step`` (falling back to the numeric directory
    suffix); incomplete/staging directories are skipped entirely. Returns
    ``None`` when no trustworthy checkpoint exists — the caller starts fresh.
    """

    root = Path(output_dir)
    if not root.is_dir():
        return None
    best: tuple[int, int, Path] | None = None
    for candidate in root.glob("checkpoint-*"):
        if not candidate.is_dir() or not is_complete_checkpoint(candidate):
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
        suffix = int(match.group(1)) if match else -1
        meta = read_checkpoint_meta(candidate)
        global_step = int(meta.get("global_step", 0) or 0)
        key = (global_step, suffix, candidate)
        if best is None or key[:2] > best[:2]:
            best = key
    return best[2] if best else None


def _non_negative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be >= 0, got {parsed}")
    return parsed


_MISSING = object()


def _set_cfg_path(cfg: Any, path: str, value: Any) -> None:
    node = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        node = cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return
    try:
        node[keys[-1]] = value
    except TypeError:
        setattr(node, keys[-1], value)


__all__ = [
    "CHECKPOINT_META_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "LORA_WEIGHTS_NAME",
    "TRAINING_CHECKPOINT_NAME",
    "TrainingCheckpoint",
    "capture_rng_state",
    "export_trainable_state",
    "find_latest_complete_checkpoint",
    "infer_next_epoch",
    "is_complete_checkpoint",
    "load_trainable_state",
    "load_training_checkpoint",
    "load_training_checkpoint_from_config",
    "prepare_metrics_csv",
    "prepare_model_config_for_training_resume",
    "read_checkpoint_meta",
    "restore_rng_state",
    "restore_training_checkpoint",
    "save_resolved_config",
    "save_training_checkpoint",
    "write_checkpoint_meta",
]
