"""Checkpoint helpers for resumable training."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch

from vrl.models.interfaces.runtime import checkpoint_owned_state_names
from vrl.trainers.weight_sync import (
    require_trainable_modules,
    to_cpu_snapshot,
    unwrap_compile_and_ddp,
)
from vrl.utils.config import cfg_get, cfg_path

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 2
_LEGACY_CHECKPOINT_SCHEMA_VERSION = 1
TRAINING_CHECKPOINT_NAME = "checkpoint.pt"
LORA_WEIGHTS_NAME = "lora_weights"
CHECKPOINT_META_NAME = "checkpoint_meta.json"
# Strict restore is the checkpoint protocol default, independent of any
# particular trainer implementation.
DEFAULT_CHECKPOINT_STRICT = True


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
    def schema_version(self) -> int:
        return _require_checkpoint_schema_version(
            self.payload,
            source="checkpoint payload",
        )

    @property
    def checkpoint_state(self) -> dict[str, Any]:
        """Return the schema-specific model state under its canonical API."""

        model = self.payload.get("model")
        if not isinstance(model, dict):
            raise TypeError("checkpoint payload missing dict field: model")
        key = (
            "owned_state"
            if self.schema_version == CHECKPOINT_SCHEMA_VERSION
            else "trainable_modules"
        )
        state = model.get(key)
        if not isinstance(state, dict):
            raise TypeError(f"checkpoint payload missing dict field: model.{key}")
        return state

    @property
    def trainable_state(self) -> dict[str, Any]:
        """Compatibility facade for callers using the schema-v1 property name."""

        return self.checkpoint_state

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


@dataclass(frozen=True, slots=True)
class TrainingResumeConfig:
    """Resolved checkpoint input shared by online and offline trainers."""

    checkpoint_path: str | None = None
    strict: bool = DEFAULT_CHECKPOINT_STRICT


@dataclass(frozen=True, slots=True)
class _CheckpointStateForRestore:
    """Validated model state plus roots using the schema-v1 full-state protocol."""

    state: dict[str, dict[str, Any]]
    full_state_roots: frozenset[str]


@dataclass(frozen=True, slots=True)
class _CheckpointRootBundle:
    """Minimal bundle view used to route one root through a strategy loader."""

    trainable_modules: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AdapterExport:
    """One PEFT adapter artifact derived from checkpoint-owned model state."""

    module: Any
    adapter_name: str = "default"

    def __post_init__(self) -> None:
        if not callable(getattr(self.module, "save_pretrained", None)):
            raise TypeError("adapter export module must expose save_pretrained()")
        _safe_relative_output_path(
            self.adapter_name,
            field="adapter export adapter_name",
            allow_nested=False,
        )
        adapter_configs = getattr(self.module, "peft_config", None)
        if isinstance(adapter_configs, Mapping) and self.adapter_name not in adapter_configs:
            raise ValueError(
                f"adapter export module has no adapter named {self.adapter_name!r}",
            )


@dataclass(frozen=True, slots=True)
class _ResolvedAdapterExport:
    """Adapter plus its derived location inside one checkpoint root."""

    export: AdapterExport
    root_name: str
    state_prefix: str


def _safe_relative_output_path(
    value: Any,
    *,
    field: str,
    allow_nested: bool,
) -> PurePosixPath:
    """Validate one dependency-facing output path before any checkpoint IO."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    segments = value.split("/")
    output_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        output_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or "\\" in value
        or any(segment in {"", ".", ".."} for segment in segments)
        or (not allow_nested and len(segments) != 1)
    ):
        path_kind = "relative path" if allow_nested else "single path segment"
        raise ValueError(f"{field} must be a safe {path_kind}: {value!r}")
    return PurePosixPath(*segments)


def _output_paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    """Return whether two artifact writers own the same or nested path."""

    return left == right or left in right.parents or right in left.parents


def _resolve_adapter_exports(
    bundle: Any,
    exports: Mapping[str, AdapterExport] | None,
) -> dict[str, _ResolvedAdapterExport]:
    """Derive each export's unique checkpoint root and relative state prefix."""

    if exports is None:
        return {}
    if not isinstance(exports, Mapping):
        raise TypeError("adapter_exports must be a mapping")
    roots = require_trainable_modules(bundle)
    resolved: dict[str, _ResolvedAdapterExport] = {}
    output_names: dict[PurePosixPath, str] = {}
    effective_output_names: dict[PurePosixPath, str] = {}
    for name, export in exports.items():
        if not isinstance(export, AdapterExport):
            raise TypeError(
                f"adapter export {name!r} must be an AdapterExport, got {type(export).__name__}",
            )
        normalized_name = _safe_relative_output_path(
            name,
            field="adapter export name",
            allow_nested=True,
        )
        effective_path = (
            normalized_name
            if export.adapter_name == "default"
            else normalized_name / export.adapter_name
        )
        existing = effective_output_names.get(effective_path)
        if existing is not None:
            raise ValueError(
                f"adapter exports {existing!r} and {name!r} write the same PEFT output path",
            )
        for existing_path, existing_name in output_names.items():
            if _output_paths_overlap(existing_path, normalized_name):
                raise ValueError(
                    f"adapter export paths {existing_name!r} and {name!r} overlap",
                )
        output_names[normalized_name] = name
        effective_output_names[effective_path] = name

        target = unwrap_compile_and_ddp(export.module)
        matches: list[tuple[str, str]] = []
        for raw_root_name, wrapped_root in roots.items():
            if not isinstance(raw_root_name, str) or not raw_root_name:
                raise ValueError("checkpoint root names must be non-empty strings")
            root = unwrap_compile_and_ddp(wrapped_root)
            if root is target or root is export.module:
                matches.append((raw_root_name, ""))
                continue
            named_modules = getattr(root, "named_modules", None)
            if not callable(named_modules):
                continue
            matches.extend(
                (raw_root_name, prefix)
                for prefix, module in named_modules(remove_duplicate=False)
                if prefix and (module is target or module is export.module)
            )
        if len(matches) != 1:
            raise ValueError(
                f"adapter export {name!r} must map to exactly one bundle checkpoint "
                f"root; found {matches}",
            )
        root_name, state_prefix = matches[0]
        resolved[name] = _ResolvedAdapterExport(
            export=export,
            root_name=root_name,
            state_prefix=state_prefix,
        )
    return resolved


def _checkpoint_stage_agreement(strategy: Any | None, succeeded: bool) -> bool:
    """Agree on rank-local work before another checkpoint collective begins."""

    if strategy is None:
        return succeeded
    agreement = getattr(strategy, "all_ranks_succeeded", None)
    context = getattr(strategy, "context", None)
    world_size = int(getattr(context, "world_size", 1))
    if not callable(agreement):
        if world_size > 1:
            raise TypeError(
                "multi-rank checkpoint strategy must expose all_ranks_succeeded()",
            )
        return succeeded
    result = agreement(succeeded)
    if not isinstance(result, bool):
        raise TypeError("strategy.all_ranks_succeeded() must return bool")
    return result


def _checkpoint_ranks_agree_bool(
    strategy: Any | None,
    value: bool,
    *,
    field: str,
) -> bool:
    """Require every checkpoint rank to choose the same boolean branch."""

    all_true = _checkpoint_stage_agreement(strategy, value)
    all_false = _checkpoint_stage_agreement(strategy, not value)
    if all_true == all_false:
        raise RuntimeError(f"checkpoint ranks disagree on {field}")
    return all_true


def _checkpoint_trainable_parameters(bundle: Any) -> list[Any]:
    """Return EMA parameters in the same order used by the trainer model."""

    model = getattr(bundle, "model", None)
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise TypeError("EMA artifact export requires bundle.model.parameters()")
    ordered = [parameter for parameter in parameters() if parameter.requires_grad]

    root_parameter_ids = {
        id(parameter)
        for root in require_trainable_modules(bundle).values()
        for parameter in unwrap_compile_and_ddp(root).parameters()
        if parameter.requires_grad
    }
    ordered_ids = {id(parameter) for parameter in ordered}
    if ordered_ids != root_parameter_ids:
        raise ValueError(
            "bundle.model trainable parameters disagree with bundle checkpoint roots",
        )
    return ordered


def _adapter_relative_state(
    root_state: Mapping[str, Any],
    *,
    prefix: str,
    artifact_name: str,
) -> dict[str, Any]:
    """Project one gathered checkpoint root into an adapter module namespace."""

    if any(not isinstance(name, str) for name in root_state):
        raise TypeError(
            f"adapter export {artifact_name!r} state keys must be strings",
        )
    if not prefix:
        state = dict(root_state)
    else:
        state_prefix = f"{prefix}."
        state = {
            name.removeprefix(state_prefix): value
            for name, value in root_state.items()
            if name.startswith(state_prefix)
        }
    if not state:
        raise ValueError(
            f"adapter export {artifact_name!r} has no checkpoint-owned state",
        )
    return state


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    trainer: Any,
    bundle: Any,
    family: str,
    progress: dict[str, Any],
    rng_state: dict[str, Any] | None = None,
    adapter_exports: Mapping[str, AdapterExport] | None = None,
    export_ema: Any | None = None,
    model_identity: dict[str, Any],
    strategy: Any | None = None,
) -> dict[str, Any]:
    """Save a generic Torch training checkpoint.

    Every model family participates through ``RuntimeBundle.trainable_modules``.
    No family-specific serialization code is needed for resume. ``checkpoint.pt``
    always stores the current/raw training state. When ``export_ema`` is passed,
    optional adapter artifacts are written from EMA weights after the raw
    training weights are restored.

    Checkpoint-state export goes through the training ``strategy`` when one is
    wired (the strategy seam owns state export; FSDP2 gathers only checkpoint-
    owned tensors). ``export_checkpoint_state`` below is the
    single_process implementation -- the strategy delegates to it -- and stays
    the default for callers without a strategy (e.g. the Wan DPO trainer).

    Checkpoint-state export is a COLLECTIVE under FSDP2 (each owned DTensor is
    all-gathered across every rank), so it runs on all ranks below. A strategy's
    context is the source of truth for which rank writes; callers cannot pass a
    contradictory second ownership flag. Gating the whole call to rank0 (the
    natural "only rank0 owns IO" instinct) deadlocks FSDP: rank0 waits at the
    all-gather for peers that never join. single_process / ddp gather locally,
    so for them this is a no-op.
    """

    _require_checkpoint_family(family, field="family")
    if not isinstance(model_identity, dict) or not model_identity:
        raise ValueError("model_identity must be a non-empty dict")
    is_primary = True
    if strategy is not None:
        context = getattr(strategy, "context", None)
        if context is None or not isinstance(getattr(context, "is_primary", None), bool):
            raise TypeError("checkpoint strategy must expose context.is_primary as bool")
        is_primary = context.is_primary
    resolved_exports: dict[str, _ResolvedAdapterExport] = {}
    ema_has_updates = False
    setup_failure: BaseException | None = None
    try:
        resolved_exports = _resolve_adapter_exports(bundle, adapter_exports)
        if resolved_exports and export_ema is not None:
            try:
                ema_has_updates = export_ema.has_updates
            except AttributeError as error:
                raise TypeError(
                    "export_ema must expose a bool has_updates property",
                ) from error
            if not isinstance(ema_has_updates, bool):
                raise TypeError("export_ema.has_updates must be a bool property")
            for method_name in ("copy_ema_to", "copy_temp_to"):
                if not callable(getattr(export_ema, method_name, None)):
                    raise TypeError(f"export_ema must expose {method_name}()")
    except BaseException as error:
        setup_failure = error
    if not _checkpoint_stage_agreement(strategy, setup_failure is None):
        if setup_failure is not None:
            raise setup_failure
        raise RuntimeError("checkpoint setup was aborted because a peer rank failed")
    if setup_failure is not None:
        raise setup_failure
    uses_ema_export = _checkpoint_ranks_agree_bool(
        strategy,
        bool(resolved_exports and export_ema is not None),
        field="EMA artifact export presence",
    )
    export_checkpoint = (
        strategy.export_checkpoint_state if strategy is not None else export_checkpoint_state
    )
    # Collectives on ALL ranks (FSDP all-gathers) — before the is_primary gate:
    # the checkpoint-state gather, and trainer.state_dict() (whose optimizer
    # moments and EMA shadows are DTensor shards that gather to full tensors).
    checkpoint_state: dict[str, Any] = {}
    checkpoint_failure: BaseException | None = None
    try:
        checkpoint_state = export_checkpoint(bundle)
    except BaseException as error:
        checkpoint_failure = error
    if not _checkpoint_stage_agreement(strategy, checkpoint_failure is None):
        if checkpoint_failure is not None:
            raise checkpoint_failure
        raise RuntimeError("checkpoint export was aborted because a peer rank failed")
    if checkpoint_failure is not None:
        raise checkpoint_failure

    trainer_state: dict[str, Any] = {}
    trainer_state_failure: BaseException | None = None
    try:
        trainer_state = trainer.state_dict()
    except BaseException as error:
        trainer_state_failure = error
    if not _checkpoint_stage_agreement(strategy, trainer_state_failure is None):
        if trainer_state_failure is not None:
            raise trainer_state_failure
        raise RuntimeError("trainer-state export was aborted because a peer rank failed")
    if trainer_state_failure is not None:
        raise trainer_state_failure
    artifact_state = checkpoint_state
    if uses_ema_export:
        ema_has_updates = _checkpoint_ranks_agree_bool(
            strategy,
            ema_has_updates,
            field="EMA update state",
        )
        if ema_has_updates:
            trainable_parameters: list[Any] = []
            parameter_failure: BaseException | None = None
            try:
                trainable_parameters = _checkpoint_trainable_parameters(bundle)
                if not trainable_parameters:
                    raise ValueError(
                        "export_ema was provided but bundle has no trainable parameters",
                    )
            except BaseException as error:
                parameter_failure = error
            if not _checkpoint_stage_agreement(strategy, parameter_failure is None):
                if parameter_failure is not None:
                    raise parameter_failure
                raise RuntimeError(
                    "EMA artifact export was aborted because a peer rank "
                    "failed parameter preflight",
                )
            if parameter_failure is not None:
                raise parameter_failure
            # The raw gather above proved the immutable root/key manifest on every
            # rank. EMA swapping is rank-local, so agree on its outcome before any
            # rank enters the second FSDP gather. A peer failure rolls successful
            # ranks back instead of leaving them blocked in a mismatched collective.
            swap_failure: BaseException | None = None
            swapped = False
            try:
                export_ema.copy_ema_to(trainable_parameters, store_temp=True)
                swapped = True
            except BaseException as error:
                swap_failure = error
            all_swapped = _checkpoint_stage_agreement(strategy, swap_failure is None)
            if not all_swapped:
                if swapped:
                    export_ema.copy_temp_to(trainable_parameters)
                if swap_failure is not None:
                    raise swap_failure
                raise RuntimeError(
                    "EMA artifact export was rolled back because a peer rank "
                    "failed before the gathered export",
                )
            if swap_failure is not None:
                raise swap_failure

            export_failure: BaseException | None = None
            restore_failure: BaseException | None = None
            try:
                artifact_state = export_checkpoint(bundle)
            except BaseException as error:
                export_failure = error
            finally:
                try:
                    export_ema.copy_temp_to(trainable_parameters)
                except BaseException as error:
                    restore_failure = error
            all_exported_and_restored = _checkpoint_stage_agreement(
                strategy,
                export_failure is None and restore_failure is None,
            )
            if not all_exported_and_restored:
                if restore_failure is not None:
                    raise RuntimeError(
                        "EMA artifact export failed to restore raw training weights",
                    ) from restore_failure
                if export_failure is not None:
                    raise export_failure
                raise RuntimeError(
                    "EMA artifact export was aborted because a peer rank failed "
                    "during gather or raw-weight restoration",
                )
            if restore_failure is not None:
                raise RuntimeError(
                    "EMA artifact export failed to restore raw training weights",
                ) from restore_failure
            if export_failure is not None:
                raise export_failure
        else:
            logger.info(
                "Skipping EMA export weights because EMA has not updated; "
                "exporting raw checkpoint-owned weights.",
            )
    published_meta: dict[str, Any] = {}
    publish_failure: BaseException | None = None
    if is_primary:
        try:
            # Atomic publish: every artifact (checkpoint.pt, exports, meta) is
            # written into a same-filesystem staging directory, fsynced, and then
            # renamed into place in one os.replace. A crash leaves either the
            # previous complete checkpoint or an ignorable *.tmp-* directory.
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
                    "owned_state": checkpoint_state,
                    "identity": dict(model_identity),
                },
                "progress": dict(progress),
                "rng": rng_state or capture_rng_state(),
            }
            try:
                checkpoint_file = path / TRAINING_CHECKPOINT_NAME
                torch.save(payload, checkpoint_file)
                with checkpoint_file.open("rb") as handle:
                    os.fsync(handle.fileno())
                published_meta = _write_checkpoint_artifacts_and_publish(
                    staging=path,
                    final_path=final_path,
                    checkpoint_file=checkpoint_file,
                    family=family,
                    model_identity=model_identity,
                    progress=progress,
                    trainer_state=trainer_state,
                    artifact_state=artifact_state,
                    adapter_exports=resolved_exports,
                )
            except BaseException:
                shutil.rmtree(path, ignore_errors=True)
                raise
        except BaseException as error:
            publish_failure = error

    # Publication is part of the all-rank checkpoint protocol. Peers wait here,
    # not in a caller-owned barrier, so a rank0 IO failure becomes the same
    # explicit exception on every rank instead of stranding peers forever.
    if not _checkpoint_stage_agreement(strategy, publish_failure is None):
        if publish_failure is not None:
            raise publish_failure
        raise RuntimeError("checkpoint publication failed on the primary rank")
    if publish_failure is not None:
        raise publish_failure
    return published_meta


def _write_checkpoint_artifacts_and_publish(
    *,
    staging: Path,
    final_path: Path,
    checkpoint_file: Path,
    family: str,
    model_identity: dict[str, Any],
    progress: dict[str, Any],
    trainer_state: dict[str, Any],
    artifact_state: dict[str, Any],
    adapter_exports: Mapping[str, _ResolvedAdapterExport],
) -> dict[str, Any]:
    """Write exports + meta into ``staging``, then atomically rename into place."""

    for name, resolved in adapter_exports.items():
        root_state = artifact_state.get(resolved.root_name)
        if not isinstance(root_state, Mapping):
            raise ValueError(
                f"adapter export {name!r} checkpoint root "
                f"{resolved.root_name!r} has no gathered state",
            )
        adapter_state = _adapter_relative_state(
            root_state,
            prefix=resolved.state_prefix,
            artifact_name=name,
        )
        resolved.export.module.save_pretrained(
            staging / name,
            state_dict=adapter_state,
            selected_adapters=[resolved.export.adapter_name],
        )

    meta = write_checkpoint_meta(
        staging,
        family=family,
        model_identity=model_identity,
        trainer_state=trainer_state,
        completed_epoch=int(progress.get("completed_epoch", progress.get("next_epoch", 0))),
        next_epoch=int(progress.get("next_epoch", progress.get("next_step", 0))),
        uses_lora=any(
            name == LORA_WEIGHTS_NAME or name.startswith(f"{LORA_WEIGHTS_NAME}/")
            for name in adapter_exports
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
    schema_version = _require_checkpoint_schema_version(
        payload,
        source="checkpoint payload",
    )
    _validate_checkpoint_payload(payload, schema_version=schema_version)
    meta = read_checkpoint_meta(checkpoint_dir)
    _validate_checkpoint_meta_matches_payload(
        meta,
        payload=payload,
        schema_version=schema_version,
    )
    return TrainingCheckpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        payload=payload,
        meta=meta,
    )


def resolve_training_resume_config(cfg: Any) -> TrainingResumeConfig:
    """Resolve the public checkpoint inputs once at the config-build boundary."""

    resume_from = str(cfg_path(cfg, "trainer.resume_from", "") or "").strip()
    value = cfg_path(cfg, "trainer.resume_strict", None)
    strict = DEFAULT_CHECKPOINT_STRICT if value is None else value
    if not isinstance(strict, bool):
        raise TypeError(
            f"trainer.resume_strict must be a boolean, got {type(strict).__name__}: {strict!r}",
        )
    return TrainingResumeConfig(
        checkpoint_path=resume_from or None,
        strict=strict,
    )


def load_training_checkpoint_for_resume(
    resume: TrainingResumeConfig,
) -> TrainingCheckpoint | None:
    """Load the checkpoint selected by one resolved resume policy."""

    if resume.checkpoint_path is None:
        return None
    return load_training_checkpoint(resume.checkpoint_path)


def load_training_checkpoint_from_config(cfg: Any) -> TrainingCheckpoint | None:
    """Public convenience facade for callers without a shared build result."""

    return load_training_checkpoint_for_resume(resolve_training_resume_config(cfg))


def resolve_training_resume_strict(cfg: Any) -> bool:
    """Compatibility facade for the shared checkpoint-restore policy."""

    return resolve_training_resume_config(cfg).strict


def prepare_model_config_for_training_resume(
    cfg: Any,
    checkpoint: TrainingCheckpoint | TrainingResumeConfig | None,
    *,
    strict: bool = DEFAULT_CHECKPOINT_STRICT,
) -> None:
    """Remove warm-start adapter paths when doing full training resume.

    Full resume restores ``RuntimeBundle.trainable_modules`` from
    ``checkpoint.pt``. Loading an unrelated ``model.lora.path`` before that can
    silently alter adapter structure, so strict mode rejects the combination.
    ``TrainingResumeConfig`` lets the config-build boundary normalize before
    checkpoint I/O; accepting a loaded checkpoint preserves the public
    compatibility facade for direct callers.
    """

    if isinstance(checkpoint, TrainingResumeConfig):
        has_checkpoint = checkpoint.checkpoint_path is not None
        strict = checkpoint.strict
    else:
        has_checkpoint = checkpoint is not None
    if not has_checkpoint:
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
    strict: bool = DEFAULT_CHECKPOINT_STRICT,
) -> None:
    """Restore model checkpoint-owned modules and trainer state."""

    if checkpoint is None:
        return
    strategy = getattr(trainer, "_strategy", None)
    restore_model_checkpoint(
        checkpoint,
        bundle=bundle,
        family=family,
        expected_model_identity=expected_model_identity,
        strict=strict,
        strategy=strategy,
    )
    trainer.load_state_dict(checkpoint.trainer_state, strict=strict)
    if strict and "optimizer" in checkpoint.trainer_state:
        restored_trainer = trainer.state_dict()
        _require_equal_tensor_tree(
            checkpoint.trainer_state["optimizer"],
            restored_trainer.get("optimizer"),
            label="restored optimizer state",
        )


def restore_model_checkpoint(
    checkpoint: TrainingCheckpoint | None,
    *,
    bundle: Any,
    family: str,
    expected_model_identity: dict[str, Any] | None = None,
    strict: bool = DEFAULT_CHECKPOINT_STRICT,
    strategy: Any | None = None,
) -> None:
    """Restore and verify checkpoint-owned model state without trainer state.

    This public facade is the shared checkpoint protocol boundary for evaluation
    tools and training resume. A distributed caller passes its strategy so
    FSDP can scatter full checkpoint tensors through the same seam used by save.
    """

    if checkpoint is None:
        return
    validate_checkpoint_compatibility(
        checkpoint,
        family=family,
        expected_model_identity=expected_model_identity,
        strict=strict,
    )
    resolved_state = _checkpoint_state_for_restore(
        checkpoint,
        bundle=bundle,
        expected_model_identity=expected_model_identity,
        strict=strict,
    )
    strategy_loader = getattr(strategy, "load_checkpoint_state", None)
    full_state_loader = getattr(strategy, "load_full_checkpoint_state", None)
    if not resolved_state.full_state_roots:
        state_for_load = {
            root_name: dict(root_state) for root_name, root_state in resolved_state.state.items()
        }
        if callable(strategy_loader):
            # Model state must use the same distributed seam as save: FSDP resumes
            # full owned tensors by scattering them back into local DTensor shards.
            strategy_loader(bundle, state_for_load, strict=strict)
        else:
            load_checkpoint_state(bundle, state_for_load, strict=strict)
    else:
        modules = require_trainable_modules(bundle)
        # A schema-v1 checkpoint may mix old full-state roots with selective
        # roots. Load one validated root at a time so FSDP chooses its full-state
        # scatter only where required without weakening schema-v2 semantics.
        for root_name in sorted(resolved_state.state):
            root_bundle = _CheckpointRootBundle(
                trainable_modules={root_name: modules[root_name]},
            )
            # DCP may replace mapping leaves with local DTensors while preparing
            # a scatter. Keep the validated checkpoint mapping immutable for the
            # post-load equality check and for other consumers of the payload.
            root_state = {root_name: dict(resolved_state.state[root_name])}
            if root_name in resolved_state.full_state_roots:
                if callable(full_state_loader):
                    full_state_loader(root_bundle, root_state, strict=strict)
                else:
                    load_full_checkpoint_state(root_bundle, root_state, strict=strict)
            elif callable(strategy_loader):
                strategy_loader(root_bundle, root_state, strict=strict)
            else:
                load_checkpoint_state(root_bundle, root_state, strict=strict)
    if strict:
        restored_checkpoint_state = (
            strategy.export_checkpoint_state(bundle)
            if callable(getattr(strategy, "export_checkpoint_state", None))
            else export_checkpoint_state(bundle)
        )
        expected_owned_state = _select_owned_checkpoint_state(
            bundle,
            resolved_state.state,
        )
        _require_equal_tensor_tree(
            expected_owned_state,
            restored_checkpoint_state,
            label="restored model weights",
        )


def validate_checkpoint_compatibility(
    checkpoint: TrainingCheckpoint | None,
    *,
    family: str,
    expected_model_identity: dict[str, Any] | None = None,
    strict: bool = DEFAULT_CHECKPOINT_STRICT,
) -> None:
    """Validate family and immutable model identity before runtime construction."""

    if checkpoint is None:
        return
    _validate_checkpoint_identity_contract(
        source="checkpoint",
        schema_version=checkpoint.schema_version,
        checkpoint_family=checkpoint.payload.get("family"),
        saved_identity=checkpoint.model_identity,
        family=family,
        expected_model_identity=expected_model_identity,
        strict=strict,
    )


def validate_checkpoint_meta_compatibility(
    meta: Mapping[str, Any],
    *,
    family: str,
    expected_model_identity: dict[str, Any] | None = None,
    strict: bool = DEFAULT_CHECKPOINT_STRICT,
) -> None:
    """Validate the cheap checkpoint sidecar before model construction.

    Schema-v2 metadata carries a provenance-only copy of the immutable model
    identity. Restore still validates the authoritative ``checkpoint.pt``
    payload and requires the sidecar copy to match it.
    """

    if not isinstance(meta, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    if not meta:
        return
    schema_version = _require_checkpoint_schema_version(
        meta,
        source="checkpoint metadata",
    )
    raw_identity = meta.get("model_identity")
    if raw_identity is not None and not isinstance(raw_identity, dict):
        raise TypeError("checkpoint metadata model_identity must be a dict")
    _validate_checkpoint_identity_contract(
        source="checkpoint metadata",
        schema_version=schema_version,
        checkpoint_family=meta.get("family"),
        saved_identity=raw_identity,
        family=family,
        expected_model_identity=expected_model_identity,
        strict=strict,
    )


def _require_checkpoint_family(value: Any, *, field: str) -> str:
    """Return one canonical family name or reject a missing protocol boundary."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _require_checkpoint_schema_version(
    fields: Mapping[str, Any],
    *,
    source: str,
) -> int:
    """Read one mandatory exact-integer schema version without coercion."""

    if "schema_version" not in fields:
        raise ValueError(f"{source} is missing schema_version")
    raw_version = fields["schema_version"]
    if type(raw_version) is not int:
        raise TypeError(f"{source} schema_version must be an integer")
    if raw_version not in {_LEGACY_CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_SCHEMA_VERSION}:
        raise ValueError(
            f"unsupported {source} schema_version={raw_version}; "
            f"expected {_LEGACY_CHECKPOINT_SCHEMA_VERSION} or {CHECKPOINT_SCHEMA_VERSION}",
        )
    return raw_version


def _validate_checkpoint_identity_contract(
    *,
    source: str,
    schema_version: int,
    checkpoint_family: Any,
    saved_identity: dict[str, Any] | None,
    family: str,
    expected_model_identity: dict[str, Any] | None,
    strict: bool,
) -> None:
    """Apply one family/identity policy to payloads and metadata sidecars."""

    if strict:
        _require_checkpoint_family(family, field="runtime family")
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        _require_checkpoint_family(
            checkpoint_family,
            field=f"schema-v2 {source} family",
        )
    if strict and checkpoint_family and checkpoint_family != family:
        raise ValueError(
            f"{source} family mismatch: checkpoint={checkpoint_family!r}, runtime={family!r}",
        )
    if strict:
        if schema_version == CHECKPOINT_SCHEMA_VERSION:
            if expected_model_identity is None:
                raise ValueError(
                    f"strict schema-v2 {source} validation requires runtime model identity",
                )
            if saved_identity is None:
                raise ValueError(
                    f"schema-v2 {source} is missing required model identity",
                )
        if (
            saved_identity is not None
            and expected_model_identity is not None
            and saved_identity != expected_model_identity
        ):
            raise ValueError(
                f"{source} model identity mismatch: "
                f"checkpoint={saved_identity!r}, runtime={expected_model_identity!r}",
            )
    else:
        if checkpoint_family and checkpoint_family != family:
            logger.warning(
                "Non-strict %s validation ignores family mismatch: checkpoint=%r, runtime=%r",
                source,
                checkpoint_family,
                family,
            )
        if saved_identity is None:
            logger.warning("Non-strict %s validation has no saved model identity", source)
        elif expected_model_identity is None:
            logger.warning("Non-strict %s validation has no runtime model identity", source)
        elif saved_identity != expected_model_identity:
            logger.warning(
                "Non-strict %s validation ignores model identity mismatch: "
                "checkpoint=%r, runtime=%r",
                source,
                saved_identity,
                expected_model_identity,
            )


def _validate_checkpoint_meta_matches_payload(
    meta: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    schema_version: int,
) -> None:
    """Reject a stale/tampered sidecar instead of trusting its cheap preflight."""

    if not meta:
        return
    meta_schema = _require_checkpoint_schema_version(
        meta,
        source="checkpoint metadata",
    )
    if meta_schema != schema_version:
        raise ValueError(
            "checkpoint metadata schema_version disagrees with checkpoint.pt: "
            f"meta={meta_schema}, payload={schema_version}",
        )
    meta_family = meta.get("family")
    payload_family = payload.get("family")
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        _require_checkpoint_family(
            meta_family,
            field="schema-v2 checkpoint metadata family",
        )
    if meta_family and payload_family and meta_family != payload_family:
        raise ValueError(
            "checkpoint metadata family disagrees with checkpoint.pt: "
            f"meta={meta_family!r}, payload={payload_family!r}",
        )
    model = payload.get("model")
    payload_identity = model.get("identity") if isinstance(model, Mapping) else None
    meta_identity = meta.get("model_identity")
    if meta_identity is not None and meta_identity != payload_identity:
        raise ValueError("checkpoint metadata model_identity disagrees with checkpoint.pt")
    if schema_version == CHECKPOINT_SCHEMA_VERSION and meta_identity is None:
        raise ValueError("schema-v2 checkpoint metadata is missing model_identity")


def _validate_checkpoint_payload(
    payload: dict[str, Any],
    *,
    schema_version: int,
) -> None:
    """Validate schema-owned roots before exposing a loaded checkpoint."""

    model = payload.get("model")
    if not isinstance(model, dict):
        raise TypeError("checkpoint payload missing dict field: model")
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        _require_checkpoint_family(
            payload.get("family"),
            field="schema-v2 checkpoint payload family",
        )
        expected_model_keys = {"identity", "owned_state"}
        if set(model) != expected_model_keys:
            raise ValueError(
                "schema-v2 checkpoint model keys mismatch: "
                f"expected={sorted(expected_model_keys)}, actual={sorted(model)}",
            )
        identity = model["identity"]
        if not isinstance(identity, dict) or not identity:
            raise ValueError(
                "schema-v2 checkpoint model.identity must be a non-empty dict",
            )
        if not isinstance(model["owned_state"], dict):
            raise TypeError("checkpoint payload field model.owned_state must be a dict")
        return
    state = model.get("trainable_modules")
    if not isinstance(state, dict):
        raise TypeError("schema-v1 checkpoint missing dict field: model.trainable_modules")


def _checkpoint_state_for_restore(
    checkpoint: TrainingCheckpoint,
    *,
    bundle: Any,
    expected_model_identity: dict[str, Any] | None,
    strict: bool,
) -> _CheckpointStateForRestore:
    """Validate v1 full/selective state and v2 exact owned state before mutation."""

    modules = require_trainable_modules(bundle)
    saved_roots = checkpoint.checkpoint_state
    missing_roots = sorted(set(modules) - set(saved_roots))
    extra_roots = sorted(set(saved_roots) - set(modules))
    if strict and (missing_roots or extra_roots):
        raise ValueError(
            f"checkpoint module roots mismatch: missing={missing_roots}, unexpected={extra_roots}",
        )
    if not strict and (missing_roots or extra_roots):
        logger.warning(
            "Non-strict checkpoint restore ignores module root mismatch: "
            "missing=%s, unexpected=%s",
            missing_roots,
            extra_roots,
        )

    normalized: dict[str, dict[str, Any]] = {}
    full_state_roots: set[str] = set()
    for root_name, wrapped in modules.items():
        if root_name not in saved_roots:
            continue
        raw_saved = saved_roots[root_name]
        if not isinstance(raw_saved, dict):
            if strict:
                raise TypeError(f"checkpoint module {root_name!r} state must be a dict")
            logger.warning(
                "Non-strict checkpoint restore skips non-dict module state %r",
                root_name,
            )
            continue
        saved = _normalize_v1_compile_prefix(
            raw_saved,
            root_name=root_name,
            schema_version=checkpoint.schema_version,
            strict=strict,
        )
        module = unwrap_compile_and_ddp(wrapped)
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(f"checkpoint module {root_name!r} does not expose state_dict()")
        runtime_state = state_dict()
        known_names = frozenset(str(name) for name in runtime_state)
        owned_names = checkpoint_owned_state_names(module)
        saved_names = frozenset(saved)
        if checkpoint.schema_version == CHECKPOINT_SCHEMA_VERSION:
            format_label = "schema-v2 owned"
            permitted_names = owned_names
        elif saved_names == known_names:
            format_label = "schema-v1 full"
            permitted_names = known_names
            full_state_roots.add(root_name)
        else:
            format_label = "schema-v1 selective"
            permitted_names = owned_names
            if strict and (checkpoint.model_identity is None or expected_model_identity is None):
                raise ValueError(
                    "strict restore rejects schema-v1 selective state without "
                    "verified model identity",
                )
        missing = sorted(owned_names - saved_names)
        unexpected = sorted(saved_names - permitted_names)
        if strict and (missing or unexpected):
            raise ValueError(
                f"{format_label} checkpoint keys mismatch for {root_name!r}: "
                f"missing={missing}, unexpected={unexpected}",
            )
        if not strict and (missing or unexpected):
            logger.warning(
                "Non-strict checkpoint restore uses matching owned keys for %r: "
                "missing=%s, unexpected=%s",
                root_name,
                missing,
                unexpected,
            )
        if root_name in full_state_roots:
            # Full schema-v1 state is the old checkpoint's source of truth,
            # including frozen base tensors. Dropping those tensors would make an
            # identity-less v1 checkpoint silently depend on the current base.
            selected = {name: saved[name] for name in sorted(known_names)}
        else:
            selected = {
                name: saved[name]
                for name in sorted(owned_names & saved_names)
                if name in known_names
            }
        _validate_checkpoint_tensor_compatibility(
            root_name=root_name,
            saved_state=selected,
            runtime_state=runtime_state,
        )
        normalized[root_name] = selected
    return _CheckpointStateForRestore(
        state=normalized,
        full_state_roots=frozenset(full_state_roots),
    )


def _validate_checkpoint_tensor_compatibility(
    *,
    root_name: str,
    saved_state: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> None:
    """Reject tensor type/shape failures before any checkpoint root is mutated."""

    for name, saved_value in saved_state.items():
        runtime_value = runtime_state[name]
        if not isinstance(runtime_value, torch.Tensor):
            continue
        field = f"{root_name}.{name}"
        if not isinstance(saved_value, torch.Tensor):
            raise TypeError(
                f"checkpoint tensor {field!r} must be a tensor, got {type(saved_value).__name__}",
            )
        # DTensor.shape is the global logical shape; comparing its local shard
        # would reject valid FSDP restores. Dtype is intentionally not checked
        # because load_state_dict supports compatible dtype casts.
        checkpoint_shape = tuple(saved_value.shape)
        runtime_shape = tuple(runtime_value.shape)
        if checkpoint_shape != runtime_shape:
            raise ValueError(
                f"checkpoint tensor {field!r} shape mismatch: "
                f"checkpoint={checkpoint_shape}, runtime={runtime_shape}",
            )


def _normalize_v1_compile_prefix(
    state: dict[str, Any],
    *,
    root_name: str,
    schema_version: int,
    strict: bool,
) -> dict[str, Any]:
    """Migrate the one wrapper prefix emitted by schema-v1 compiled exports."""

    if schema_version != _LEGACY_CHECKPOINT_SCHEMA_VERSION or not state:
        return state
    prefix = "_orig_mod."
    prefixed = [str(name).startswith(prefix) for name in state]
    if not any(prefixed):
        return state
    if not all(prefixed):
        if strict:
            raise ValueError(
                f"schema-v1 checkpoint module {root_name!r} mixes compiled and "
                "uncompiled state keys",
            )
        logger.warning(
            "Non-strict checkpoint restore cannot normalize mixed schema-v1 "
            "compile prefixes for %r",
            root_name,
        )
        return state
    normalized = {str(name).removeprefix(prefix): value for name, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError(
            f"schema-v1 checkpoint module {root_name!r} has colliding compiled state keys",
        )
    return normalized


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


def export_checkpoint_state(bundle: Any) -> dict[str, dict[str, Any]]:
    """Export exact checkpoint-owned module state as detached CPU snapshots."""

    modules = require_trainable_modules(bundle)
    out: dict[str, dict[str, Any]] = {}
    for name, wrapped in modules.items():
        module = unwrap_compile_and_ddp(wrapped)
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(f"checkpoint module {name!r} does not expose state_dict()")
        module_state = state_dict()
        owned_names = checkpoint_owned_state_names(module)
        if not owned_names:
            raise ValueError(f"checkpoint module {name!r} has no checkpoint-owned state")
        missing = sorted(owned_names - set(module_state))
        if missing:
            raise ValueError(
                f"checkpoint module {name!r} state_dict() is missing owned state: {missing}",
            )
        out[name] = to_cpu_snapshot(
            {key: module_state[key] for key in sorted(owned_names)},
        )
    return out


def load_checkpoint_state(
    bundle: Any,
    state: dict[str, Any],
    *,
    strict: bool = DEFAULT_CHECKPOINT_STRICT,
) -> None:
    """Load exact checkpoint-owned state into a runtime bundle."""

    modules = require_trainable_modules(bundle)
    missing = sorted(set(modules) - set(state))
    extra = sorted(set(state) - set(modules))
    if strict and (missing or extra):
        raise ValueError(
            f"checkpoint module roots mismatch: missing={missing}, unexpected={extra}",
        )
    for name, wrapped in modules.items():
        if name not in state:
            continue
        module = unwrap_compile_and_ddp(wrapped)
        module_state = state[name]
        if not isinstance(module_state, dict):
            raise TypeError(f"checkpoint module {name!r} state must be a dict")
        owned_names = checkpoint_owned_state_names(module)
        missing_keys = sorted(owned_names - set(module_state))
        extra_keys = sorted(set(module_state) - owned_names)
        if strict and (missing_keys or extra_keys):
            raise ValueError(
                f"checkpoint owned keys mismatch for {name!r}: "
                f"missing={missing_keys}, unexpected={extra_keys}",
            )
        selected = {key: value for key, value in module_state.items() if key in owned_names}
        if not selected:
            continue
        load_state_dict = getattr(module, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError(f"checkpoint module {name!r} does not expose load_state_dict()")
        load_state_dict(selected, strict=False)


def load_full_checkpoint_state(
    bundle: Any,
    state: dict[str, Any],
    *,
    strict: bool = DEFAULT_CHECKPOINT_STRICT,
) -> None:
    """Load schema-v1 full module state, including frozen parameters and buffers."""

    modules = require_trainable_modules(bundle)
    missing = sorted(set(modules) - set(state))
    extra = sorted(set(state) - set(modules))
    if strict and (missing or extra):
        raise ValueError(
            f"checkpoint module roots mismatch: missing={missing}, unexpected={extra}",
        )

    validated: dict[str, tuple[Any, dict[str, Any]]] = {}
    for name, wrapped in modules.items():
        if name not in state:
            continue
        module = unwrap_compile_and_ddp(wrapped)
        module_state = state[name]
        if not isinstance(module_state, dict):
            raise TypeError(f"checkpoint module {name!r} state must be a dict")
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(f"checkpoint module {name!r} does not expose state_dict()")
        known_names = frozenset(str(key) for key in state_dict())
        missing_keys = sorted(known_names - set(module_state))
        extra_keys = sorted(set(module_state) - known_names)
        if strict and (missing_keys or extra_keys):
            raise ValueError(
                f"full checkpoint keys mismatch for {name!r}: "
                f"missing={missing_keys}, unexpected={extra_keys}",
            )
        validated[name] = (module, module_state)

    # Validate every root/key before the first load so a malformed full v1
    # checkpoint cannot partially mutate an earlier module.
    for name in sorted(validated):
        module, module_state = validated[name]
        load_state_dict = getattr(module, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError(f"checkpoint module {name!r} does not expose load_state_dict()")
        load_state_dict(module_state, strict=strict)


def _select_owned_checkpoint_state(
    bundle: Any,
    state: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Project validated full/selective state onto the runtime-owned key space."""

    modules = require_trainable_modules(bundle)
    selected: dict[str, dict[str, Any]] = {}
    for root_name, wrapped in modules.items():
        saved = state.get(root_name)
        if saved is None:
            continue
        module = unwrap_compile_and_ddp(wrapped)
        owned_names = checkpoint_owned_state_names(module)
        selected[root_name] = {name: saved[name] for name in sorted(owned_names) if name in saved}
    return selected


def export_trainable_state(bundle: Any) -> dict[str, dict[str, Any]]:
    """Compatibility facade; use :func:`export_checkpoint_state` for resume."""

    return export_checkpoint_state(bundle)


def load_trainable_state(
    bundle: Any,
    state: dict[str, Any],
    *,
    strict: bool = DEFAULT_CHECKPOINT_STRICT,
) -> None:
    """Compatibility facade; use :func:`load_checkpoint_state` for resume."""

    load_checkpoint_state(bundle, state, strict=strict)


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
    columns: Sequence[str],
    *,
    resume_at: tuple[str, int] | None,
) -> None:
    """Create a metrics CSV or align it atomically with a resume checkpoint.

    A checkpoint at position N cannot support metrics already written for N or
    later: those updates were not captured in the checkpoint and will be
    recomputed. Keeping them would create duplicate positions after resume.
    """

    path = Path(csv_path)
    if isinstance(columns, (str, bytes)):
        raise ValueError("metrics columns must be a sequence of column names")
    column_names = tuple(columns)
    if (
        not column_names
        or any(
            not isinstance(name, str) or not name or any(char in name for char in ",\r\n")
            for name in column_names
        )
        or len(column_names) != len(set(column_names))
    ):
        raise ValueError("metrics columns must be unique non-empty CSV-safe strings")
    normalized_header = ",".join(column_names) + "\n"
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

    if position_column not in column_names:
        raise ValueError(f"metrics CSV is missing resume column {position_column!r}: {path}")
    position_index = column_names.index(position_column)
    retained_lines: list[str] = []
    previous_position = -1
    truncated_rows = 0
    for line_number, line in enumerate(lines[1:], start=2):
        values = next(csv.reader([line]))
        if len(values) != len(column_names):
            raise ValueError(
                f"metrics CSV row {line_number} has {len(values)} columns; "
                f"expected {len(column_names)}: {path}",
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
    model_identity: dict[str, Any],
    trainer_state: dict[str, Any],
    completed_epoch: int,
    next_epoch: int,
    uses_lora: bool,
    checkpoint_file_bytes: int | None = None,
) -> dict[str, Any]:
    """Write human-readable checkpoint metadata next to ``checkpoint.pt``.

    ``checkpoint_file_bytes`` records the published size of ``checkpoint.pt``
    so completeness checks (supervisor resume discovery) can reject truncated
    copies without loading the payload. ``model_identity`` is a provenance-only
    copy for cheap pre-model preflight; ``checkpoint.pt`` remains authoritative.
    """

    _require_checkpoint_family(family, field="family")
    if not isinstance(model_identity, dict) or not model_identity:
        raise ValueError("model_identity must be a non-empty dict")
    meta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": family,
        "model_identity": dict(model_identity),
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
    "DEFAULT_CHECKPOINT_STRICT",
    "LORA_WEIGHTS_NAME",
    "TRAINING_CHECKPOINT_NAME",
    "AdapterExport",
    "TrainingCheckpoint",
    "TrainingResumeConfig",
    "capture_rng_state",
    "export_checkpoint_state",
    "export_trainable_state",
    "find_latest_complete_checkpoint",
    "infer_next_epoch",
    "is_complete_checkpoint",
    "load_checkpoint_state",
    "load_full_checkpoint_state",
    "load_trainable_state",
    "load_training_checkpoint",
    "load_training_checkpoint_for_resume",
    "load_training_checkpoint_from_config",
    "prepare_metrics_csv",
    "prepare_model_config_for_training_resume",
    "read_checkpoint_meta",
    "resolve_training_resume_config",
    "resolve_training_resume_strict",
    "restore_model_checkpoint",
    "restore_rng_state",
    "restore_training_checkpoint",
    "save_resolved_config",
    "save_training_checkpoint",
    "validate_checkpoint_compatibility",
    "validate_checkpoint_meta_compatibility",
    "write_checkpoint_meta",
]
