"""Validation and required-access helpers for merged training configs.

Structural validation flows through parse_config() -> RootConfig (schema.py).
This module owns the gates that read the parsed root: the torch.compile
compatibility matrix, the rollout drift guard, and the production Kling gate
whose file-existence and manifest checks must not enter the Pydantic schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import (
    RewardConfig,
    RootConfig,
    _extract_error_message,
    parse_config,
)


def validate_reward_config(cfg: DictConfig) -> RewardConfig:
    """Validate reward component shape and model-backed reward kwargs."""
    if "reward" not in cfg:
        raise ValueError("config missing required field: reward")
    reward_raw = OmegaConf.to_container(cfg.reward, resolve=True, throw_on_missing=True) or {}
    try:
        return RewardConfig.model_validate(reward_raw)
    except ValidationError as exc:
        raise ValueError(_extract_error_message(exc, section="reward")) from exc


def validate_production_kling_video_reward_config(root: RootConfig) -> None:
    """Production Kling VideoReward gate: structural contract + path existence."""
    validate_production_reward_contract(root)
    data = root.data
    for name in ("manifest", "eval_manifest", "source_report"):
        value = str((getattr(data, name) if data is not None else None) or "").strip()
        if not value:
            raise ValueError(f"config missing required field: data.{name}")
        if not Path(value).exists():
            raise ValueError(f"data.{name} does not exist: {value}")
    assert data is not None
    task_type = str(data.task_type or "")
    if task_type == "video2world":
        _validate_video_world_production_data(root)
    if task_type == "image_to_video":
        _validate_image_to_video_production_data(root)


def validate_production_reward_contract(root: RootConfig) -> None:
    """Structural production contract for the Kling VideoReward.

    Reads the reward kwargs mapping directly — per-reward config knowledge
    deliberately does not live in the schema (rewards own their contracts at
    construction; this gate exists because production misconfiguration is
    unrecoverable mid-run).
    """
    reward = root.reward
    vr_kwargs = (reward.kwargs.get("kling_video_reward") if reward is not None else None) or {}
    if str(vr_kwargs.get("media_type", "")) != "video":
        raise ValueError(
            "production.kling_video_reward requires "
            "reward.kwargs.kling_video_reward.media_type=video"
        )
    if str(vr_kwargs.get("artifact_format", "")) != "mp4":
        raise ValueError("production.kling_video_reward requires artifact_format=mp4")
    if not str(vr_kwargs.get("reward_name", "")).strip():
        raise ValueError(
            "production.kling_video_reward requires reward.kwargs.kling_video_reward.reward_name"
        )
    worker_config = vr_kwargs.get("worker_config") or {}
    from vrl.rewards.functions.kling_video_reward import (
        PRODUCTION_LOCKED_WORKER_CONFIG_KEYS,
    )

    forbidden = sorted(k for k in PRODUCTION_LOCKED_WORKER_CONFIG_KEYS if k in worker_config)
    if forbidden:
        raise ValueError(
            "production.kling_video_reward worker_config should name the reward "
            "model directly; "
            f"remove extra loader fields: {', '.join(forbidden)}",
        )
    task_type = str((root.data.task_type if root.data is not None else None) or "")
    if task_type not in {"text_to_video", "image_to_video", "video2world"}:
        raise ValueError(
            "production.kling_video_reward requires "
            "data.task_type=text_to_video, image_to_video, or video2world"
        )


def _validate_video_world_production_data(root: RootConfig) -> None:
    from vrl.trainers.data.artifacts import validate_source_backed_video_world_manifest_pair

    data = root.data
    assert data is not None
    data_root = str(data.artifact_data_root or "").strip()
    kwargs = {"data_root": data_root} if data_root else {}
    reward_components = root.reward.components if root.reward is not None else {}
    # The target-clip-reading reward is target_dino_similarity (successor to the deleted
    # pixel-L1 target_video_similarity); it consumes metadata['target_video'], so its
    # presence is what makes target clips a hard manifest requirement.
    require_target_video = "target_dino_similarity" in reward_components
    validate_source_backed_video_world_manifest_pair(
        str(data.manifest),
        str(data.eval_manifest),
        require_target_video=require_target_video,
        **kwargs,
    )
    _validate_video_world_source_report(Path(str(data.source_report)))


def _validate_video_world_source_report(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "dataset",
        "source",
        "repo_id",
        "source_split",
        "decode_method",
        "train_rows",
        "eval_rows",
        "train_manifest",
        "eval_manifest",
        "reference_dir",
        "validation_summary",
    }
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise ValueError(
            f"data.source_report is missing Video2World provenance fields: {missing}",
        )
    if int(payload.get("train_rows") or 0) <= 0 or int(payload.get("eval_rows") or 0) <= 0:
        raise ValueError("data.source_report must record non-empty train and eval rows")
    validation_summary = payload.get("validation_summary")
    if not isinstance(validation_summary, dict) or not validation_summary:
        raise ValueError("data.source_report must include a non-empty validation_summary")


def _validate_image_to_video_production_data(root: RootConfig) -> None:
    data = root.data
    assert data is not None
    data_root = str(data.artifact_data_root or "").strip()
    if not data_root:
        raise ValueError("config missing required field: data.artifact_data_root")
    preprocessing = data.preprocessing
    image_field = str((preprocessing.image_field if preprocessing else None) or "image")
    caption_field = str((preprocessing.caption_field if preprocessing else None) or "caption")
    train_count = _validate_image_to_video_manifest(
        Path(str(data.manifest)),
        data_root=Path(data_root),
        image_field=image_field,
        caption_field=caption_field,
    )
    eval_count = _validate_image_to_video_manifest(
        Path(str(data.eval_manifest)),
        data_root=Path(data_root),
        image_field=image_field,
        caption_field=caption_field,
    )
    _validate_image_to_video_source_report(
        Path(str(data.source_report)),
        train_count=train_count,
        eval_count=eval_count,
    )


def _validate_image_to_video_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
    image_field: str,
    caption_field: str,
) -> int:
    from vrl.utils.artifacts import resolve_artifact_path

    # Shares the {source_repo, source_frame_index, decode_method, conditioning}
    # provenance sub-vocabulary with
    # vrl/utils/artifacts.py SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS —
    # keep in sync. This is a separate schema (Image2Video manifest rows use
    # source_video_url, not source_video), so the two are not unified.
    required_metadata = {
        "source_repo",
        "source_video_url",
        "source_frame_index",
        "decode_method",
        "conditioning",
    }
    row_count = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}: row {row_index} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}: row {row_index} must be an object")
            image = str(row.get(image_field, "")).strip()
            caption = str(row.get(caption_field, "")).strip()
            if not image:
                raise ValueError(f"{manifest_path}: row {row_index} missing {image_field}")
            if not caption:
                raise ValueError(f"{manifest_path}: row {row_index} missing {caption_field}")
            resolved_image = resolve_artifact_path(image, data_root=data_root)
            if not resolved_image.exists():
                raise ValueError(
                    f"{manifest_path}: row {row_index} image does not exist: {resolved_image}",
                )
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"{manifest_path}: row {row_index} metadata is required")
            missing = sorted(
                field
                for field in required_metadata
                if metadata.get(field) is None or str(metadata.get(field)).strip() == ""
            )
            if missing:
                raise ValueError(
                    f"{manifest_path}: row {row_index} missing source metadata: {missing}",
                )
    if row_count == 0:
        raise ValueError(f"{manifest_path} must contain at least one image-to-video row")
    return row_count


def _validate_image_to_video_source_report(
    path: Path,
    *,
    train_count: int,
    eval_count: int,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Report-level (dataset-wide) schema — distinct from the per-row provenance
    # vocabulary in _validate_image_to_video_manifest and
    # vrl/utils/artifacts.py SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS.
    # Do NOT fold this into those: these keys describe the whole source dump
    # (source_csv, train_rows, reference_dir), not a single manifest row.
    required_keys = {
        "dataset",
        "source_repo",
        "source_csv",
        "source_split",
        "decode_method",
        "train_rows",
        "eval_rows",
        "train_manifest",
        "eval_manifest",
        "reference_dir",
    }
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise ValueError(
            f"data.source_report is missing Image2Video provenance fields: {missing}",
        )
    if payload.get("dataset") != "videophy_i2v":
        raise ValueError("data.source_report dataset must be videophy_i2v")
    if int(payload.get("train_rows") or 0) != train_count:
        raise ValueError("data.source_report train_rows does not match data.manifest")
    if int(payload.get("eval_rows") or 0) != eval_count:
        raise ValueError("data.source_report eval_rows does not match data.eval_manifest")


def validate_training_config(cfg: DictConfig) -> tuple[RootConfig, PrecisionPolicy]:
    """Validate config once and return the typed config plus its precision policy.

    Returns the pair rather than a wrapper struct: the precision policy is a pure
    derivation of ``root``, and it is returned only so the caller does not resolve
    it a second time (asserted by tests/config/test_builders.py).
    """
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    # Every torch.compile incompatibility at once. Checked at config load — where
    # the all-experiments test sees it — because a model-layer
    # torch_compile.enable=true default can silently flip compile on underneath a
    # recipe that needs checkpointing, FSDP, or a multi-rank engine.
    require_compile_compatible(root)
    require_guarded_rollout_drift(root, precision)
    if root.production is not None and root.production.kling_video_reward.enabled:
        validate_production_kling_video_reward_config(root)
    return root, precision


def compile_conflicts(root: RootConfig) -> tuple[str, ...]:
    """Every feature this config turns on that cannot coexist with torch.compile.

    ONE home for the compile compatibility matrix. Each of these was discovered
    separately and used to be enforced somewhere different — grad-checkpointing
    in the trainer, FSDP2 in the strategy builder, sequence parallelism nowhere
    at all — so adding the fifth meant first finding the other four. They are
    all decided by config keys, so config load is where they belong: a
    combination that can never run should fail before a GPU is touched.

    Returns one message per conflict (empty when compatible). The blockwise-fp8
    conflict is deliberately NOT here: it is caught in ``vrl.models.loader``
    where the resolved quantization recipe lives, and this function is given
    only the parsed config.
    """

    compile_block = root.model.torch_compile if root.model is not None else None
    if compile_block is None or not bool(compile_block.enable):
        return ()

    # Each conflict below binds one build role, so the matrix honors
    # ``model.torch_compile.scope``: trainer constraints cannot veto a
    # rollout-only compile, and rollout constraints cannot veto a replay-only
    # one. The (block, role) decision is owned by the typed build contract.
    from vrl.models.interfaces.runtime import torch_compile_for_role

    compiles_replay = torch_compile_for_role(compile_block, "replay") is not None
    compiles_rollout = torch_compile_for_role(compile_block, "rollout") is not None

    conflicts: list[str] = []

    # torch.compile traces torch.utils.checkpoint into an InternalTorchDynamoError
    # (measured for full and selective alike), and inductor's min-cut partitioner
    # already does automatic selective recompute.
    from vrl.trainers.activation_checkpointing import resolve_gradient_checkpointing_mode

    checkpointing = resolve_gradient_checkpointing_mode(root)
    if compiles_replay and checkpointing != "off":
        conflicts.append(
            f"actor.gradient_checkpointing={checkpointing!r}: torch.compile traces "
            "torch.utils.checkpoint into an InternalTorchDynamoError, and its "
            "min-cut partitioner already does automatic selective recompute. Pick "
            "one — compile alone (preferred when it fits memory), eager + "
            "checkpointing, or model.torch_compile.scope=rollout to keep the "
            "trainer eager while the rollout policy compiles.",
        )

    distributed = root.distributed
    training = None if distributed is None else distributed.training
    strategy = "single_process" if training is None else str(training.strategy)
    # Inductor graph capture is unsound with FSDP2's reshard-after-forward
    # all-gathers. Previously only caught when the strategy was built, which is
    # after config load, so a bad recipe surfaced later than it needed to.
    if compiles_replay and strategy == "fsdp":
        conflicts.append(
            "distributed.training.strategy=fsdp: torch.compile (inductor graph "
            "capture) is unsound with FSDP2 fully_shard's reshard-after-forward "
            "all-gathers. model.torch_compile.scope=rollout keeps the FSDP2 "
            "replay policy eager while the rollout policy compiles.",
        )

    # Sequence parallelism is installed by the rollout WORKER, after the family
    # builder has already compiled the policy core: the installer swaps every
    # attention processor and registers forward hooks on the first/last block,
    # mutating a module inductor has already traced. sd3_5 declares BOTH
    # supports_torch_compile and a sequence_parallel_installer, so this is
    # reachable from config -- and it had no gate at all before.
    resources = None if distributed is None else distributed.resources
    gpus_per_engine = 1 if resources is None else int(resources.rollout.gpus_per_engine)
    if compiles_rollout and gpus_per_engine > 1:
        conflicts.append(
            f"distributed.resources.rollout.gpus_per_engine={gpus_per_engine}: "
            "sequence parallelism installs attention processors and forward hooks "
            "on the policy core AFTER the model is built and compiled, mutating "
            "the module torch.compile already traced.",
        )

    return tuple(conflicts)


def require_compile_compatible(root: RootConfig) -> None:
    """Refuse a config that enables torch.compile beside an incompatible feature."""

    conflicts = compile_conflicts(root)
    if not conflicts:
        return
    joined = "\n  - ".join(conflicts)
    raise ValueError(
        f"model.torch_compile.enable=true cannot combine with:\n  - {joined}",
    )


def require_guarded_rollout_drift(root: RootConfig, precision: PrecisionPolicy) -> None:
    """Refuse a rollout approximation that no drift correction will cover.

    Quantization needs no check here: it changes the rollout precision label, so
    ``stages_match`` goes False and the trainer already installs TIS correction
    plus a drift guard whose default ``mode="auto"`` resolves to ``"fail"``.

    A request-scoped approximation is the uncovered case. TeaCache reuses a
    cached ``noise_pred`` on skipped denoise steps, so the collection-time
    log-prob stops matching the trainer's exact replay forward -- while BOTH
    roles keep the same precision label, leaving every automatic correction off.
    Silently, the run would train on uncorrected off-policy gradients and still
    report convergence.
    """

    from vrl.nn.optimization import unguarded_drift_sources

    sampling = root.sampling
    sources = unguarded_drift_sources(
        sampling.model_dump(mode="python", exclude_none=True) if sampling is not None else None,
        precision,
    )
    if not sources:
        return
    # The same escape hatch the precision-split path honors: an explicit expert
    # block means the user has chosen the correction policy deliberately.
    explicit = set() if root.trainer is None else root.trainer.model_fields_set
    if "precision_drift_guard" in explicit or "precision_correction" in explicit:
        return
    raise ValueError(
        f"sampling enables {', '.join(sources)}, which makes the rollout log-probs "
        "diverge from the trainer's exact replay forward, but rollout and training "
        "precision are identical so no drift guard or importance-sampling "
        "correction is armed. Set an explicit trainer.precision_correction / "
        "trainer.precision_drift_guard for this run, or disable the optimization.",
    )


__all__ = [
    "validate_production_kling_video_reward_config",
    "validate_production_reward_contract",
    "validate_reward_config",
    "validate_training_config",
]
