"""Resolve reproducible Anima held-out evaluation inputs and preflight records."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from vrl.scripts.families.cosmos.anima.generation_protocol import AnimaSampling
from vrl.utils.artifacts import sha256_file

if TYPE_CHECKING:
    from vrl.run import ResolvedModel
    from vrl.trainers.data import PromptExample

logger = logging.getLogger(__name__)

# Registered evaluation protocol. These are fixed comparison inputs, not
# workflow defaults: changing one creates a different, incomparable curve.
CURVE_EPOCHS = (5, 10, 15, 20)
PROMPTS_PER_BUCKET_STYLE = 2
SUPPORTED_PROMPT_STYLES = frozenset(("language", "tag", "natural_language"))
SAMPLES_PER_PROMPT = 2
BASE_SEED = 20260826
BLIND_SEED = 20260827
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_RESAMPLES = 10_000
LUNA_TIE_EPSILON = 0.02
REPORT_SCHEMA = "vrl.anima-codex-quality-curve/v1"
GENERATION_MANIFEST_SCHEMA = "vrl.anima-codex-quality-generation/v1"
COMPLETION_MARKER_SCHEMA = "vrl.anima-codex-quality-complete/v1"
GENERATION_MANIFEST_NAME = "generation_manifest.json"
COMPLETION_MARKER_NAME = "evaluation_complete.json"


@dataclass(frozen=True, slots=True)
class CheckpointTarget:
    """One base/checkpoint arm in the resolved evaluation comparison."""

    label: str
    epoch: int
    path: Path | None
    # Also binds retry manifests to the exact checkpoint compatibility record.
    meta: dict[str, Any]
    checkpoint_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvalPrompt:
    """One deterministically selected held-out prompt."""

    prompt_index: int
    # Display/provenance-only: selection order, not a runtime dataset lookup.
    manifest_index: int
    prompt: str
    # Display/provenance-only dimensions of the balanced prompt protocol.
    bucket: str
    prompt_style: str


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Resolved run-owned inputs computed once and consumed by every phase."""

    run_dir: Path
    # Display/provenance-only source paths; their contents were already parsed.
    config_path: Path
    eval_manifest_path: Path
    config_sha256: str
    eval_manifest_sha256: str
    eval_policy_source: str
    eval_policy_sha256: str
    resolved_model: ResolvedModel
    targets: tuple[CheckpointTarget, ...]
    prompts: tuple[EvalPrompt, ...]
    training_reward_components: tuple[str, ...]
    sampling: AnimaSampling
    negative_prompt: str
    luna_config: dict[str, Any]

    def sampling_record(self) -> dict[str, int | float | str]:
        """Serialize the generation policy without merging it into the typed core."""

        return {**self.sampling.to_record(), "negative_prompt": self.negative_prompt}


@dataclass(frozen=True, slots=True)
class ResolvedEvaluationPolicy:
    """Held-out prompt and judge policy projected from one config source."""

    source: str
    sha256: str
    eval_manifest_path: Path
    eval_manifest_sha256: str
    luna_config: dict[str, Any]


def _validate_curve_epochs(epochs: tuple[int, ...]) -> tuple[int, ...]:
    if not epochs:
        raise ValueError("epochs must select at least one checkpoint")
    registered_prefix = CURVE_EPOCHS[: len(epochs)]
    if epochs != registered_prefix:
        raise ValueError(
            "epochs must be a prefix of the registered 5,10,15,20 curve; "
            f"got {','.join(str(epoch) for epoch in epochs)}",
        )
    return epochs


def _resolve_protocol(
    run_dir: Path,
    *,
    device_name: str,
    curve_epochs: tuple[int, ...] = CURVE_EPOCHS,
    checkpoint_specs: tuple[str, ...] = (),
    prompts_per_bucket_style: int = PROMPTS_PER_BUCKET_STYLE,
    eval_policy_config: str | None = None,
    eval_policy_overrides: tuple[str, ...] = (),
) -> EvaluationProtocol:
    from vrl.config.loading import load_config
    from vrl.config.precision import resolve_precision_policy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry
    from vrl.run import resolve_model
    from vrl.scripts.eval._device import resolve_eval_device
    from vrl.trainers.checkpointing import validate_checkpoint_meta_compatibility
    from vrl.trainers.data import load_prompt_manifest

    resolved_run_dir = run_dir.expanduser().resolve()
    config_path = resolved_run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"training run has no resolved config: {config_path}")
    cfg = load_config(config_path)
    root = parse_config(cfg)
    if root.model is None or root.trainer is None:
        raise ValueError("Anima checkpoint evaluation requires model and trainer config")
    if root.reward is None:
        raise ValueError("Anima checkpoint evaluation requires the training reward config")
    if str(root.model.family) != "cosmos-predict2-anima":
        raise ValueError(f"expected cosmos-predict2-anima, got {root.model.family!r}")
    if checkpoint_specs:
        if curve_epochs != CURVE_EPOCHS:
            raise ValueError("--checkpoint cannot be combined with a non-default --epochs")
        targets = _discover_explicit_targets(
            checkpoint_specs,
            expected_uses_lora=bool(root.model.use_lora),
        )
    else:
        if bool(root.model.use_lora):
            raise ValueError(
                "the registered 5/10/15/20 curve requires model.use_lora=false; "
                "select LoRA checkpoints explicitly with --checkpoint",
            )
        if int(root.trainer.total_epochs or 0) != CURVE_EPOCHS[-1]:
            raise ValueError(
                f"registered curve requires {CURVE_EPOCHS[-1]} updates, "
                f"got trainer.total_epochs={root.trainer.total_epochs!r}",
            )
        if int(root.trainer.save_freq or 0) != CURVE_EPOCHS[0]:
            raise ValueError(
                f"registered curve requires save_freq={CURVE_EPOCHS[0]}, "
                f"got {root.trainer.save_freq!r}",
            )
        targets = _discover_targets(resolved_run_dir, curve_epochs=curve_epochs)

    eval_policy = _resolve_evaluation_policy(
        cfg,
        run_config_path=config_path,
        policy_config=eval_policy_config,
        policy_overrides=eval_policy_overrides,
        arm_count=len(targets),
    )
    prompts = _select_balanced_prompts(
        load_prompt_manifest(eval_policy.eval_manifest_path),
        prompts_per_bucket_style=prompts_per_bucket_style,
    )

    precision = resolve_precision_policy(root)
    device = resolve_eval_device(device_name)
    entry = get_model_family_entry(str(root.model.family))
    resolved_model = resolve_model(
        entry,
        root,
        device,
        precision=precision,
        for_rollout=True,
    )
    for target in targets:
        if target.path is None:
            continue
        validate_checkpoint_meta_compatibility(
            target.meta,
            family=entry.family,
            expected_model_identity=resolved_model.identity,
            strict=True,
        )

    return EvaluationProtocol(
        run_dir=resolved_run_dir,
        config_path=config_path,
        eval_manifest_path=eval_policy.eval_manifest_path,
        config_sha256=sha256_file(config_path),
        eval_manifest_sha256=eval_policy.eval_manifest_sha256,
        eval_policy_source=eval_policy.source,
        eval_policy_sha256=eval_policy.sha256,
        resolved_model=resolved_model,
        targets=targets,
        prompts=prompts,
        training_reward_components=tuple(
            sorted(name for name, weight in root.reward.components.items() if float(weight) > 0.0),
        ),
        sampling=_resolve_sampling(cfg),
        # The public Anima sampling schema has no global negative-prompt knob;
        # this fixed evaluation protocol uses the original empty conditioning.
        negative_prompt="",
        luna_config=eval_policy.luna_config,
    )


def _discover_targets(
    run_dir: Path,
    *,
    curve_epochs: tuple[int, ...] = CURVE_EPOCHS,
) -> tuple[CheckpointTarget, ...]:
    epochs = _validate_curve_epochs(curve_epochs)
    targets = [
        CheckpointTarget(
            label="base",
            epoch=0,
            path=None,
            meta={},
            checkpoint_sha256=None,
        ),
    ]
    for epoch in epochs:
        checkpoint_name = (
            "checkpoint-final" if epoch == CURVE_EPOCHS[-1] else f"checkpoint-{epoch}"
        )
        path = (run_dir / checkpoint_name).resolve()
        targets.append(_checkpoint_target(path, epoch=epoch, expected_uses_lora=False))
    return tuple(targets)


def _discover_explicit_targets(
    values: tuple[str, ...],
    *,
    expected_uses_lora: bool,
) -> tuple[CheckpointTarget, ...]:
    """Resolve explicit CLI arms to their checkpoint source-of-truth roots."""

    targets = [
        CheckpointTarget(
            label="base",
            epoch=0,
            path=None,
            meta={},
            checkpoint_sha256=None,
        ),
    ]
    labels: list[str] = []
    for value in values:
        label, path = _parse_checkpoint_spec(value)
        if label == "base":
            raise ValueError("'base' is reserved for the adapter-disabled arm")
        labels.append(label)
        target = _checkpoint_target(
            path,
            epoch=None,
            expected_uses_lora=expected_uses_lora,
        )
        targets.append(
            CheckpointTarget(
                label=label,
                epoch=target.epoch,
                path=target.path,
                meta=target.meta,
                checkpoint_sha256=target.checkpoint_sha256,
            ),
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"checkpoint labels must be unique: {labels}")
    return tuple(targets)


def _parse_checkpoint_spec(value: str) -> tuple[str, Path]:
    from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME

    text = str(value).strip()
    if not text:
        raise ValueError("--checkpoint values must be non-empty")
    if "=" in text:
        raw_label, raw_path = text.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
    else:
        path = Path(text).expanduser().resolve()
        checkpoint_path = path.parent if path.name == LORA_WEIGHTS_NAME else path
        raw_label = checkpoint_path.name
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_label.strip()).strip("._-")
    if not label:
        raise ValueError(f"checkpoint label resolved empty for {value!r}")
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    return label, path


def _checkpoint_target(
    path: Path,
    *,
    epoch: int | None,
    expected_uses_lora: bool = False,
) -> CheckpointTarget:
    from vrl.trainers.checkpointing import (
        LORA_WEIGHTS_NAME,
        TRAINING_CHECKPOINT_NAME,
        is_complete_checkpoint,
        read_checkpoint_meta,
    )

    checkpoint_path = path.parent if path.name == LORA_WEIGHTS_NAME else path
    if not is_complete_checkpoint(checkpoint_path):
        epoch_label = "requested arm" if epoch is None else f"epoch {epoch}"
        raise ValueError(
            f"missing or incomplete checkpoint for {epoch_label}: {checkpoint_path}",
        )
    meta = dict(read_checkpoint_meta(checkpoint_path))
    completed_epoch = int(meta.get("completed_epoch", -1))
    if epoch is not None and completed_epoch != epoch:
        raise ValueError(
            f"checkpoint epoch mismatch for {checkpoint_path}: "
            f"completed_epoch={meta.get('completed_epoch')!r}, expected={epoch}",
        )
    if meta.get("uses_lora") is not expected_uses_lora:
        raise ValueError(
            f"checkpoint uses_lora={meta.get('uses_lora')!r} does not match "
            f"resolved model.use_lora={expected_uses_lora}: {checkpoint_path}",
        )
    checkpoint_file = checkpoint_path / TRAINING_CHECKPOINT_NAME
    logger.info("Hashing checkpoint for evaluation protocol binding: %s", checkpoint_file)
    return CheckpointTarget(
        label=checkpoint_path.name,
        epoch=completed_epoch,
        path=checkpoint_path,
        meta=meta,
        checkpoint_sha256=sha256_file(checkpoint_file),
    )


def _select_balanced_prompts(
    examples: list[PromptExample],
    *,
    prompts_per_bucket_style: int = PROMPTS_PER_BUCKET_STYLE,
) -> tuple[EvalPrompt, ...]:
    if prompts_per_bucket_style < 1:
        raise ValueError("prompts_per_bucket_style must be >= 1")
    available: dict[tuple[str, str], list[tuple[int, PromptExample]]] = {}
    for manifest_index, example in enumerate(examples):
        metadata = dict(getattr(example, "metadata", {}) or {})
        bucket = str(metadata.get("bucket", "")).strip()
        prompt_style = str(metadata.get("prompt_style", "")).strip()
        if not bucket or prompt_style not in SUPPORTED_PROMPT_STYLES:
            raise ValueError(
                f"eval prompt {manifest_index} needs bucket and prompt_style in "
                f"{sorted(SUPPORTED_PROMPT_STYLES)}",
            )
        available.setdefault((bucket, prompt_style), []).append((manifest_index, example))

    buckets = sorted({bucket for bucket, _style in available})
    if not buckets:
        raise ValueError("evaluation manifest has no prompt buckets")
    prompt_styles = sorted({style for _bucket, style in available})
    selected: list[tuple[int, PromptExample, str, str]] = []
    for bucket in buckets:
        for prompt_style in prompt_styles:
            rows = available.get((bucket, prompt_style), [])
            if len(rows) < prompts_per_bucket_style:
                raise ValueError(
                    f"eval bucket/style {bucket}/{prompt_style} has {len(rows)} rows; "
                    f"needs {prompts_per_bucket_style}",
                )
            selected.extend(
                (manifest_index, example, bucket, prompt_style)
                for manifest_index, example in rows[:prompts_per_bucket_style]
            )
    selected.sort(key=lambda row: row[0])
    prompts = tuple(
        EvalPrompt(
            prompt_index=prompt_index,
            manifest_index=manifest_index,
            prompt=str(example.prompt),
            bucket=bucket,
            prompt_style=prompt_style,
        )
        for prompt_index, (manifest_index, example, bucket, prompt_style) in enumerate(selected)
    )
    if len({prompt.prompt for prompt in prompts}) != len(prompts):
        raise ValueError("selected held-out prompts must be text-unique for Luna grouping")
    return prompts


def _resolve_sampling(cfg: DictConfig) -> AnimaSampling:
    """Use normal inference with the run's resolution, step count, and CFG."""

    return AnimaSampling(
        width=int(OmegaConf.select(cfg, "sampling.width", default=512)),
        height=int(OmegaConf.select(cfg, "sampling.height", default=512)),
        num_steps=int(OmegaConf.select(cfg, "sampling.num_steps", default=20)),
        guidance_scale=float(
            OmegaConf.select(cfg, "sampling.guidance_scale", default=4.5),
        ),
        max_sequence_length=int(
            OmegaConf.select(cfg, "sampling.max_sequence_length", default=128),
        ),
    )


def _resolve_evaluation_policy(
    run_cfg: DictConfig,
    *,
    run_config_path: Path,
    policy_config: str | Path | None,
    arm_count: int,
    policy_overrides: tuple[str, ...] = (),
) -> ResolvedEvaluationPolicy:
    """Project only held-out data and Luna policy from the selected config."""

    from vrl.config.loading import load_config

    if policy_config is None:
        source = str(run_config_path)
        policy_cfg = (
            load_config(run_config_path, overrides=list(policy_overrides))
            if policy_overrides
            else run_cfg
        )
    else:
        source = str(policy_config).strip()
        if not source:
            raise ValueError("eval policy config must be non-empty")
        # A relative string is deliberately a bundled logical name. Custom
        # config files must be absolute, matching load_config's public contract.
        policy_cfg = load_config(source, overrides=list(policy_overrides))

    if policy_overrides:
        source = json.dumps({"config": source, "overrides": list(policy_overrides)})

    eval_manifest = str(
        OmegaConf.select(policy_cfg, "data.eval_manifest", default="") or "",
    ).strip()
    if not eval_manifest:
        raise ValueError(f"evaluation policy {source!r} is missing data.eval_manifest")
    eval_manifest_path = Path(eval_manifest).expanduser().resolve()
    eval_manifest_sha256 = sha256_file(eval_manifest_path)
    luna_config = _resolve_luna_config(policy_cfg, arm_count=arm_count)
    sha256 = _sha256_json(
        {
            "eval_manifest": eval_manifest,
            "eval_manifest_sha256": eval_manifest_sha256,
            "luna_config": luna_config,
        },
    )
    return ResolvedEvaluationPolicy(
        source=source,
        sha256=sha256,
        eval_manifest_path=eval_manifest_path,
        eval_manifest_sha256=eval_manifest_sha256,
        luna_config=luna_config,
    )


def _resolve_luna_config(cfg: DictConfig, *, arm_count: int) -> dict[str, Any]:
    raw = OmegaConf.select(cfg, "reward.kwargs.codex_image_qa", default=None)
    if raw is None:
        raise ValueError("resolved config has no codex_image_qa reward configuration")
    plain = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(plain, dict):
        raise TypeError("reward.kwargs.codex_image_qa must be a mapping")
    worker_config = dict(plain)
    command = [str(part) for part in worker_config.get("command", [])]
    if "gpt-5.6-luna" not in command:
        raise ValueError("held-out evaluator requires the pinned gpt-5.6-luna reward command")
    # Training owns its scored_rollouts directory. Evaluation persists exact
    # images and contact sheets under its own output schema instead.
    worker_config.pop("scored_rollout_dir", None)
    worker_config["images_per_call"] = arm_count
    worker_config["scored_rollout_dir"] = ""
    return worker_config


def _evaluation_protocol_record(protocol: EvaluationProtocol) -> dict[str, Any]:
    """Return the exact generation/scoring contract bound to persisted stages."""

    return {
        "report_schema": REPORT_SCHEMA,
        "run_dir": str(protocol.run_dir),
        "resolved_config_sha256": protocol.config_sha256,
        "evaluation_policy": {
            "source": protocol.eval_policy_source,
            "resolved_sha256": protocol.eval_policy_sha256,
        },
        "eval_manifest_sha256": protocol.eval_manifest_sha256,
        "training_reward_components": list(protocol.training_reward_components),
        "model_identity": protocol.resolved_model.identity,
        # Persisted schema compatibility: older recoverable generation stages
        # and completed archives include this redundant curve projection.
        "curve_epochs": [target.epoch for target in protocol.targets[1:]],
        "targets": [
            {
                "label": target.label,
                "epoch": target.epoch,
                "path": str(target.path or ""),
                "meta": target.meta,
                "checkpoint_sha256": target.checkpoint_sha256,
            }
            for target in protocol.targets
        ],
        "prompts": [
            {
                "prompt_index": prompt.prompt_index,
                "manifest_index": prompt.manifest_index,
                "prompt": prompt.prompt,
                "bucket": prompt.bucket,
                "prompt_style": prompt.prompt_style,
            }
            for prompt in protocol.prompts
        ],
        "sampling": protocol.sampling_record(),
        "generation": {
            "base_seed": BASE_SEED,
            "samples_per_prompt": SAMPLES_PER_PROMPT,
        },
        "scoring": {
            "blind_seed": BLIND_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "luna_tie_epsilon": LUNA_TIE_EPSILON,
            "luna_config": protocol.luna_config,
        },
    }


def _protocol_sha256(protocol: EvaluationProtocol) -> str:
    return _sha256_json(_evaluation_protocol_record(protocol))


def _preflight_record(protocol: EvaluationProtocol) -> dict[str, Any]:
    return {
        "run_dir": str(protocol.run_dir),
        "config": str(protocol.config_path),
        "evaluation_policy": {
            "source": protocol.eval_policy_source,
            "resolved_sha256": protocol.eval_policy_sha256,
        },
        "eval_manifest": str(protocol.eval_manifest_path),
        "protocol_sha256": _protocol_sha256(protocol),
        "targets": [
            {
                "label": target.label,
                "epoch": target.epoch,
                "path": str(target.path or ""),
                "checkpoint_sha256": target.checkpoint_sha256,
            }
            for target in protocol.targets
        ],
        "prompt_count": len(protocol.prompts),
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "image_count": len(protocol.targets) * len(protocol.prompts) * SAMPLES_PER_PROMPT,
        "sampling": protocol.sampling_record(),
        "model_identity": protocol.resolved_model.identity,
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
