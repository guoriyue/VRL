"""Generate and verify reusable Anima held-out image grids."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import os
import re
import tempfile
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
class GeneratedImage:
    """One exact generated cell before reward scoring."""

    checkpoint_label: str
    epoch: int
    prompt_index: int
    # Display/provenance-only copy of the selected manifest row.
    manifest_index: int
    sample_index: int
    seed: int
    prompt: str
    # Display/provenance-only prompt strata.
    bucket: str
    prompt_style: str
    path: Path
    # Integrity binding used both by retry validation and scored sample output.
    image_sha256: str


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


def _generate_curve(
    protocol: EvaluationProtocol,
    output_dir: Path,
) -> list[GeneratedImage]:
    import torch

    from vrl.scripts.families.cosmos.anima.generate import generate_images
    from vrl.trainers.checkpointing import load_training_checkpoint, restore_model_checkpoint
    from vrl.utils.cuda_memory import release_cuda_memory

    bundle = protocol.resolved_model.materialize(context="Anima held-out evaluation")
    model = bundle.model.eval()
    generated: list[GeneratedImage] = []
    checkpoint_read = False
    try:
        for target in protocol.targets:
            if target.path is None:
                if checkpoint_read:
                    raise RuntimeError("base generation must precede every checkpoint restore")
            else:
                checkpoint = load_training_checkpoint(target.path)
                if checkpoint.next_epoch != target.epoch:
                    raise ValueError(
                        f"checkpoint progress changed during evaluation: {target.path} "
                        f"next_epoch={checkpoint.next_epoch}, expected={target.epoch}",
                    )
                restore_model_checkpoint(
                    checkpoint,
                    bundle=bundle,
                    family="cosmos-predict2-anima",
                    expected_model_identity=protocol.resolved_model.identity,
                    strict=True,
                )
                del checkpoint
                gc.collect()
                checkpoint_read = True

            arm_dir = output_dir / "images" / target.label
            arm_dir.mkdir(parents=True, exist_ok=True)
            adapter_context = (
                model.disable_adapter() if target.path is None else contextlib.nullcontext()
            )
            with adapter_context:
                for prompt in protocol.prompts:
                    for sample_index in range(SAMPLES_PER_PROMPT):
                        seed = _sample_seed(prompt.prompt_index, sample_index)
                        logger.info(
                            "Generating arm=%s prompt=%d sample=%d seed=%d",
                            target.label,
                            prompt.prompt_index,
                            sample_index,
                            seed,
                        )
                        images = generate_images(
                            model,
                            prompt=prompt.prompt,
                            negative_prompt=protocol.negative_prompt,
                            seed=seed,
                            samples_per_prompt=1,
                            sampling=protocol.sampling,
                            torch=torch,
                        )
                        if len(images) != 1:
                            raise RuntimeError(f"Anima generation returned {len(images)} images")
                        path = output_dir / _image_relative_path(
                            target.label,
                            prompt.prompt_index,
                            sample_index,
                        )
                        images[0].save(path, format="PNG")
                        generated.append(
                            GeneratedImage(
                                checkpoint_label=target.label,
                                epoch=target.epoch,
                                prompt_index=prompt.prompt_index,
                                manifest_index=prompt.manifest_index,
                                sample_index=sample_index,
                                seed=seed,
                                prompt=prompt.prompt,
                                bucket=prompt.bucket,
                                prompt_style=prompt.prompt_style,
                                path=path.resolve(),
                                image_sha256=sha256_file(path),
                            ),
                        )
                        del images
    finally:
        del model, bundle
        release_cuda_memory()
    expected = len(protocol.targets) * len(protocol.prompts) * SAMPLES_PER_PROMPT
    if len(generated) != expected:
        raise RuntimeError(f"generated image count mismatch: {len(generated)} != {expected}")
    return generated


def _sample_seed(prompt_index: int, sample_index: int) -> int:
    return BASE_SEED + prompt_index * SAMPLES_PER_PROMPT + sample_index


def _image_relative_path(label: str, prompt_index: int, sample_index: int) -> Path:
    return Path("images") / label / f"prompt{prompt_index:04d}_sample{sample_index:02d}.png"


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


def _write_generation_stage(
    protocol: EvaluationProtocol,
    generated: list[GeneratedImage],
    output_dir: Path,
) -> None:
    """Commit a complete, content-addressed image grid as the retry boundary."""

    _validate_owned_output_tree(output_dir)
    by_key = {
        (image.checkpoint_label, image.prompt_index, image.sample_index): image
        for image in generated
    }
    if len(by_key) != len(generated):
        raise ValueError("generated stage contains duplicate arm/prompt/sample cells")

    rows: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for target in protocol.targets:
        for prompt in protocol.prompts:
            for sample_index in range(SAMPLES_PER_PROMPT):
                key = (target.label, prompt.prompt_index, sample_index)
                image = by_key.get(key)
                if image is None:
                    raise ValueError(f"generated stage is missing cell {key!r}")
                expected_metadata = (
                    target.epoch,
                    prompt.manifest_index,
                    _sample_seed(prompt.prompt_index, sample_index),
                    prompt.prompt,
                    prompt.bucket,
                    prompt.prompt_style,
                )
                actual_metadata = (
                    image.epoch,
                    image.manifest_index,
                    image.seed,
                    image.prompt,
                    image.bucket,
                    image.prompt_style,
                )
                if actual_metadata != expected_metadata:
                    raise ValueError(f"generated cell {key!r} has inconsistent protocol metadata")
                relative_path = _image_relative_path(*key)
                expected_path = (output_dir / relative_path).resolve()
                if not expected_path.is_relative_to(output_dir.resolve()):
                    raise ValueError(f"generated image escapes output directory: {expected_path}")
                if image.path.resolve() != expected_path:
                    raise ValueError(
                        f"generated cell {key!r} has unexpected path: {image.path}",
                    )
                _validate_generated_png(
                    expected_path,
                    expected_sha256=image.image_sha256,
                    expected_size=(
                        protocol.sampling.width,
                        protocol.sampling.height,
                    ),
                )
                expected_paths.add(relative_path)
                rows.append(
                    {
                        "checkpoint_label": target.label,
                        "prompt_index": prompt.prompt_index,
                        "sample_index": sample_index,
                        "path": relative_path.as_posix(),
                        "sha256": image.image_sha256,
                    },
                )
    if len(rows) != len(generated):
        raise ValueError(
            f"generated stage has unexpected cells: {len(generated)} != {len(rows)}",
        )
    _validate_exact_image_set(output_dir, expected_paths)
    protocol_record = _evaluation_protocol_record(protocol)
    _write_json(
        output_dir / GENERATION_MANIFEST_NAME,
        {
            "schema": GENERATION_MANIFEST_SCHEMA,
            "protocol_sha256": _sha256_json(protocol_record),
            "protocol": protocol_record,
            "image_count": len(rows),
            "images": rows,
        },
    )


def _load_generation_stage(
    protocol: EvaluationProtocol,
    output_dir: Path,
) -> list[GeneratedImage]:
    """Load a prior image grid only after proving it matches this protocol."""

    _validate_owned_output_tree(output_dir)
    manifest_path = output_dir / GENERATION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "existing evaluation output has no complete generation stage; "
            f"use a fresh --output-dir instead of mixing partial files: {output_dir}",
        )
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") != GENERATION_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported generation manifest schema: {manifest_path}")
    current_protocol = _evaluation_protocol_record(protocol)
    stored_protocol = manifest.get("protocol")
    if not isinstance(stored_protocol, dict):
        raise TypeError(f"generation manifest protocol must be a mapping: {manifest_path}")
    stored_sha256 = str(manifest.get("protocol_sha256", ""))
    if stored_sha256 != _sha256_json(stored_protocol):
        raise ValueError(f"generation manifest protocol hash is invalid: {manifest_path}")
    if stored_sha256 != _sha256_json(current_protocol) or stored_protocol != current_protocol:
        raise ValueError(
            "existing generated images belong to a different evaluation protocol; "
            "use a fresh --output-dir",
        )

    raw_rows = manifest.get("images")
    if not isinstance(raw_rows, list):
        raise TypeError(f"generation manifest images must be a list: {manifest_path}")
    expected_count = len(protocol.targets) * len(protocol.prompts) * SAMPLES_PER_PROMPT
    if type(manifest.get("image_count")) is not int or manifest["image_count"] != expected_count:
        raise ValueError(
            f"generation manifest image_count must be {expected_count}: {manifest_path}",
        )
    if len(raw_rows) != expected_count:
        raise ValueError(
            f"generation manifest has {len(raw_rows)} rows; expected {expected_count}",
        )

    rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise TypeError(f"generation manifest image rows must be mappings: {manifest_path}")
        label = row.get("checkpoint_label")
        prompt_index = row.get("prompt_index")
        sample_index = row.get("sample_index")
        if (
            not isinstance(label, str)
            or type(prompt_index) is not int
            or type(sample_index) is not int
        ):
            raise TypeError(f"generation manifest cell identity is invalid: {row!r}")
        key = (label, prompt_index, sample_index)
        if key in rows_by_key:
            raise ValueError(f"generation manifest contains duplicate cell {key!r}")
        rows_by_key[key] = row

    generated: list[GeneratedImage] = []
    expected_paths: set[Path] = set()
    for target in protocol.targets:
        for prompt in protocol.prompts:
            for sample_index in range(SAMPLES_PER_PROMPT):
                key = (target.label, prompt.prompt_index, sample_index)
                row = rows_by_key.get(key)
                if row is None:
                    raise ValueError(f"generation manifest is missing cell {key!r}")
                relative_path = _image_relative_path(*key)
                if row.get("path") != relative_path.as_posix():
                    raise ValueError(f"generation manifest cell {key!r} has an invalid path")
                image_sha256 = row.get("sha256")
                if not _is_sha256(image_sha256):
                    raise ValueError(f"generation manifest cell {key!r} has an invalid SHA-256")
                image_path = (output_dir / relative_path).resolve()
                if not image_path.is_relative_to(output_dir.resolve()):
                    raise ValueError(f"generation image escapes output directory: {image_path}")
                _validate_generated_png(
                    image_path,
                    expected_sha256=image_sha256,
                    expected_size=(
                        protocol.sampling.width,
                        protocol.sampling.height,
                    ),
                )
                expected_paths.add(relative_path)
                generated.append(
                    GeneratedImage(
                        checkpoint_label=target.label,
                        epoch=target.epoch,
                        prompt_index=prompt.prompt_index,
                        manifest_index=prompt.manifest_index,
                        sample_index=sample_index,
                        seed=_sample_seed(prompt.prompt_index, sample_index),
                        prompt=prompt.prompt,
                        bucket=prompt.bucket,
                        prompt_style=prompt.prompt_style,
                        path=image_path,
                        image_sha256=image_sha256,
                    ),
                )
    if len(rows_by_key) != len(generated):
        raise ValueError("generation manifest contains cells outside the registered grid")
    _validate_exact_image_set(output_dir, expected_paths)
    return generated


def _validate_generated_png(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: tuple[int, int],
) -> None:
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(f"generated image is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"generated image hash mismatch for {path}: {actual_sha256} != {expected_sha256}",
        )
    try:
        with Image.open(path) as image:
            size = image.size
            image_format = image.format
            image.verify()
    except Exception as exc:
        raise ValueError(f"generated image is not a valid PNG: {path}") from exc
    if size != expected_size:
        raise ValueError(f"generated image size mismatch for {path}: {size} != {expected_size}")
    if image_format != "PNG":
        raise ValueError(f"generated image format mismatch for {path}: {image_format} != PNG")


def _validate_exact_image_set(output_dir: Path, expected_paths: set[Path]) -> None:
    image_dir = output_dir / "images"
    actual_paths = {
        path.relative_to(output_dir) for path in image_dir.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(
            f"generated image set is not exact: missing={missing}, extra={extra}",
        )


def _validate_owned_output_tree(output_dir: Path) -> None:
    """Reject links or nodes that escape the evaluator-owned output tree."""

    if output_dir.is_symlink():
        raise ValueError(f"evaluation output directory must not be a symlink: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"evaluation output is not a directory: {output_dir}")
    root = output_dir.resolve()
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"evaluation output tree must not contain symlinks: {path}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"evaluation output path escapes its owner: {path}")


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
