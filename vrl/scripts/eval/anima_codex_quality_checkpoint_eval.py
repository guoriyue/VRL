"""Score and publish blinded Anima held-out quality comparisons."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from vrl.scripts.families.cosmos.anima.generation_protocol import AnimaSampling
from vrl.utils.artifacts import sha256_file

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

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
class ScoredImage:
    """A generated cell with its blinded position and diagnostics."""

    image: GeneratedImage
    blind_cell: str
    luna_score: float
    saturation: float
    brightness: float
    edge_energy: float


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


def _blind_arm_order(labels: tuple[str, ...], prompt_index: int) -> tuple[str, ...]:
    seed_material = f"{BLIND_SEED}:{prompt_index}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled)


def _score_curve(
    protocol: EvaluationProtocol,
    generated: list[GeneratedImage],
) -> tuple[list[ScoredImage], dict[int, tuple[str, ...]]]:
    from PIL import Image

    from vrl.rewards.inference import RewardInferenceArtifact
    from vrl.rewards.models.codex_image_qa import CodexImageQARewardModel
    from vrl.scripts.eval.image_statistics import brightness, edge_energy, load_image, saturation

    by_key = {
        (image.checkpoint_label, image.prompt_index, image.sample_index): image
        for image in generated
    }
    if len(by_key) != len(generated):
        raise ValueError("generated image grid contains duplicate arm/prompt/sample cells")
    labels = tuple(target.label for target in protocol.targets)
    ordered_images: list[tuple[GeneratedImage, str]] = []
    blind_orders: dict[int, tuple[str, ...]] = {}
    artifacts: list[RewardInferenceArtifact] = []
    open_images: list[Image.Image] = []
    for prompt in protocol.prompts:
        order = _blind_arm_order(labels, prompt.prompt_index)
        blind_orders[prompt.prompt_index] = order
        for sample_index in range(SAMPLES_PER_PROMPT):
            for cell_index, label in enumerate(order):
                image = by_key[(label, prompt.prompt_index, sample_index)]
                cell = chr(ord("A") + cell_index)
                with Image.open(image.path) as source:
                    media = source.convert("RGB")
                open_images.append(media)
                ordered_images.append((image, cell))
                artifacts.append(
                    RewardInferenceArtifact(
                        artifact_id=(
                            f"prompt{prompt.prompt_index:04d}-sample{sample_index:02d}-cell{cell}"
                        ),
                        sample_id=(
                            f"prompt{prompt.prompt_index:04d}-sample{sample_index:02d}-cell{cell}"
                        ),
                        path=str(image.path),
                        prompt=image.prompt,
                        media=media,
                        metadata={
                            "prompt_index": prompt.prompt_index,
                            "sample_index": sample_index,
                            "blind_cell": cell,
                        },
                    ),
                )

    scorer = CodexImageQARewardModel(protocol.luna_config)
    score_maps = scorer.score_batch(artifacts)
    if len(score_maps) != len(ordered_images):
        raise ValueError(
            f"Luna score count mismatch: {len(score_maps)} != {len(ordered_images)}",
        )
    scored: list[ScoredImage] = []
    for (image, cell), score_map in zip(ordered_images, score_maps, strict=True):
        score = float(score_map["codex_image_qa"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid Luna score for {image.path}: {score!r}")
        image_array = load_image(image.path)
        scored.append(
            ScoredImage(
                image=image,
                blind_cell=cell,
                luna_score=score,
                saturation=saturation(image_array),
                brightness=brightness(image_array),
                edge_energy=edge_energy(image_array),
            ),
        )
    del artifacts, open_images, scorer
    return scored, blind_orders


def _dual_seed_diversity(
    scored: list[ScoredImage],
) -> dict[str, dict[int, dict[str, float]]]:
    import numpy as np

    from vrl.scripts.eval.image_statistics import load_image

    grouped: dict[tuple[str, int], list[ScoredImage]] = {}
    for row in scored:
        grouped.setdefault((row.image.checkpoint_label, row.image.prompt_index), []).append(row)
    result: dict[str, dict[int, dict[str, float]]] = {}
    for (label, prompt_index), rows in grouped.items():
        rows.sort(key=lambda row: row.image.sample_index)
        if len(rows) != SAMPLES_PER_PROMPT:
            raise ValueError(
                f"dual-seed diversity needs {SAMPLES_PER_PROMPT} rows for "
                f"{label}/prompt{prompt_index}, got {len(rows)}",
            )
        left = load_image(rows[0].image.path)
        right = load_image(rows[1].image.path)
        pixel_rms = float(np.sqrt(np.mean(np.square(left - right))))
        histograms = []
        for image in (left, right):
            histogram = np.concatenate(
                [
                    np.histogram(image[..., channel], bins=32, range=(0.0, 1.0))[0]
                    for channel in range(3)
                ],
            ).astype(np.float64)
            histogram /= histogram.sum() + 1e-12
            histograms.append(histogram)
        color_hist_l2 = float(np.linalg.norm(histograms[0] - histograms[1]))
        result.setdefault(label, {})[prompt_index] = {
            "pixel_rms": pixel_rms,
            "color_hist_l2": color_hist_l2,
        }
    return result


def _summarize(
    protocol: EvaluationProtocol,
    scored: list[ScoredImage],
    diversity: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    targets = protocol.targets
    labels = tuple(target.label for target in targets)
    epoch_by_label = {target.label: target.epoch for target in targets}
    rows_by_label = {
        label: [row for row in scored if row.image.checkpoint_label == label] for label in labels
    }
    arms: dict[str, Any] = {}
    prompt_luna: dict[str, dict[int, float]] = {}
    prompt_saturation: dict[str, dict[int, float]] = {}
    prompt_brightness: dict[str, dict[int, float]] = {}
    prompt_edge: dict[str, dict[int, float]] = {}
    bucket_by_prompt: dict[int, str] = {}
    for row in scored:
        previous = bucket_by_prompt.setdefault(row.image.prompt_index, row.image.bucket)
        if previous != row.image.bucket:
            raise ValueError(f"prompt {row.image.prompt_index} changed bucket across arms")
    for label, rows in rows_by_label.items():
        prompt_luna[label] = _prompt_means(rows, "luna_score")
        prompt_saturation[label] = _prompt_means(rows, "saturation")
        prompt_brightness[label] = _prompt_means(rows, "brightness")
        prompt_edge[label] = _prompt_means(rows, "edge_energy")
        pixel_values = [value["pixel_rms"] for value in diversity[label].values()]
        hist_values = [value["color_hist_l2"] for value in diversity[label].values()]
        arms[label] = {
            "epoch": epoch_by_label[label],
            "image_count": len(rows),
            "prompt_count": len(prompt_luna[label]),
            "luna": _value_summary([row.luna_score for row in rows]),
            "saturation": _value_summary([row.saturation for row in rows]),
            "brightness": _value_summary([row.brightness for row in rows]),
            "edge_energy": _value_summary([row.edge_energy for row in rows]),
            "dual_seed_diversity": {
                "pixel_rms": _value_summary(pixel_values),
                "color_hist_l2": _value_summary(hist_values),
            },
        }

    base_label = labels[0]
    paired: dict[str, Any] = {}
    for target_index, target in enumerate(targets[1:], start=1):
        label = target.label
        paired[label] = {
            "epoch": target.epoch,
            "luna": _paired_delta(
                prompt_luna[base_label],
                prompt_luna[label],
                tie_epsilon=LUNA_TIE_EPSILON,
                bootstrap_seed=BOOTSTRAP_SEED + target_index * 10,
            ),
            "luna_by_bucket": _paired_by_bucket(
                prompt_luna[base_label],
                prompt_luna[label],
                bucket_by_prompt,
            ),
            "saturation": _paired_delta(
                prompt_saturation[base_label],
                prompt_saturation[label],
                tie_epsilon=None,
                bootstrap_seed=BOOTSTRAP_SEED + target_index * 10 + 1,
            ),
            "brightness": _paired_delta(
                prompt_brightness[base_label],
                prompt_brightness[label],
                tie_epsilon=None,
                bootstrap_seed=BOOTSTRAP_SEED + target_index * 10 + 2,
            ),
            "edge_energy": _paired_delta(
                prompt_edge[base_label],
                prompt_edge[label],
                tie_epsilon=None,
                bootstrap_seed=BOOTSTRAP_SEED + target_index * 10 + 3,
            ),
            "dual_seed_diversity": {
                metric: _paired_delta(
                    {
                        prompt_index: values[metric]
                        for prompt_index, values in diversity[base_label].items()
                    },
                    {
                        prompt_index: values[metric]
                        for prompt_index, values in diversity[label].items()
                    },
                    tie_epsilon=None,
                    bootstrap_seed=BOOTSTRAP_SEED + target_index * 10 + offset,
                )
                for offset, metric in enumerate(("pixel_rms", "color_hist_l2"), start=4)
            },
        }

    endpoint = paired[targets[-1].label]
    luna_was_training_reward = "codex_image_qa" in protocol.training_reward_components
    assessment = {
        "heldout_luna_gain_supported": endpoint["luna"]["ci95_low"] > 0.0,
        "systematic_saturation_shift_supported": (
            endpoint["saturation"]["ci95_low"] > 0.0 or endpoint["saturation"]["ci95_high"] < 0.0
        ),
        "systematic_brightness_shift_supported": (
            endpoint["brightness"]["ci95_low"] > 0.0 or endpoint["brightness"]["ci95_high"] < 0.0
        ),
        "edge_detail_regression_supported": endpoint["edge_energy"]["ci95_high"] < 0.0,
        "pixel_diversity_regression_supported": (
            endpoint["dual_seed_diversity"]["pixel_rms"]["ci95_high"] < 0.0
        ),
        "color_diversity_regression_supported": (
            endpoint["dual_seed_diversity"]["color_hist_l2"]["ci95_high"] < 0.0
        ),
        "luna_was_training_reward": luna_was_training_reward,
        "human_blind_review": "pending",
        "interpretation_limit": (
            (
                "Luna is both the training reward and this held-out judge; a gain shows "
                "held-out reward generalization, not independent human-quality proof."
            )
            if luna_was_training_reward
            else (
                "Luna was not the training reward for this run; it is a separately "
                "implemented automated held-out judge. A different scorer does not prove "
                "semantic independence because its rubric may overlap the training "
                "objective, and it is not independent human-quality proof."
            )
        ),
    }
    return {
        "schema": REPORT_SCHEMA,
        "comparison_unit": f"prompt mean across {SAMPLES_PER_PROMPT} fixed seeds",
        "bootstrap": {
            "unit": "prompt",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "luna_tie_epsilon": LUNA_TIE_EPSILON,
        "arms": arms,
        "paired_vs_base": paired,
        "endpoint_assessment": assessment,
    }


def _prompt_means(rows: list[ScoredImage], field: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.image.prompt_index, []).append(float(getattr(row, field)))
    for prompt_index, values in grouped.items():
        if len(values) != SAMPLES_PER_PROMPT:
            raise ValueError(
                f"prompt {prompt_index} has {len(values)} {field} rows; "
                f"expected {SAMPLES_PER_PROMPT}",
            )
    return {prompt_index: statistics.fmean(values) for prompt_index, values in grouped.items()}


def _value_summary(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("summary values must be non-empty and finite")
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": std,
        "stderr": std / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _paired_delta(
    base: dict[int, float],
    current: dict[int, float],
    *,
    tie_epsilon: float | None,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if base.keys() != current.keys() or not base:
        raise ValueError("paired metrics require identical non-empty prompt keys")
    deltas = [current[index] - base[index] for index in sorted(base)]
    low, high = _bootstrap_mean_ci(deltas, seed=bootstrap_seed)
    result: dict[str, Any] = {
        "prompt_count": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "ci95_low": low,
        "ci95_high": high,
    }
    if tie_epsilon is not None:
        result.update(
            {
                "wins": sum(delta > tie_epsilon for delta in deltas),
                "ties": sum(abs(delta) <= tie_epsilon for delta in deltas),
                "losses": sum(delta < -tie_epsilon for delta in deltas),
            },
        )
    return result


def _paired_by_bucket(
    base: dict[int, float],
    current: dict[int, float],
    bucket_by_prompt: dict[int, str],
) -> dict[str, dict[str, float | int]]:
    if base.keys() != current.keys() or base.keys() != bucket_by_prompt.keys():
        raise ValueError("bucket deltas require identical prompt keys")
    grouped: dict[str, list[float]] = {}
    for prompt_index in sorted(base):
        grouped.setdefault(bucket_by_prompt[prompt_index], []).append(
            current[prompt_index] - base[prompt_index],
        )
    return {bucket: _value_summary(values) for bucket, values in sorted(grouped.items())}


def _bootstrap_mean_ci(values: list[float], *, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap values must be non-empty")
    rng = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    means.sort()
    low_index = int(0.025 * (BOOTSTRAP_RESAMPLES - 1))
    high_index = int(0.975 * (BOOTSTRAP_RESAMPLES - 1))
    return means[low_index], means[high_index]


def _prepare_contact_sheet_staging(
    output_dir: Path,
    blind_orders: dict[int, tuple[str, ...]],
) -> Path:
    """Remove only this evaluator's stale PNG temps from a prior hard stop."""

    contact_dir = output_dir / "contact_sheets"
    blind_dir = contact_dir / "blind"
    staging_dir = contact_dir / ".staging"
    contact_dir.mkdir(parents=True, exist_ok=True)
    prompt_prefixes = tuple(
        f".prompt{prompt_index:04d}.png." for prompt_index in sorted(blind_orders)
    )

    def is_owned_temp(path: Path) -> bool:
        return path.name.endswith(".tmp") and path.name.startswith(prompt_prefixes)

    for directory in (blind_dir, staging_dir):
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise NotADirectoryError(f"contact-sheet path is not a directory: {directory}")
        for path in directory.iterdir():
            if is_owned_temp(path) and path.is_file():
                path.unlink()
            elif directory == staging_dir:
                raise ValueError(f"contact-sheet staging contains an unknown artifact: {path}")
    staging_dir.mkdir(exist_ok=True)
    return staging_dir


def _write_contact_sheets(
    scored: list[ScoredImage],
    blind_orders: dict[int, tuple[str, ...]],
    output_dir: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    _validate_owned_output_tree(output_dir)
    staging_dir = _prepare_contact_sheet_staging(output_dir, blind_orders)
    rows_by_key = {
        (row.image.checkpoint_label, row.image.prompt_index, row.image.sample_index): row
        for row in scored
    }
    sheet_dir = output_dir / "contact_sheets" / "blind"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    tile = 512
    header = 32
    row_label_width = 96
    gap = 2
    for prompt_index in sorted(blind_orders):
        order = blind_orders[prompt_index]
        first = rows_by_key[(order[0], prompt_index, 0)].image
        canvas = Image.new(
            "RGB",
            (
                row_label_width + len(order) * tile + (len(order) + 1) * gap,
                header + SAMPLES_PER_PROMPT * tile + (SAMPLES_PER_PROMPT + 1) * gap,
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for cell_index, _label in enumerate(order):
            cell = chr(ord("A") + cell_index)
            x = row_label_width + gap + cell_index * (tile + gap)
            draw.text((x + tile // 2 - 4, 8), cell, fill="black")
        for sample_index in range(SAMPLES_PER_PROMPT):
            y = header + gap + sample_index * (tile + gap)
            seed = _sample_seed(prompt_index, sample_index)
            draw.text((4, y + tile // 2), f"seed {seed}", fill="black")
            for cell_index, label in enumerate(order):
                row = rows_by_key[(label, prompt_index, sample_index)]
                with Image.open(row.image.path) as source:
                    image = ImageOps.fit(source.convert("RGB"), (tile, tile))
                x = row_label_width + gap + cell_index * (tile + gap)
                canvas.paste(image, (x, y))
        sheet_path = sheet_dir / f"prompt{prompt_index:04d}.png"
        _write_png_atomic(sheet_path, canvas, staging_dir=staging_dir)
        manifest_rows.append(
            {
                "prompt_index": prompt_index,
                "manifest_index": first.manifest_index,
                "prompt": first.prompt,
                "bucket": first.bucket,
                "prompt_style": first.prompt_style,
                "sheet": str(sheet_path.relative_to(output_dir)),
                "seeds": [
                    _sample_seed(prompt_index, index) for index in range(SAMPLES_PER_PROMPT)
                ],
                "cells": [chr(ord("A") + index) for index in range(len(order))],
            },
        )
        key_rows.append(
            {
                "prompt_index": prompt_index,
                "cell_to_arm": {chr(ord("A") + index): label for index, label in enumerate(order)},
            },
        )
    _write_jsonl(output_dir / "contact_sheets" / "manifest.jsonl", manifest_rows)
    _write_json(
        output_dir / "blind_key.json",
        {
            "schema_version": 1,
            "blind_seed": BLIND_SEED,
            "note": "Review contact sheets before opening this key.",
            "prompts": key_rows,
        },
    )
    staging_dir.rmdir()


def _write_samples(scored: list[ScoredImage], path: Path) -> None:
    rows = []
    for row in scored:
        image = row.image
        rows.append(
            {
                "checkpoint_label": image.checkpoint_label,
                "epoch": image.epoch,
                "prompt_index": image.prompt_index,
                "manifest_index": image.manifest_index,
                "sample_index": image.sample_index,
                "seed": image.seed,
                "prompt": image.prompt,
                "bucket": image.bucket,
                "prompt_style": image.prompt_style,
                "blind_cell": row.blind_cell,
                "image_path": str(image.path),
                "image_sha256": image.image_sha256,
                "luna_score": row.luna_score,
                "saturation": row.saturation,
                "brightness": row.brightness,
                "edge_energy": row.edge_energy,
            },
        )
    _write_jsonl(path, rows)


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


def _provenance_record(
    protocol: EvaluationProtocol,
    scored: list[ScoredImage],
    blind_orders: dict[int, tuple[str, ...]],
) -> dict[str, Any]:
    prompt_template = str(protocol.luna_config.get("prompt_template", ""))
    grid_template = str(protocol.luna_config.get("grid_prompt_template", ""))
    selected_strata = {(prompt.bucket, prompt.prompt_style) for prompt in protocol.prompts}
    prompts_per_bucket_style = len(protocol.prompts) // len(selected_strata)
    return {
        "schema": REPORT_SCHEMA,
        "run_dir": str(protocol.run_dir),
        "resolved_config": {
            "path": str(protocol.config_path),
            "sha256": protocol.config_sha256,
        },
        "evaluation_policy": {
            "source": protocol.eval_policy_source,
            "resolved_sha256": protocol.eval_policy_sha256,
        },
        "training_reward_components": list(protocol.training_reward_components),
        "model_identity": protocol.resolved_model.identity,
        "checkpoints": [
            {
                "label": target.label,
                "epoch": target.epoch,
                "path": str(target.path or ""),
                "meta": target.meta,
                "checkpoint_sha256": target.checkpoint_sha256,
            }
            for target in protocol.targets
        ],
        "heldout_manifest": {
            "path": str(protocol.eval_manifest_path),
            "sha256": protocol.eval_manifest_sha256,
            "selection": {
                "prompt_styles": sorted(
                    {prompt.prompt_style for prompt in protocol.prompts},
                ),
                "per_bucket_style": prompts_per_bucket_style,
                "selected_count": len(protocol.prompts),
                "manifest_indices": [prompt.manifest_index for prompt in protocol.prompts],
            },
        },
        "seed_grid": {
            "base_seed": BASE_SEED,
            "samples_per_prompt": SAMPLES_PER_PROMPT,
            "formula": "base_seed + prompt_index * samples_per_prompt + sample_index",
        },
        "sampling": protocol.sampling_record(),
        "luna": {
            "command": protocol.luna_config["command"],
            "images_per_call": protocol.luna_config["images_per_call"],
            "tile_size": protocol.luna_config.get("tile_size"),
            "max_concurrency": protocol.luna_config.get("max_concurrency"),
            "prompt_template_sha256": _sha256_text(prompt_template),
            "grid_prompt_template_sha256": _sha256_text(grid_template),
            "scored_rollout_dir_disabled": True,
        },
        "blind": {
            "seed": BLIND_SEED,
            "scope": "one arm permutation per prompt, shared by both seeds",
            "permutation_sha256": _sha256_text(
                json.dumps(blind_orders, sort_keys=True, separators=(",", ":")),
            ),
        },
        "outputs": {
            "image_count": len(scored),
            "generation_manifest": GENERATION_MANIFEST_NAME,
            "samples": "samples.jsonl",
            "summary": "summary.json",
            "blind_contact_sheets": "contact_sheets/blind",
            "blind_key": "blind_key.json",
            "completion_marker": COMPLETION_MARKER_NAME,
        },
        "protocol_sha256": _protocol_sha256(protocol),
    }


def _write_completion_marker(protocol: EvaluationProtocol, output_dir: Path) -> None:
    """Publish the final immutable-report marker after every report file exists."""

    _validate_owned_output_tree(output_dir)
    _load_generation_stage(protocol, output_dir)
    expected_paths = _completion_artifact_paths(protocol, output_dir)
    _validate_exact_contact_sheet_set(protocol, output_dir)
    artifact_hashes: dict[str, str] = {}
    for path in expected_paths:
        if not path.is_file():
            raise FileNotFoundError(f"completed evaluation artifact is missing: {path}")
        artifact_hashes[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    _write_json(
        output_dir / COMPLETION_MARKER_NAME,
        {
            "schema": COMPLETION_MARKER_SCHEMA,
            "protocol_sha256": _protocol_sha256(protocol),
            "artifacts": artifact_hashes,
        },
    )


def _reject_completed_output(protocol: EvaluationProtocol, output_dir: Path) -> None:
    _validate_owned_output_tree(output_dir)
    marker_path = output_dir / COMPLETION_MARKER_NAME
    if not marker_path.exists():
        return
    _load_generation_stage(protocol, output_dir)
    _validate_exact_contact_sheet_set(protocol, output_dir)
    marker = _read_json_object(marker_path)
    if marker.get("schema") != COMPLETION_MARKER_SCHEMA:
        raise ValueError(f"invalid completed evaluation marker: {marker_path}")
    artifact_hashes = marker.get("artifacts")
    if not isinstance(artifact_hashes, dict):
        raise TypeError(f"completed evaluation artifacts must be a mapping: {marker_path}")
    expected_paths = _completion_artifact_paths(protocol, output_dir)
    expected_names = {path.relative_to(output_dir).as_posix() for path in expected_paths}
    if set(artifact_hashes) != expected_names:
        raise ValueError(f"completed evaluation artifact manifest is not exact: {marker_path}")
    for relative_path, expected_sha256 in artifact_hashes.items():
        if not _is_sha256(expected_sha256):
            raise ValueError(f"completed evaluation artifact has an invalid hash: {relative_path}")
        path = output_dir / relative_path
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"completed evaluation artifact failed integrity check: {path}")
    if marker.get("protocol_sha256") != _protocol_sha256(protocol):
        raise FileExistsError(
            "refusing to overwrite a completed report from a different protocol; "
            f"use a fresh --output-dir: {output_dir}",
        )
    raise FileExistsError(f"refusing to overwrite completed evaluation output: {output_dir}")


def _completion_artifact_paths(
    protocol: EvaluationProtocol,
    output_dir: Path,
) -> list[Path]:
    paths = [
        Path(GENERATION_MANIFEST_NAME),
        Path("samples.jsonl"),
        Path("summary.json"),
        Path("provenance.json"),
        Path("blind_key.json"),
        Path("contact_sheets/manifest.jsonl"),
    ]
    paths.extend(
        Path("contact_sheets/blind") / f"prompt{prompt.prompt_index:04d}.png"
        for prompt in protocol.prompts
    )
    return [output_dir / relative_path for relative_path in paths]


def _validate_exact_contact_sheet_set(
    protocol: EvaluationProtocol,
    output_dir: Path,
) -> None:
    sheet_dir = output_dir / "contact_sheets" / "blind"
    expected_sheets = {
        sheet_dir / f"prompt{prompt.prompt_index:04d}.png" for prompt in protocol.prompts
    }
    actual_sheets = (
        {path for path in sheet_dir.rglob("*") if path.is_file()} if sheet_dir.is_dir() else set()
    )
    if actual_sheets != expected_sheets:
        missing = sorted(str(path) for path in expected_sheets - actual_sheets)
        extra = sorted(str(path) for path in actual_sheets - expected_sheets)
        raise ValueError(f"blind contact-sheet set is not exact: missing={missing}, extra={extra}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


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


def _write_png_atomic(path: Path, image: PILImage, *, staging_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=staging_dir,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        image.save(temporary_path, format="PNG")
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
