"""Shared checkpoint evaluation driver for model-family eval scripts."""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import importlib
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from vrl.config.loader import build_configs, load_config, require
from vrl.engine import FamilyPipelineRegistry, GenerationRuntime, GenerationWorker
from vrl.rewards.multi import MultiReward
from vrl.rollouts.collector import build_rollout_collector, collector_config_cls
from vrl.rollouts.families.specs import get_rollout_family_entry
from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    TrainingCheckpoint,
    load_trainable_state,
    load_training_checkpoint,
)
from vrl.trainers.data import PromptExample, load_prompt_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FamilyEvalDefinition:
    family: str
    default_config: str
    default_run_dir: str
    default_splits: tuple[str, ...] = ("train",)
    default_checkpoints: tuple[str, ...] = ("base", "final")
    default_weights: tuple[str, ...] = ("raw",)
    default_steps: tuple[int, ...] = ()
    step_override_key: str | None = "num_steps"
    description: str = "Evaluate checkpoints for one VRL model family."


@dataclass(frozen=True, slots=True)
class EvalTarget:
    name: str
    checkpoint_dir: Path | None


@dataclass(frozen=True, slots=True)
class EvalCase:
    target: EvalTarget
    weight_kind: str
    split: str
    step_value: int | None

    @property
    def label(self) -> str:
        step = "config" if self.step_value is None else f"steps_{self.step_value:03d}"
        return f"{self.target.name}/{self.weight_kind}/{self.split}/{step}"


async def run_family_eval(definition: FamilyEvalDefinition) -> None:
    args = _parse_args(definition)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    cfg, run_dir = _load_eval_config(definition, args)
    _force_local_backend(cfg)

    out_dir = _resolve_out_dir(cfg, run_dir, args.out_dir)

    examples_by_split = _load_examples_by_split(cfg, args.splits, args.max_prompts)
    targets = _resolve_targets(run_dir, args.checkpoints)
    if args.dry_run:
        logger.info(
            "Dry run OK: family=%s targets=%s splits=%s out_dir=%s",
            definition.family,
            [target.name for target in targets],
            {split: len(examples) for split, examples in examples_by_split.items()},
            out_dir,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _metric_fieldnames(cfg)
    metrics_path = out_dir / "metrics.csv"
    samples_path = out_dir / "samples.csv"
    _write_csv_header(metrics_path, fieldnames.metrics)
    _write_csv_header(samples_path, fieldnames.samples)

    for target in targets:
        logger.info("Building %s eval stack for target=%s", definition.family, target.name)
        bundle, collector, reward_fn, reference_image = _build_eval_stack(definition, cfg)
        checkpoint = _load_target_checkpoint(target)
        try:
            if checkpoint is not None:
                load_trainable_state(bundle, checkpoint.trainable_state, strict=True)
            for weight_kind in args.weights:
                if not _apply_weight_kind(bundle, checkpoint, weight_kind):
                    logger.warning("Skipping %s/%s: unavailable weights", target.name, weight_kind)
                    continue
                for split in args.splits:
                    for step_value in _step_values(definition, args.steps):
                        case = EvalCase(
                            target=target,
                            weight_kind=str(weight_kind),
                            split=str(split),
                            step_value=step_value,
                        )
                        metric_row, sample_rows = await _run_case(
                            definition,
                            case,
                            collector=collector,
                            reward_fn=reward_fn,
                            reference_image=reference_image,
                            examples=examples_by_split[str(split)],
                            cfg=cfg,
                            args=args,
                            out_dir=out_dir,
                        )
                        _append_csv(metrics_path, fieldnames.metrics, [metric_row])
                        _append_csv(samples_path, fieldnames.samples, sample_rows)
        finally:
            await collector.shutdown()
            del collector, reward_fn, bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Wrote %s eval to %s", definition.family, out_dir)


@dataclass(frozen=True, slots=True)
class _CsvFieldnames:
    metrics: list[str]
    samples: list[str]


def _parse_args(definition: FamilyEvalDefinition) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=definition.description)
    parser.add_argument("--run-dir", type=Path, default=Path(definition.default_run_dir))
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config name under configs/ or a YAML path. Defaults to "
            "RUN_DIR/resolved_config.yaml when --run-dir is set, otherwise "
            f"{definition.default_config}."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=list(definition.default_checkpoints),
        help="Checkpoint targets: base/0, final, or integer checkpoint epochs.",
    )
    parser.add_argument("--weights", nargs="+", default=list(definition.default_weights))
    parser.add_argument("--splits", nargs="+", default=list(definition.default_splits))
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, manifests, checkpoints, and output paths without loading models.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. model.path=/path/to/checkpoint",
    )
    args = parser.parse_args()
    if int(args.seeds) <= 0:
        raise ValueError("--seeds must be positive")
    return args


def _load_eval_config(
    definition: FamilyEvalDefinition,
    args: argparse.Namespace,
) -> tuple[DictConfig, Path | None]:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else None
    config_arg = args.config
    if config_arg is None and run_dir is not None:
        resolved = run_dir / "resolved_config.yaml"
        config_arg = str(resolved) if resolved.exists() else definition.default_config
    if config_arg is None:
        config_arg = definition.default_config
    cfg = load_config(config_arg, overrides=list(args.overrides or []))
    return cfg, run_dir


def _force_local_backend(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.distributed.backend = "local"
    cfg.distributed.rollout.num_workers = 1
    cfg.distributed.rollout.gpus_per_worker = 0.0
    cfg.distributed.rollout.release_after_collect = False


def _resolve_out_dir(
    cfg: DictConfig,
    run_dir: Path | None,
    arg_out_dir: Path | None,
) -> Path:
    if arg_out_dir is not None:
        return arg_out_dir.expanduser().resolve()
    root = run_dir if run_dir is not None else Path(str(require(cfg, "trainer.output_dir")))
    return (root / "eval").expanduser().resolve()


def _resolve_targets(run_dir: Path | None, raw_targets: list[str]) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for raw in raw_targets:
        text = str(raw).strip().lower()
        if text in {"base", "0", "epoch_0000"}:
            targets.append(EvalTarget(name="base", checkpoint_dir=None))
            continue
        if run_dir is None:
            raise ValueError(f"--run-dir is required for checkpoint target {raw!r}")
        if text in {"final", "checkpoint-final"}:
            checkpoint_dir = run_dir / "checkpoint-final"
            name = "final"
        else:
            epoch = int(text)
            checkpoint_dir = run_dir / f"checkpoint-{epoch}"
            name = f"checkpoint_{epoch:04d}"
        if not (checkpoint_dir / TRAINING_CHECKPOINT_NAME).exists():
            raise FileNotFoundError(f"Missing checkpoint target: {checkpoint_dir}")
        targets.append(EvalTarget(name=name, checkpoint_dir=checkpoint_dir))
    return targets


def _load_examples_by_split(
    cfg: DictConfig,
    splits: list[str],
    max_prompts: int,
) -> dict[str, list[PromptExample]]:
    out: dict[str, list[PromptExample]] = {}
    for split in splits:
        manifest = _manifest_for_split(cfg, str(split))
        examples = load_prompt_manifest(manifest)[:max_prompts]
        if not examples:
            raise ValueError(f"{split} split has no examples: {manifest}")
        out[str(split)] = examples
    return out


def _manifest_for_split(cfg: DictConfig, split: str) -> Path:
    if split in {"train", "data"}:
        return Path(str(require(cfg, "data.manifest")))
    if split in {"eval", "test", "validation", "val"}:
        eval_manifest = OmegaConf.select(cfg, "eval.manifest", default=None)
        if eval_manifest:
            return Path(str(eval_manifest))
        prompts_file = OmegaConf.select(cfg, "eval.prompts_file", default=None)
        if prompts_file:
            return Path(str(prompts_file))
    raise ValueError(
        f"Cannot resolve manifest for split={split!r}; use train or configure eval.manifest/prompts_file",
    )


def _build_eval_stack(
    definition: FamilyEvalDefinition,
    cfg: DictConfig,
) -> tuple[Any, Any, MultiReward, Any]:
    built = build_configs(cfg)
    trainer_config = built["trainer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if bool(trainer_config.bf16) else torch.float16
    torch.manual_seed(int(trainer_config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(trainer_config.seed))

    entry = get_rollout_family_entry(definition.family)
    spec_extractor = _import_from_path(entry.runtime_spec_extractor)
    runtime_builder = _import_from_path(entry.runtime_builder)
    bundle = runtime_builder(spec_extractor(cfg, device, weight_dtype))

    reward_weights, reward_kwargs = built["reward"]
    if not reward_weights:
        raise ValueError("At least one reward component must have weight > 0.")
    reward_fn = MultiReward.from_dict(
        reward_weights,
        device=str(device),
        reward_kwargs=reward_kwargs,
    )

    reference_image = _load_reference_image(cfg)
    collector_config = _collector_config_from_cfg(definition.family, cfg)
    collector = build_rollout_collector(
        definition.family,
        model=bundle.policy,
        reward_fn=reward_fn,
        config=collector_config,
        reference_image=reference_image,
    )

    executor_cls = _import_from_path(entry.executor_cls)
    executor_kwargs = dict(collector.executor_kwargs)
    executor = executor_cls(bundle.policy, **executor_kwargs)
    registry = FamilyPipelineRegistry()
    registry.register(executor)
    collector.set_runtime(GenerationRuntime(GenerationWorker(registry)))
    return bundle, collector, reward_fn, reference_image


def _collector_config_from_cfg(family: str, cfg: DictConfig) -> Any:
    cls = collector_config_cls(family)
    values: dict[str, Any] = {}
    for field in fields(cls):
        name = field.name
        found, value = _select_cfg_value(cfg, name)
        if found:
            values[name] = value
    return cls(**values)


def _select_cfg_value(cfg: DictConfig, name: str) -> tuple[bool, Any]:
    aliases = {
        "kl_reward": ("algorithm.kl_reward",),
        "sde_type": ("rollout.sde.type",),
        "sde_window_size": ("rollout.sde.window_size",),
        "sde_window_range": ("rollout.sde.window_range",),
    }
    paths = aliases.get(name, ())
    paths = (
        *paths,
        f"sampling.{name}",
        f"rollout.{name}",
        f"algorithm.{name}",
    )
    for path in paths:
        value = OmegaConf.select(cfg, path, default=None)
        if value is not None:
            if name == "sde_window_range":
                return True, tuple(value)
            return True, value
    return False, None


def _import_from_path(path: str) -> Any:
    module_path, attr = path.split(":", 1)
    return getattr(importlib.import_module(module_path), attr)


def _load_reference_image(cfg: DictConfig) -> Any:
    ref_image_path = OmegaConf.select(cfg, "model.reference_image", default=None)
    if not ref_image_path:
        return None
    from PIL import Image

    return Image.open(str(ref_image_path)).convert("RGB")


def _load_target_checkpoint(target: EvalTarget) -> TrainingCheckpoint | None:
    if target.checkpoint_dir is None:
        return None
    return load_training_checkpoint(target.checkpoint_dir)


def _apply_weight_kind(
    bundle: Any,
    checkpoint: TrainingCheckpoint | None,
    weight_kind: str,
) -> bool:
    if weight_kind == "raw":
        if checkpoint is not None:
            load_trainable_state(bundle, checkpoint.trainable_state, strict=True)
        return True
    if weight_kind != "ema":
        raise ValueError(f"Unsupported weight kind: {weight_kind!r}")
    if checkpoint is None:
        return False
    ema_state = checkpoint.trainer_state.get("ema")
    if not isinstance(ema_state, dict):
        return False
    ema_parameters = ema_state.get("ema_parameters")
    if not isinstance(ema_parameters, list):
        return False
    trainable = _trainable_parameters(bundle)
    if len(trainable) != len(ema_parameters):
        raise ValueError(
            "EMA parameter count mismatch: "
            f"trainable={len(trainable)} ema={len(ema_parameters)}",
        )
    with torch.no_grad():
        for idx, (param, ema_param) in enumerate(zip(trainable, ema_parameters, strict=True)):
            if tuple(param.shape) != tuple(ema_param.shape):
                raise ValueError(
                    f"EMA parameter shape mismatch at {idx}: "
                    f"param={tuple(param.shape)} ema={tuple(ema_param.shape)}",
                )
            param.copy_(ema_param.to(device=param.device, dtype=param.dtype))
    return True


def _trainable_parameters(bundle: Any) -> list[torch.nn.Parameter]:
    out: list[torch.nn.Parameter] = []
    for module in bundle.trainable_modules.values():
        out.extend(p for p in module.parameters() if p.requires_grad)
    return out


def _step_values(
    definition: FamilyEvalDefinition,
    raw_steps: list[int] | None,
) -> list[int | None]:
    if definition.step_override_key is None:
        return [None]
    steps = tuple(raw_steps) if raw_steps is not None else definition.default_steps
    if not steps:
        return [None]
    return [int(step) for step in steps]


async def _run_case(
    definition: FamilyEvalDefinition,
    case: EvalCase,
    *,
    collector: Any,
    reward_fn: MultiReward,
    reference_image: Any,
    examples: list[PromptExample],
    cfg: DictConfig,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_dir = out_dir / case.label
    case_dir.mkdir(parents=True, exist_ok=True)
    reward_fn.reset_components()
    base_seed = _eval_seed(cfg)
    sample_rows: list[dict[str, Any]] = []
    media_paths: list[Path] = []
    rewards: list[float] = []

    logger.info("Running %s", case.label)
    for prompt_idx, example in enumerate(examples):
        for seed_idx in range(int(args.seeds)):
            seed = base_seed + prompt_idx * int(args.seeds) + seed_idx
            request_overrides = _request_overrides(definition, case, cfg, args)
            batch = await collector.collect(
                [example.prompt],
                group_size=1,
                target_text=example.target_text,
                references=example.references,
                task_type=example.task_type,
                sample_metadata=example.metadata,
                reference_image=reference_image,
                seed=seed,
                request_overrides=request_overrides,
            )
            reward = float(batch.rewards.detach().float().cpu()[0].item())
            rewards.append(reward)
            media_path = None
            if batch.videos is not None:
                media_path = case_dir / f"{prompt_idx:03d}_seed_{seed_idx:02d}_r{reward:.3f}.png"
                _save_media_tensor(batch.videos[0], media_path)
                media_paths.append(media_path)
            sample_rows.append(
                {
                    "target": case.target.name,
                    "weight_kind": case.weight_kind,
                    "split": case.split,
                    "step_value": case.step_value if case.step_value is not None else "",
                    "prompt_index": prompt_idx,
                    "seed": seed,
                    "reward": reward,
                    "target_text": example.target_text,
                    "prompt": example.prompt,
                    "media_path": str(media_path) if media_path is not None else "",
                },
            )

    contact_sheet = case_dir / "contact_sheet.png"
    _make_contact_sheet(media_paths, contact_sheet, title=case.label)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    metric_row: dict[str, Any] = {
        "target": case.target.name,
        "weight_kind": case.weight_kind,
        "split": case.split,
        "step_value": case.step_value if case.step_value is not None else "",
        "reward_mean": float(reward_tensor.mean().item()),
        "reward_std": float(reward_tensor.std().item()) if reward_tensor.numel() > 1 else 0.0,
        "num_samples": len(rewards),
        "contact_sheet": str(contact_sheet) if media_paths else "",
    }
    for name, values in reward_fn.last_components.items():
        if values:
            metric_row[f"r_{name}"] = float(sum(values) / len(values))
    return metric_row, sample_rows


def _eval_seed(cfg: DictConfig) -> int:
    return int(
        OmegaConf.select(
            cfg,
            "eval.seed",
            default=OmegaConf.select(cfg, "trainer.seed", default=0),
        ),
    )


def _request_overrides(
    definition: FamilyEvalDefinition,
    case: EvalCase,
    cfg: DictConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if definition.step_override_key and case.step_value is not None:
        overrides[definition.step_override_key] = int(case.step_value)
    sample_batch_size = args.sample_batch_size
    if sample_batch_size is None:
        sample_batch_size = OmegaConf.select(cfg, "eval.sample_batch_size", default=None)
    if sample_batch_size is not None:
        overrides["sample_batch_size"] = int(sample_batch_size)
    eval_noise = OmegaConf.select(cfg, "eval.noise_level", default=None)
    if eval_noise is not None:
        overrides["noise_level"] = float(eval_noise)
    return overrides


def _metric_fieldnames(cfg: DictConfig) -> _CsvFieldnames:
    reward_components = list((OmegaConf.select(cfg, "reward.components", default={}) or {}).keys())
    metrics = [
        "target",
        "weight_kind",
        "split",
        "step_value",
        "reward_mean",
        "reward_std",
        "num_samples",
        *(f"r_{name}" for name in reward_components),
        "contact_sheet",
    ]
    samples = [
        "target",
        "weight_kind",
        "split",
        "step_value",
        "prompt_index",
        "seed",
        "reward",
        "target_text",
        "prompt",
        "media_path",
    ]
    return _CsvFieldnames(metrics=metrics, samples=samples)


def _write_csv_header(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(rows)


def _save_media_tensor(tensor: Any, path: Path) -> None:
    from PIL import Image

    array = tensor.detach().float().cpu().clamp(0, 1)
    if array.ndim == 4:
        # [C, T, H, W] -> middle frame [C, H, W].
        array = array[:, array.shape[1] // 2]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0)
    pixels = (array.numpy() * 255).astype("uint8")
    Image.fromarray(pixels.squeeze()).save(path)


def _make_contact_sheet(image_paths: list[Path], out_path: Path, *, title: str) -> None:
    if not image_paths:
        return
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_w, thumb_h = 256, 256
    title_h = 36
    label_h = 24
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, title_h + rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill=(20, 20, 20), font=font)
    for idx, image in enumerate(images):
        image.thumbnail((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = title_h + (idx // cols) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb_h + 5), image_paths[idx].stem[:32], fill=(20, 20, 20), font=font)
    sheet.save(out_path)


def main_sync(definition: FamilyEvalDefinition) -> None:
    asyncio.run(run_family_eval(definition))


__all__ = ["FamilyEvalDefinition", "main_sync", "run_family_eval"]
