"""Compare image checkpoints with independently composed rewards and one curve.

Generation uses the registered full-sequence denoise model and its native step
scheduler. This is not the frozen SANA official-pipeline benchmark, a video
benchmark, or an autoregressive image-token evaluator. Generated images are
content-bound before scoring, so failed reward calls can be retried without
loading the generator. Completed reports are immutable.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import logging
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from vrl.scripts.eval.denoise_generation import (
    GeneratorRuntimeIdentity,
    ImageSampling,
    generate_images,
    seed_for,
)
from vrl.utils.artifacts import sha256_file

if TYPE_CHECKING:
    from vrl.run import ResolvedModel
    from vrl.trainers.checkpointing import CheckpointTarget
    from vrl.trainers.data import PromptExample

logger = logging.getLogger(__name__)


def load_target(value: str, *, uses_lora: bool) -> CheckpointTarget:
    """Parse one ``label=path`` (or bare path) arm and pin its checkpoint."""

    from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME, CheckpointTarget

    label, separator, raw_path = value.partition("=")
    path = Path(raw_path if separator else value).expanduser().resolve()
    if path.name == LORA_WEIGHTS_NAME:
        path = path.parent
    label = label.strip() if separator else path.name
    if (
        label == "base"
        or not label
        or any(
            not (character.isascii() and (character.isalnum() or character in "_-"))
            for character in label
        )
    ):
        raise ValueError(
            "checkpoint labels must use letters, digits, '_' or '-'; base is reserved"
        )
    target = CheckpointTarget.load(path, label=label, digest=True)
    if target.meta.get("uses_lora") is not uses_lora:
        raise ValueError(f"checkpoint uses_lora differs from the run model: {path}")
    return target


@dataclass(frozen=True, slots=True)
class EvalPrompt:
    """Selected manifest row; metadata remains available to any reward."""

    # Display/provenance-only: the position in the original input manifest.
    manifest_index: int
    example: PromptExample


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """One resolved generation and scoring contract, also bound to retries."""

    resolved_model: ResolvedModel
    targets: tuple[CheckpointTarget, ...]
    prompts: tuple[EvalPrompt, ...]
    sampling: ImageSampling
    reward: dict[str, Any]
    samples_per_prompt: int
    seed: int
    blind_seed: int
    negative_prompt: str
    reward_device: str
    tie_epsilon: float
    bootstrap_resamples: int
    # Input hashes are behavior-consumed by archive compatibility checks.
    config_sha256: str
    manifest_sha256: str
    training_reward_components: tuple[str, ...]
    runtime_identity: GeneratorRuntimeIdentity = field(
        default_factory=GeneratorRuntimeIdentity.capture,
    )

    def record(self) -> dict[str, Any]:
        return {
            "schema": "vrl.image-checkpoint-evaluation/v1",
            "model_identity": self.resolved_model.identity,
            "generator_runtime": self.runtime_identity.to_record(),
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "targets": [
                {**asdict(target), "path": str(target.path or "")} for target in self.targets
            ],
            "prompts": [
                {"manifest_index": prompt.manifest_index, **asdict(prompt.example)}
                for prompt in self.prompts
            ],
            "sampling": asdict(self.sampling),
            "negative_prompt": self.negative_prompt,
            "generation_device": str(self.resolved_model.build.device),
            "generation_dtype": str(self.resolved_model.build.parameter_dtype),
            "generation_precision": asdict(self.resolved_model.build.precision),
            "samples_per_prompt": self.samples_per_prompt,
            "seed": self.seed,
            "blind_seed": self.blind_seed,
            "reward": self.reward,
            "reward_device": self.reward_device,
            "tie_epsilon": self.tie_epsilon,
            "bootstrap_resamples": self.bootstrap_resamples,
            "training_reward_components": list(self.training_reward_components),
        }

    def cells(self) -> Iterator[dict[str, Any]]:
        """Canonical arm/prompt/sample order used by generation and retry checks."""
        for target in self.targets:
            for prompt_index, prompt in enumerate(self.prompts):
                for sample_index in range(self.samples_per_prompt):
                    yield {
                        "checkpoint_label": target.label,
                        "epoch": target.epoch,
                        "prompt_index": prompt_index,
                        "manifest_index": prompt.manifest_index,
                        "sample_index": sample_index,
                        "seed": seed_for(
                            base_seed=self.seed,
                            prompt_index=prompt_index,
                            sample_index=sample_index,
                            samples_per_prompt=self.samples_per_prompt,
                        ),
                        "prompt": prompt.example.prompt,
                        "reward_metadata": prompt.example.reward_metadata(),
                        "image_path": f"images/{target.label}/prompt{prompt_index:04d}_sample{sample_index:02d}.png",
                    }

    def blind_orders(self) -> dict[int, list[str]]:
        orders = {}
        for index in range(len(self.prompts)):
            labels = [target.label for target in self.targets]
            random.Random(f"{self.blind_seed}:{index}").shuffle(labels)
            orders[index] = labels
        return orders

    def generate(self, output_dir: Path) -> list[dict[str, Any]]:
        import torch

        from vrl.trainers.checkpointing import (
            TRAINING_CHECKPOINT_NAME,
            load_training_checkpoint,
            restore_model_checkpoint,
        )
        from vrl.utils.cuda_memory import release_cuda_memory

        if (
            self.targets[0].label != "base"
            or self.targets[0].path is not None
            or any(target.path is None for target in self.targets[1:])
        ):
            raise ValueError("generate the base arm before restoring any checkpoint")
        if GeneratorRuntimeIdentity.capture() != self.runtime_identity:
            raise ValueError("generator runtime changed after preflight")
        bundle = self.resolved_model.materialize(context="image checkpoint evaluation")
        model = bundle.model.eval()
        rows = list(self.cells())
        try:
            # Full-parameter restores overwrite the base; always generate it first.
            for target in self.targets:
                if target.path is not None:
                    if (
                        sha256_file(target.path / TRAINING_CHECKPOINT_NAME)
                        != target.checkpoint_sha256
                    ):
                        raise ValueError(f"checkpoint changed after preflight: {target.path}")
                    checkpoint = load_training_checkpoint(target.path)
                    if checkpoint.next_epoch != target.epoch:
                        raise ValueError(
                            f"checkpoint state epoch differs from metadata: {target.path}"
                        )
                    restore_model_checkpoint(
                        checkpoint,
                        bundle=bundle,
                        family=self.resolved_model.entry.family,
                        expected_model_identity=self.resolved_model.identity,
                        strict=True,
                    )
                    del checkpoint
                    gc.collect()
                with model.disable_adapter() if target.path is None else contextlib.nullcontext():
                    for row in rows:
                        if row["checkpoint_label"] != target.label:
                            continue
                        images = generate_images(
                            model,
                            prompt=row["prompt"],
                            negative_prompt=self.negative_prompt,
                            seed=row["seed"],
                            samples_per_prompt=1,
                            sampling=self.sampling,
                            torch=torch,
                        )
                        if len(images) != 1:
                            raise RuntimeError(f"expected one image, received {len(images)}")
                        path = output_dir / row["image_path"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        images[0].save(path, format="PNG")
                        row["image_sha256"] = sha256_file(path)
                        logger.info("Generated %s", row["image_path"])
                        del images
        finally:
            del model, bundle
            release_cuda_memory()
        return rows


@dataclass(frozen=True, slots=True)
class EvaluationArchive:
    """Own the generation/retry boundary and atomically published report."""

    directory: Path
    plan: EvaluationPlan

    def check_tree(self) -> None:
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError(f"evaluation output must be a real directory: {self.directory}")
        if any(path.is_symlink() for path in self.directory.rglob("*")):
            raise ValueError("evaluation output tree must not contain symlinks")

    def validate_images(self, rows: Any) -> list[dict[str, Any]]:
        from PIL import Image

        self.check_tree()
        expected = list(self.plan.cells())
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError("generated image grid is incomplete")
        for row, cell in zip(rows, expected, strict=True):
            if (
                not isinstance(row, dict)
                or {key: value for key, value in row.items() if key != "image_sha256"} != cell
            ):
                raise ValueError("generated image grid differs from the evaluation protocol")
            path = self.directory / cell["image_path"]
            if not path.is_file() or sha256_file(path) != row.get("image_sha256"):
                raise ValueError(f"generated image failed integrity check: {path}")
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (
                    self.plan.sampling.width,
                    self.plan.sampling.height,
                ):
                    raise ValueError(f"generated PNG geometry differs: {path}")
                image.verify()
        actual = {
            path.relative_to(self.directory).as_posix()
            for path in (self.directory / "images").rglob("*")
            if path.is_file()
        }
        if actual != {cell["image_path"] for cell in expected}:
            raise ValueError("generated image set is not exact")
        return rows

    def publish_generation(self, rows: list[dict[str, Any]]) -> None:
        self.validate_images(rows)
        _write_json(
            self.directory / "generation_manifest.json",
            {"protocol": self.plan.record(), "images": rows},
        )

    def load_generation(self) -> list[dict[str, Any]]:
        self.check_tree()
        path = self.directory / "generation_manifest.json"
        if not path.is_file():
            raise ValueError("output has no committed generation stage; use a fresh --output-dir")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != self.plan.record():
            raise ValueError("saved generation belongs to a different evaluation protocol")
        return self.validate_images(payload.get("images"))

    def reject_completed(self) -> None:
        report = self.directory / "report"
        if not report.exists():
            return
        self.check_tree()
        marker = json.loads((report / "evaluation_complete.json").read_text(encoding="utf-8"))
        hashes = {
            path.relative_to(report).as_posix(): sha256_file(path)
            for path in report.rglob("*")
            if path.is_file() and path.name != "evaluation_complete.json"
        }
        if marker != {"protocol": self.plan.record(), "artifacts": hashes}:
            raise ValueError("completed report failed integrity check")
        raise FileExistsError("refusing to overwrite a completed evaluation report")

    def publish_report(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        from vrl.scripts.eval.score_report import write_curve_report, write_scores

        self.load_generation()
        self.reject_completed()
        # A complete directory is the commit: readers never observe half a report,
        # and a failed reward call leaves the earlier generation stage reusable.
        with tempfile.TemporaryDirectory(prefix=".report-", dir=self.directory) as temporary:
            staging = Path(temporary)
            write_scores(rows, staging)
            summary = write_curve_report(
                rows,
                staging,
                tie_epsilon=self.plan.tie_epsilon,
                bootstrap_resamples=self.plan.bootstrap_resamples,
                seed=self.plan.seed,
            )
            _write_contact_sheets(self.plan, rows, staging)
            _write_json(
                staging / "provenance.json",
                {
                    "protocol": self.plan.record(),
                    "training_reward_overlap": sorted(
                        set(self.plan.reward["components"])
                        & set(self.plan.training_reward_components)
                    ),
                    "interpretation": "Held-out reward gains are not independent human-quality proof; review the blinded images.",
                },
            )
            hashes = {
                path.relative_to(staging).as_posix(): sha256_file(path)
                for path in staging.rglob("*")
                if path.is_file()
            }
            _write_json(
                staging / "evaluation_complete.json",
                {"protocol": self.plan.record(), "artifacts": hashes},
            )
            os.replace(staging, self.directory / "report")
        return summary


def discover_targets(
    run_dir: Path,
    *,
    checkpoint_specs: Sequence[str] = (),
    epochs: Sequence[int] = (),
    uses_lora: bool,
) -> tuple[CheckpointTarget, ...]:
    from vrl.trainers.checkpointing import CheckpointTarget, is_complete_checkpoint

    if checkpoint_specs and epochs:
        raise ValueError("--checkpoint and --epochs are mutually exclusive")
    if any(type(epoch) is not int or epoch < 0 for epoch in epochs) or len(set(epochs)) != len(
        epochs
    ):
        raise ValueError("--epochs must contain unique non-negative integers")
    if checkpoint_specs:
        targets = [load_target(value, uses_lora=uses_lora) for value in checkpoint_specs]
    else:
        by_epoch: dict[int, CheckpointTarget] = {}
        for path in sorted(run_dir.glob("checkpoint-*")):
            if not path.is_dir() or not is_complete_checkpoint(path):
                continue
            target = load_target(str(path), uses_lora=uses_lora)
            previous = by_epoch.get(target.epoch)
            if previous is None or target.label == "checkpoint-final":
                by_epoch[target.epoch] = target
        if epochs and not set(epochs) <= set(by_epoch):
            raise ValueError(
                f"missing complete checkpoint epochs: {sorted(set(epochs) - set(by_epoch))}"
            )
        targets = [by_epoch[epoch] for epoch in sorted(epochs or by_epoch)]
    if not targets or len({target.label for target in targets}) != len(targets):
        raise ValueError("select at least one checkpoint with unique labels")
    return (CheckpointTarget.base(), *targets)


def select_prompts(
    examples: list[PromptExample],
    *,
    limit: int = 0,
    per_stratum: int = 0,
    strata: Sequence[str] = (),
) -> tuple[EvalPrompt, ...]:
    if limit < 0 or per_stratum < 0 or (per_stratum > 0) != bool(strata):
        raise ValueError("use non-negative limits and specify --strata with --per-stratum")
    if limit and per_stratum:
        raise ValueError("--limit and --per-stratum are mutually exclusive")
    selected = []
    groups: dict[tuple[str, ...], int] = defaultdict(int)
    for index, example in enumerate(examples):
        if per_stratum:
            values = tuple(str(example.metadata.get(key, "")).strip() for key in strata)
            if any(not value for value in values):
                raise ValueError(f"eval prompt {index} needs metadata for strata {list(strata)}")
            groups[values] += 1
            if groups[values] > per_stratum:
                continue
        selected.append(EvalPrompt(index, example))
        if limit and len(selected) >= limit:
            break
    if any(count < per_stratum for count in groups.values()):
        raise ValueError("an evaluation stratum has fewer rows than --per-stratum")
    if not selected or any(not row.example.prompt.strip() for row in selected):
        raise ValueError("evaluation requires non-empty prompts")
    # The stepwise text-to-image path does not consume image/video conditioning.
    if any(row.example.reference_image or row.example.reference_video for row in selected):
        raise ValueError("this evaluator only supports text-conditioned image generation")
    if any(row.example.request_overrides for row in selected):
        raise ValueError(
            "per-prompt request_overrides are not supported by the fixed sampling grid"
        )
    if any(row.example.task_type not in {"", "text_to_image"} for row in selected):
        raise ValueError("evaluation prompt task_type must be text_to_image or unset")
    return tuple(selected)


def resolve_plan(args: argparse.Namespace) -> EvaluationPlan:
    from vrl.config.loading import load_config
    from vrl.config.precision import resolve_precision_policy
    from vrl.config.schema import RewardConfig, parse_config
    from vrl.models.families.registry import get_model_family_entry
    from vrl.run import resolve_model
    from vrl.scripts.eval._device import resolve_eval_device
    from vrl.trainers.checkpointing import validate_checkpoint_meta_compatibility
    from vrl.trainers.data import load_prompt_manifest

    if (
        args.samples_per_prompt < 1
        or args.seed < 0
        or args.bootstrap_resamples < 1
        or not math.isfinite(args.tie_epsilon)
        or args.tie_epsilon < 0
    ):
        raise ValueError("invalid sampling/statistics options")
    config_path = args.run_dir.expanduser().resolve() / "resolved_config.yaml"
    cfg = load_config(config_path)
    root = parse_config(cfg)
    if root.model is None:
        raise ValueError("checkpoint evaluation requires a model config")
    entry = get_model_family_entry(root.model.family)
    if (
        entry.task != "t2i"
        or entry.policy_semantics.step_kind != "denoise"
        or entry.policy_semantics.generation_regime != "full_sequence"
    ):
        raise ValueError(
            "image checkpoint evaluation requires a full-sequence denoise text-to-image family"
        )
    targets = discover_targets(
        config_path.parent,
        checkpoint_specs=args.checkpoint,
        epochs=args.epochs or (),
        uses_lora=bool(root.model.use_lora),
    )
    resolved = resolve_model(
        entry,
        root,
        resolve_eval_device(args.device),
        precision=resolve_precision_policy(root.precision),
        for_rollout=True,
    )
    for target in targets[1:]:
        validate_checkpoint_meta_compatibility(
            target.meta,
            family=entry.family,
            expected_model_identity=resolved.identity,
            strict=True,
        )

    # Only reward/data are projected from this independent policy. Its model,
    # trainer and sampling fields never replace the saved generation contract.
    policy = load_config(
        args.eval_policy_config or config_path, overrides=args.eval_policy_override
    )
    manifest = args.manifest or OmegaConf.select(policy, "data.eval_manifest", default=None)
    if not manifest:
        raise ValueError("supply --manifest or data.eval_manifest in the evaluation policy")
    manifest_path = Path(manifest).expanduser().resolve()
    raw_reward = OmegaConf.select(policy, "reward", default=None)
    reward = OmegaConf.to_container(raw_reward, resolve=True) if raw_reward is not None else None
    if not isinstance(reward, dict) or not reward.get("components"):
        raise ValueError("evaluation policy must select reward.components")
    reward = {key: reward.get(key) or {} for key in ("components", "kwargs", "inference")}
    RewardConfig.model_validate(reward)
    if any(not math.isfinite(float(weight)) for weight in reward["components"].values()):
        raise ValueError("evaluation reward weights must be finite")
    if "total" in reward["components"]:
        raise ValueError("reward component name 'total' is reserved for the weighted score")
    # Keep destinations out of the semantic policy. Evaluation owns every debug
    # and transport file it creates, never the training reward directories.
    for name, settings in reward["kwargs"].items():
        reward["kwargs"][name] = dict(settings or {})
    pending = [("reward.kwargs", reward["kwargs"])]
    while pending:
        path, settings = pending.pop()
        for key in ("debug_dir", "scored_rollout_dir", "artifact_dir"):
            settings.pop(key, None)
        # Composite rewards pass nested configs to their children. Check the
        # declared grouping contract without changing a judge's comparison mode
        # or silently carrying training rollout geometry into checkpoint eval.
        expected = settings.get("expected_group_size", 0)
        if expected and expected != len(targets):
            raise ValueError(
                f"{path}.expected_group_size must match the {len(targets)} evaluation arms"
            )
        per_call = settings.get("images_per_call", 1)
        if not isinstance(per_call, int) or per_call < 1 or 1 < per_call < len(targets):
            raise ValueError(
                f"{path}.images_per_call must be 1 or cover all {len(targets)} evaluation arms"
            )
        pending.extend(
            (f"{path}.{key}", value) for key, value in settings.items() if isinstance(value, dict)
        )
    return EvaluationPlan(
        resolved,
        targets,
        select_prompts(
            load_prompt_manifest(manifest_path),
            limit=args.limit,
            per_stratum=args.per_stratum,
            strata=args.strata,
        ),
        ImageSampling.from_config(OmegaConf.to_container(cfg.sampling, resolve=True)),
        reward,
        args.samples_per_prompt,
        args.seed,
        args.blind_seed,
        args.negative_prompt,
        str(resolve_eval_device(args.reward_device)),
        args.tie_epsilon,
        args.bootstrap_resamples,
        sha256_file(config_path),
        sha256_file(manifest_path),
        tuple(root.reward.components) if root.reward else (),
    )


async def score_images(
    plan: EvaluationPlan, rows: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from PIL import Image

    from vrl.config.builders import RewardRuntimeConfig
    from vrl.config.schema import RewardConfig
    from vrl.rewards.base import DiskArtifactRewardFunction
    from vrl.rewards.functions.registry import MultiReward, _register_builtins, get_reward
    from vrl.rewards.types import REWARD_GROUP_ID_METADATA_KEY, RewardSample
    from vrl.scripts.eval.image_statistics import (
        brightness,
        edge_energy,
        sample_diversity,
        saturation,
    )

    _register_builtins()
    kwargs = {
        name: dict(plan.reward["kwargs"].get(name) or {}) for name in plan.reward["components"]
    }
    for name, settings in kwargs.items():
        if issubclass(get_reward(name), DiskArtifactRewardFunction):
            settings["artifact_dir"] = str(output_dir / "reward_artifacts" / name)
    scorer = MultiReward.from_dict(
        plan.reward["components"],
        device=plan.reward_device,
        reward_kwargs=kwargs,
        memory_parking_required=False,
        inference_configs=RewardRuntimeConfig.from_cfg(
            RewardConfig.model_validate(plan.reward),
        ).inference_configs,
    )
    by_key = {
        (row["checkpoint_label"], row["prompt_index"], row["sample_index"]): row for row in rows
    }
    orders = plan.blind_orders()
    scored = []
    try:
        for prompt_index, prompt in enumerate(plan.prompts):
            for sample_index in range(plan.samples_per_prompt):
                batch_rows, samples, diagnostics = [], [], []
                for cell, label in enumerate(orders[prompt_index]):
                    row = by_key[label, prompt_index, sample_index]
                    with Image.open(output_dir / row["image_path"]) as image:
                        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                    samples.append(
                        RewardSample(
                            prompt=prompt.example.prompt,
                            output=torch.from_numpy(array.copy()).permute(2, 0, 1),
                            sample_id=f"p{prompt_index}-s{sample_index}-cell{cell}",
                            metadata={
                                **row["reward_metadata"],
                                REWARD_GROUP_ID_METADATA_KEY: f"p{prompt_index}-s{sample_index}",
                            },
                        )
                    )
                    batch_rows.append(
                        {
                            **row,
                            "blind_cell": cell + 1,
                            "image_path": str((output_dir / row["image_path"]).resolve()),
                        }
                    )
                    diagnostics.append(
                        {
                            "r_saturation": saturation(array),
                            "r_brightness": brightness(array),
                            "r_edge_energy": edge_energy(array),
                        }
                    )
                scores = await scorer.score_batch(samples)
                if len(scores.scores) != len(batch_rows):
                    raise ValueError("reward returned a different number of scores than images")
                for index, row in enumerate(batch_rows):
                    row["r_total"] = scores.scores[index]
                    row.update(
                        {f"r_{name}": values[index] for name, values in scores.components.items()}
                    )
                    if set(row) & set(diagnostics[index]):
                        raise ValueError("reward component names collide with image diagnostics")
                    row.update(diagnostics[index])
                    scored.append(row)
    finally:
        await scorer.shutdown()
    if plan.samples_per_prompt > 1:
        groups = defaultdict(list)
        for row in scored:
            groups[row["checkpoint_label"], row["prompt_index"]].append(row)
        for group in groups.values():
            diagnostics = {
                f"r_{name}": value
                for name, value in sample_diversity(
                    [Path(row["image_path"]) for row in group]
                ).items()
            }
            for row in group:
                if set(row) & set(diagnostics):
                    raise ValueError("reward component names collide with diversity diagnostics")
                row.update(diagnostics)
    return scored


def _write_contact_sheets(
    plan: EvaluationPlan, rows: list[dict[str, Any]], report_dir: Path
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    by_key = {
        (row["checkpoint_label"], row["prompt_index"], row["sample_index"]): row for row in rows
    }
    sheet_dir = report_dir / "contact_sheets"
    sheet_dir.mkdir()
    orders = plan.blind_orders()
    manifest = []
    for prompt_index, labels in orders.items():
        tile, header = 256, 24
        canvas = Image.new(
            "RGB", (len(labels) * tile, plan.samples_per_prompt * (tile + header)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for sample_index in range(plan.samples_per_prompt):
            for cell, label in enumerate(labels):
                row = by_key[label, prompt_index, sample_index]
                x, y = cell * tile, sample_index * (tile + header)
                draw.text((x + 4, y + 4), f"{cell + 1} / seed {row['seed']}", fill="black")
                with Image.open(row["image_path"]) as image:
                    canvas.paste(
                        ImageOps.contain(image.convert("RGB"), (tile, tile)), (x, y + header)
                    )
        filename = f"prompt{prompt_index:04d}.png"
        canvas.save(sheet_dir / filename)
        manifest.append(
            {
                "sheet": filename,
                "prompt_index": prompt_index,
                "manifest_index": plan.prompts[prompt_index].manifest_index,
                "prompt": plan.prompts[prompt_index].example.prompt,
            }
        )
    _write_json(sheet_dir / "manifest.json", manifest)
    _write_json(
        report_dir / "blind_key.json",
        {"note": "Review contact sheets before opening this key.", "orders": orders},
    )


def _write_json(path: Path, value: Any) -> None:
    """Publish a complete JSON file without exposing a truncated retry record."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="[LABEL=]PATH")
    parser.add_argument(
        "--epochs", type=lambda value: tuple(int(part) for part in value.split(",")), default=()
    )
    parser.add_argument("--eval-policy-config", default=None)
    parser.add_argument("--eval-policy-override", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strata", nargs="+", default=[])
    parser.add_argument("--per-stratum", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blind-seed", type=int, default=1)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reward-device", default="cpu")
    parser.add_argument("--tie-epsilon", type=float, default=0.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    plan = resolve_plan(args)
    if args.dry_run:
        print(json.dumps(plan.record(), indent=2, sort_keys=True))
        return
    output_dir = (
        (args.output_dir or args.run_dir / "checkpoint_evaluation").expanduser().absolute()
    )
    archive = EvaluationArchive(output_dir, plan)
    if output_dir.exists():
        rows = archive.load_generation()
        archive.reject_completed()
    else:
        output_dir.mkdir(parents=True)
        rows = plan.generate(output_dir)
        archive.publish_generation(rows)
    scored = asyncio.run(score_images(plan, rows, output_dir))
    summary = archive.publish_report(scored)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
