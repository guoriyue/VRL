"""Compare image checkpoints with independently composed rewards and one curve.

Generation uses the registered full-sequence denoise model and its native step
scheduler. This is not the frozen SANA official-pipeline benchmark or a video
benchmark. Generated images are content-bound before scoring, so failed reward calls can be retried without
loading the generator. Completed reports are immutable.
Local rewards run in-process after generation; configured HTTP judges stay remote.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from vrl.scripts.eval.denoise_generation import (
    ImageSampling,
    generate_images,
    seed_for,
)
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import write_json
from vrl.utils.paths import RootedPaths, user_path

if TYPE_CHECKING:
    from vrl.models.families.registry import ModelFamilyEntry
    from vrl.models.interfaces import ModelBuild
    from vrl.trainers.checkpointing import CheckpointTarget
    from vrl.trainers.data.prompts import PromptExample

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalPrompt:
    """Selected manifest row; metadata remains available to any reward."""

    # Display/provenance-only: the position in the original input manifest.
    manifest_index: int
    example: PromptExample


@dataclass(frozen=True, slots=True)
class CheckpointEvaluation:
    """Everything one checkpoint evaluation generates and scores with, bound to retries."""

    entry: ModelFamilyEntry
    model_build: ModelBuild
    model_identity: dict[str, Any]
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

    def record(self) -> dict[str, Any]:
        return {
            "schema": "vrl.image-checkpoint-evaluation/v1",
            "model_identity": self.model_identity,
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "targets": [
                {**asdict(target), "path": str(target.path or "")} for target in self.targets
            ],
            "prompts": [
                {"manifest_index": prompt.manifest_index, **asdict(prompt.example)}
                for prompt in self.prompts
            ],
            "sampling": self.sampling.to_record(),
            "negative_prompt": self.negative_prompt,
            "generation_device": str(self.model_build.device),
            "generation_dtype": str(self.model_build.parameter_dtype),
            "generation_precision": asdict(self.model_build.precision),
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
                    # Reward metadata, plus the reference images an edit reward
                    # compares against.
                    reward_metadata = prompt.example.reward_metadata()
                    if prompt.example.reference_images:
                        reward_metadata["reference_images"] = list(prompt.example.reference_images)
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
                        "reward_metadata": reward_metadata,
                        "image_path": f"images/{target.label}/prompt{prompt_index:04d}_sample{sample_index:02d}.png",
                    }

    def _generate_edit(
        self, model: Any, example: PromptExample, seed: int
    ) -> tuple[list[Any], str]:
        """One reference-conditioned edit through the family's own batch executor.

        The executor is the rollout path (reference loading, conditioning,
        packing), run with the native deterministic scheduler instead of the
        SDE. The hash of the initial latent tensor proves arms were paired on
        identical noise, not just on a seed label.
        """

        import hashlib

        from vrl.generation.execution.sample_batches import GenerationSampleBatch
        from vrl.generation.steps.denoise.config import DenoiseRequestOptions
        from vrl.generation.types import GenerationRequest
        from vrl.utils.config import import_from_path
        from vrl.utils.media import to_pil_image

        entry = self.entry
        executor = import_from_path(entry.executor_cls)(model, gatherer=entry.new_gatherer())
        request = GenerationRequest(
            request_id=f"eval-{seed}",
            family=entry.family,
            task=entry.task,
            inputs=[example.generation_input()],
            samples_per_prompt=1,
            # A key the family declares but left unset (max_sequence_length is
            # None for Qwen-Image-2.1) is not a request value.
            sampling={
                **{k: v for k, v in self.sampling.to_record().items() if v is not None},
                "seed": seed,
            },
            denoise=DenoiseRequestOptions(denoise_mode="native"),
        )
        batch = executor.forward_batch(
            request, GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1)
        )
        initial = batch.observations[:, 0].detach().cpu().contiguous()
        latent_hash = hashlib.sha256(initial.float().numpy().tobytes()).hexdigest()
        generated = executor.merge_generation_batches(request, [batch])
        return [to_pil_image(generated.output[0])], latent_hash

    def blind_orders(self) -> dict[int, list[str]]:
        orders = {}
        for index in range(len(self.prompts)):
            labels = [target.label for target in self.targets]
            random.Random(f"{self.blind_seed}:{index}").shuffle(labels)
            orders[index] = labels
        return orders

    def generate(self, output_dir: Path) -> list[dict[str, Any]]:
        from vrl.trainers.checkpointing import TrainingCheckpoint
        from vrl.utils.cuda_memory import release_cuda_memory

        if (
            self.targets[0].label != "base"
            or self.targets[0].path is not None
            or any(target.path is None for target in self.targets[1:])
        ):
            raise ValueError("generate the base arm before restoring any checkpoint")
        bundle = self.entry.build_rollout(self.model_build, expected_identity=self.model_identity)
        model = bundle.model.eval()
        rows = list(self.cells())
        try:
            # Full-parameter restores overwrite the base; always generate it first.
            for target in self.targets:
                if target.path is not None:
                    checkpoint = TrainingCheckpoint.load(target.path)
                    if checkpoint.next_epoch != target.epoch:
                        raise ValueError(
                            f"checkpoint state epoch differs from metadata: {target.path}"
                        )
                    checkpoint.restore_model(
                        bundle=bundle,
                        family=self.entry.family,
                        expected_model_identity=self.model_identity,
                    )
                    del checkpoint
                    gc.collect()
                with model.disable_adapter() if target.path is None else contextlib.nullcontext():
                    for row in rows:
                        if row["checkpoint_label"] != target.label:
                            continue
                        example = self.prompts[row["prompt_index"]].example
                        if example.reference_images:
                            images, latent_hash = self._generate_edit(model, example, row["seed"])
                            row["initial_latents_sha256"] = latent_hash
                        else:
                            images = generate_images(
                                model,
                                prompt=row["prompt"],
                                negative_prompt=self.negative_prompt,
                                seed=row["seed"],
                                samples_per_prompt=1,
                                sampling=self.sampling,
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

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> CheckpointEvaluation:
        """Resolve the run, checkpoints, prompts, sampling and reward named on the CLI."""

        from vrl.config.loading import load_config
        from vrl.config.precision import PrecisionPolicy
        from vrl.config.schema import RewardConfig
        from vrl.models.families.registry import get_model_family_entry
        from vrl.scripts.eval._device import resolve_eval_device
        from vrl.trainers.checkpointing import CheckpointTarget
        from vrl.trainers.data.prompts import load_prompt_dataset_index

        if (
            any(
                value is not None and value < 1
                for value in (args.width, args.height, args.num_steps)
            )
            or args.samples_per_prompt < 1
            or args.seed < 0
            or args.bootstrap_resamples < 1
            or not math.isfinite(args.tie_epsilon)
            or args.tie_epsilon < 0
        ):
            raise ValueError("invalid sampling/statistics options")
        from vrl.trainers.checkpointing import RESOLVED_CONFIG_NAME, load_resolved_run_config

        config_path = user_path(args.run_dir) / RESOLVED_CONFIG_NAME
        _, root = load_resolved_run_config(args.run_dir)
        if root.model is None:
            raise ValueError("checkpoint evaluation requires a model config")
        entry = get_model_family_entry(root.model.family)
        if entry.task != "t2i" or entry.generation_regime != "full_sequence":
            raise ValueError(
                "image checkpoint evaluation requires a full-sequence denoise text-to-image family"
            )
        if args.base_only:
            # A baseline (e.g. checking a reward against a benchmark before any
            # training): the untrained model is the only arm.
            if args.checkpoint or args.epochs:
                raise ValueError("--base-only takes no --checkpoint or --epochs")
            targets = (CheckpointTarget.base(),)
        else:
            targets = discover_targets(
                config_path.parent,
                checkpoint_specs=args.checkpoint,
                epochs=args.epochs or (),
                uses_lora=bool(root.model.use_lora),
            )
        model_build = entry.resolve_model_build(
            root,
            resolve_eval_device(args.device),
            precision=PrecisionPolicy.from_section(root.precision),
            for_rollout=True,
        )
        model_identity = model_build.checkpoint_identity()
        for target in targets[1:]:
            target.validate_compatibility(
                family=entry.family,
                expected_model_identity=model_identity,
            )

        # Only reward/data are projected from this independent policy. Its model,
        # trainer and sampling fields never replace the saved generation contract.
        policy = load_config(
            args.eval_policy_config or config_path, overrides=args.eval_policy_override
        )
        manifest = args.manifest or OmegaConf.select(policy, "data.eval_manifest", default=None)
        if not manifest:
            raise ValueError("supply --manifest or data.eval_manifest in the evaluation policy")
        manifest_path = user_path(manifest)
        raw_reward = OmegaConf.select(policy, "reward", default=None)
        reward = (
            OmegaConf.to_container(raw_reward, resolve=True) if raw_reward is not None else None
        )
        if not isinstance(reward, dict) or not reward.get("components"):
            raise ValueError("evaluation policy must select reward.components")
        reward = {key: reward.get(key) or {} for key in ("components", "kwargs", "inference")}
        RewardConfig.model_validate(reward)
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
                (f"{path}.{key}", value)
                for key, value in settings.items()
                if isinstance(value, dict)
            )
        roots = RootedPaths.data_root(allow_absolute=True)
        prompts = select_prompts(
            [
                example.with_resolved_references(roots)
                for example in load_prompt_dataset_index(manifest_path)
            ],
            limit=args.limit,
            per_stratum=args.per_stratum,
            strata=args.strata,
        )
        if any(prompt.example.reference_images for prompt in prompts):
            from vrl.generation.bindings.full_sequence import ReferenceConditionedBatches
            from vrl.utils.config import import_from_path

            if not issubclass(import_from_path(entry.executor_cls), ReferenceConditionedBatches):
                raise ValueError(f"{entry.family} does not take reference images")
        return cls(
            entry,
            model_build,
            model_identity,
            targets,
            prompts,
            # Only geometry and step count may differ from training; the guidance
            # scale and prompt length stay the run's own.
            ImageSampling.from_root(
                root,
                overrides={
                    "width": args.width,
                    "height": args.height,
                    "num_steps": args.num_steps,
                },
            ),
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


@dataclass(frozen=True, slots=True)
class EvaluationArchive:
    """Own the generation/retry boundary and atomically published report."""

    directory: Path
    evaluation: CheckpointEvaluation

    def check_tree(self) -> None:
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError(f"evaluation output must be a real directory: {self.directory}")
        if any(path.is_symlink() for path in self.directory.rglob("*")):
            raise ValueError("evaluation output tree must not contain symlinks")

    def validate_images(self, rows: Any) -> list[dict[str, Any]]:
        from PIL import Image

        self.check_tree()
        expected = list(self.evaluation.cells())
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError("generated image grid is incomplete")
        for row, cell in zip(rows, expected, strict=True):
            if (
                not isinstance(row, dict)
                or {
                    key: value
                    for key, value in row.items()
                    if key not in ("image_sha256", "initial_latents_sha256")
                }
                != cell
            ):
                raise ValueError("generated image grid differs from the evaluation protocol")
            path = self.directory / cell["image_path"]
            if not path.is_file() or sha256_file(path) != row.get("image_sha256"):
                raise ValueError(f"generated image failed integrity check: {path}")
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (
                    self.evaluation.sampling.width,
                    self.evaluation.sampling.height,
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
        write_json(
            self.directory / "generation_manifest.json",
            {"protocol": self.evaluation.record(), "images": rows},
        )

    def load_generation(self) -> list[dict[str, Any]]:
        self.check_tree()
        path = self.directory / "generation_manifest.json"
        if not path.is_file():
            raise ValueError("output has no committed generation stage; use a fresh --output-dir")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != self.evaluation.record():
            raise ValueError("saved generation belongs to a different evaluation protocol")
        return self.validate_images(payload.get("images"))

    def reject_completed(self) -> None:
        if (self.directory / "report").exists():
            raise FileExistsError("refusing to overwrite a completed evaluation report")

    def publish_report(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        from PIL import Image, ImageDraw, ImageOps

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
                tie_epsilon=self.evaluation.tie_epsilon,
                bootstrap_resamples=self.evaluation.bootstrap_resamples,
                seed=self.evaluation.seed,
            )
            # Blinded contact sheets: one PNG per prompt, arms in a seeded
            # shuffled order, with the key written beside them.
            by_key = {
                (row["checkpoint_label"], row["prompt_index"], row["sample_index"]): row
                for row in rows
            }
            sheet_dir = staging / "contact_sheets"
            sheet_dir.mkdir()
            orders = self.evaluation.blind_orders()
            manifest = []
            for prompt_index, labels in orders.items():
                tile, header = 256, 24
                canvas = Image.new(
                    "RGB",
                    (len(labels) * tile, self.evaluation.samples_per_prompt * (tile + header)),
                    "white",
                )
                draw = ImageDraw.Draw(canvas)
                for sample_index in range(self.evaluation.samples_per_prompt):
                    for cell, label in enumerate(labels):
                        row = by_key[label, prompt_index, sample_index]
                        x, y = cell * tile, sample_index * (tile + header)
                        draw.text((x + 4, y + 4), f"{cell + 1} / seed {row['seed']}", fill="black")
                        with Image.open(row["image_path"]) as image:
                            canvas.paste(
                                ImageOps.contain(image.convert("RGB"), (tile, tile)),
                                (x, y + header),
                            )
                filename = f"prompt{prompt_index:04d}.png"
                canvas.save(sheet_dir / filename)
                manifest.append(
                    {
                        "sheet": filename,
                        "prompt_index": prompt_index,
                        "manifest_index": self.evaluation.prompts[prompt_index].manifest_index,
                        "prompt": self.evaluation.prompts[prompt_index].example.prompt,
                    }
                )
            write_json(sheet_dir / "manifest.json", manifest)
            write_json(
                staging / "blind_key.json",
                {"note": "Review contact sheets before opening this key.", "orders": orders},
            )
            write_json(
                staging / "provenance.json",
                {
                    "protocol": self.evaluation.record(),
                    "training_reward_overlap": sorted(
                        set(self.evaluation.reward["components"])
                        & set(self.evaluation.training_reward_components)
                    ),
                    "interpretation": "Held-out reward gains are not independent human-quality proof; review the blinded images.",
                },
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
        arms = CheckpointTarget.from_cli_values(
            checkpoint_specs, run_dir=run_dir, reserved_label="base"
        )
        targets = [CheckpointTarget.load(arm.path, label=arm.label, digest=True) for arm in arms]
    else:
        by_epoch: dict[int, CheckpointTarget] = {}
        for path in sorted(run_dir.glob("checkpoint-*")):
            if not path.is_dir() or not is_complete_checkpoint(path):
                continue
            target = CheckpointTarget.load(path, label=path.name, digest=True)
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
    for target in targets:
        if target.meta.get("uses_lora") is not uses_lora:
            raise ValueError(f"checkpoint uses_lora differs from the run model: {target.path}")
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
    # Reference images run through the family executor; video conditioning has
    # no image-evaluation path.
    if any(row.example.reference_video for row in selected):
        raise ValueError("this evaluator does not support reference_video conditioning")
    if any(row.example.request_overrides for row in selected):
        raise ValueError(
            "per-prompt request_overrides are not supported by the fixed sampling grid"
        )
    if any(row.example.task_type not in {"", "text_to_image"} for row in selected):
        raise ValueError("evaluation prompt task_type must be text_to_image or unset")
    return tuple(selected)


async def score_images(
    evaluation: CheckpointEvaluation, rows: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from PIL import Image

    from vrl.config.builders import RewardRuntimeConfig
    from vrl.config.reward_inference import RewardInferenceConfig
    from vrl.config.schema import RewardConfig
    from vrl.rewards.functions.registry import MultiReward
    from vrl.rewards.types import RewardSample

    kwargs = {
        name: dict(evaluation.reward["kwargs"].get(name) or {})
        for name in evaluation.reward["components"]
    }
    deployments = RewardRuntimeConfig.from_section(
        RewardConfig.model_validate(evaluation.reward),
    ).inference_configs
    # Generation is complete before evaluation scores saved images. Local
    # judges run here, without pretending to own the training Ray placement;
    # operator-owned HTTP judges keep their configured endpoint and identity.
    deployments = {
        name: deployment if deployment.kind == "http" else RewardInferenceConfig(kind="in_process")
        for name, deployment in deployments.items()
    }
    logger.info(
        "Offline reward deployments: %s", {name: cfg.kind for name, cfg in deployments.items()}
    )
    scorer = MultiReward.from_mapping(
        evaluation.reward["components"],
        device=evaluation.reward_device,
        reward_kwargs=kwargs,
        inference_configs=deployments,
    )
    by_key = {
        (row["checkpoint_label"], row["prompt_index"], row["sample_index"]): row for row in rows
    }
    orders = evaluation.blind_orders()
    scored = []
    try:
        for prompt_index, prompt in enumerate(evaluation.prompts):
            for sample_index in range(evaluation.samples_per_prompt):
                batch_rows, samples = [], []
                for cell, label in enumerate(orders[prompt_index]):
                    row = by_key[label, prompt_index, sample_index]
                    with Image.open(output_dir / row["image_path"]) as image:
                        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                    samples.append(
                        RewardSample(
                            prompt=prompt.example.prompt,
                            output=torch.from_numpy(array.copy()).permute(2, 0, 1),
                            sample_id=f"p{prompt_index}-s{sample_index}-cell{cell}",
                            metadata=dict(row["reward_metadata"]),
                        )
                    )
                    batch_rows.append(
                        {
                            **row,
                            "blind_cell": cell + 1,
                            "image_path": str((output_dir / row["image_path"]).resolve()),
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
                    scored.append(row)
    finally:
        await scorer.shutdown()
    return scored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="[LABEL=]PATH")
    parser.add_argument(
        "--epochs", type=lambda value: tuple(int(part) for part in value.split(",")), default=()
    )
    parser.add_argument(
        "--base-only", action="store_true", help="evaluate the untrained model only"
    )
    parser.add_argument("--eval-policy-config", default=None)
    parser.add_argument("--eval-policy-override", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strata", nargs="+", default=[])
    parser.add_argument("--per-stratum", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    # Evaluate at another resolution / schedule than training (e.g. train
    # Qwen-Image-2.1 at 512px x 10 steps, evaluate at its native 2048px x 40).
    # Unset flags keep the run's eval/sampling values.
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
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
    evaluation = CheckpointEvaluation.from_args(args)
    if args.dry_run:
        print(json.dumps(evaluation.record(), indent=2, sort_keys=True))
        return
    output_dir = (
        (args.output_dir or args.run_dir / "checkpoint_evaluation").expanduser().absolute()
    )
    archive = EvaluationArchive(output_dir, evaluation)
    if output_dir.exists():
        rows = archive.load_generation()
        archive.reject_completed()
    else:
        output_dir.mkdir(parents=True)
        rows = evaluation.generate(output_dir)
        archive.publish_generation(rows)
    scored = asyncio.run(score_images(evaluation, rows, output_dir))
    summary = archive.publish_report(scored)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
