"""Generate a few individual images through the configured production rollout."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from vrl.generation.execution.planner import EnginePlan
from vrl.rollouts.collector.requests import GenerationRequestBuilder

# Stable test-fixture values make repeated config previews directly comparable.
_PREVIEW_IMAGE_COUNT = 4
_PREVIEW_BASE_SEED = 20260718


def build_preview_request(
    builder: GenerationRequestBuilder,
    example: Any,
    *,
    seed: int,
) -> Any:
    """Adapt one real training example into a one-image production request."""

    overrides = dict(example.request_overrides)
    # Preserve the current precedence: an explicit seed in the example's
    # request_overrides wins; otherwise the deterministic preview seed is used.
    overrides.setdefault("seed", seed)
    request = builder.build(
        [example.generation_input()],
        group_size=1,
        metadata=example.reward_metadata(),
        request_overrides=overrides,
    ).request
    # Ray resolves ``auto`` from runtime memory. The direct preview has exactly
    # one sample, so only that unresolved sentinel needs a local value. Preserve
    # every explicit numeric batch size from the experiment YAML.
    if request.samples_per_generation_batch == "auto":
        request = replace(request, samples_per_generation_batch=1)
    return request


def write_preview_image(
    output: Any,
    path: Path,
    *,
    expected_request_id: str,
    expected_prompt: str,
) -> None:
    """Persist the sole image while checking request/output identity."""

    from vrl.utils.media import write_png

    if len(output.sample_rows) != 1:
        raise RuntimeError(
            f"production executor returned {len(output.sample_rows)} sample rows; expected 1",
        )
    row = output.sample_rows[0]
    if output.request_id != expected_request_id or row.prompt != expected_prompt:
        raise RuntimeError(
            "production output changed request/prompt identity: "
            f"expected=({expected_request_id!r}, {expected_prompt!r}) "
            f"actual=({output.request_id!r}, {row.prompt!r})",
        )
    if len(output.output) != 1:
        raise RuntimeError(
            f"production executor returned {len(output.output)} images; expected 1",
        )
    write_png(output.output[0], path)


def generate_rollout_preview(
    cfg: Any,
    output_dir: str | Path,
    *,
    config_name: str,
) -> dict[str, Any]:
    """Run the exact image rollout configured by an RL experiment YAML."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("rollout preview requires a CUDA GPU")

    from vrl.config.validation import require_training_config
    from vrl.models.dtypes import dtype_to_wire_name
    from vrl.models.families.registry import (
        GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR,
        get_model_family_entry,
    )
    from vrl.models.interfaces.replay import require_runtime_model
    from vrl.rollouts.collector.config import RolloutCollectorConfig
    from vrl.trainers.data import load_prompt_examples_from_config
    from vrl.utils.config import import_from_path, to_builtin_deep

    validated = require_training_config(cfg)
    root = validated.root
    if root.model is None:
        raise ValueError("rollout preview requires model configuration")
    resume_from = str(
        (root.trainer.resume_from if root.trainer is not None else None) or ""
    ).strip()
    if resume_from:
        raise ValueError(
            "rollout preview does not restore trainer.resume_from checkpoints; "
            "set model.path or model.lora.path to the exact inference weights",
        )

    family = str(root.model.family)
    entry = get_model_family_entry(family)
    if entry.task != "t2i":
        raise ValueError(
            f"image rollout preview requires a t2i family; {entry.family!r} has task {entry.task!r}",
        )

    examples = load_prompt_examples_from_config(cfg.data)
    if not examples:
        raise ValueError("rollout preview requires at least one prompt in data.manifest")
    examples = examples[:_PREVIEW_IMAGE_COUNT]

    target = Path(output_dir).expanduser().resolve()
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"rollout preview output directory must not already exist: {target}",
        ) from error

    device = torch.device("cuda")
    build = entry.resolve_model_build(
        root,
        device,
        precision=validated.precision,
        for_rollout=True,
    )
    bundle = entry.build_rollout(build)
    # No quantization backstop here, matching the rollout worker: the shared
    # builder's optimization pass already fails loud on a zero-match swap.
    model = require_runtime_model(bundle.model, owner="RuntimeBundle.model")

    executor_kwargs = entry.executor_kwargs(root)
    executor_kwargs["gatherer"] = entry.new_gatherer()
    if entry.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR:
        executor_kwargs.update(family=entry.family, task=entry.task)
    executor_cls = import_from_path(entry.executor_cls)
    executor = executor_cls(model, **executor_kwargs)

    request_builder = GenerationRequestBuilder(
        entry=entry,
        config=RolloutCollectorConfig.from_root(root),
    )
    items: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        seed = _PREVIEW_BASE_SEED + index
        request = build_preview_request(request_builder, example, seed=seed)
        request.request_id = f"{entry.family}-rollout-preview-{index}"
        rows = request.sample_rows()
        plan = EnginePlan.from_request(request)
        with torch.inference_mode():
            output = executor.forward_plan(request, rows, plan)

        file_name = f"{index:03d}.png"
        write_preview_image(
            output,
            target / file_name,
            expected_request_id=request.request_id,
            expected_prompt=example.prompt,
        )
        items.append(
            {
                "file": file_name,
                "prompt": example.prompt,
                "seed": seed,
                "sampling": to_builtin_deep(request.sampling),
            },
        )

    rollout = build.require_rollout()
    preview = {
        "config": config_name,
        "family": entry.family,
        "task": entry.task,
        "model": {
            "path": str(build.model_name_or_path),
            "revision": root.model.revision,
        },
        "precision": {
            "parameter_dtype": dtype_to_wire_name(build.parameter_dtype),
            "prompt_encoder_dtype": dtype_to_wire_name(rollout.prompt_encoder_dtype),
            "dtype": bundle.precision.dtype,
            "outer_autocast": bundle.precision.outer_autocast,
            "float32_precision": bundle.precision.float32_precision,
            "quantization": (
                None
                if bundle.precision.quantization is None
                else {
                    "format": bundle.precision.quantization.format,
                    "recipe": bundle.precision.quantization.recipe,
                }
            ),
        },
        "items": items,
    }
    (target / "preview.json").write_text(
        json.dumps(preview, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return preview


__all__ = [
    "build_preview_request",
    "generate_rollout_preview",
    "write_preview_image",
]
