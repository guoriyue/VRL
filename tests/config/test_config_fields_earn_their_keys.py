"""A YAML key exists because a bundled experiment moves it off its default.

The rule (2026-11-04 field audit): a config field is a key only when at least one
bundled recipe sets a non-default value, or two runs need different values;
otherwise it is a module constant with a comment saying why, or it is deleted.
The test parses every bundled experiment, walks the typed config objects it
builds, and records which fields ever differ from their defaults. Fields that
never do must be listed in ``_UNMOVED`` with the reason they are still keys; the
list is exact, so a recipe that starts setting one shrinks it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from vrl.config.loading import list_bundled_configs, load_config
from vrl.config.schema import parse_config

_UNMOVED: dict[str, str] = {
    # Published defaults a run may tune; the value in code is the reference.
    "actor.max_norm": "grad clip at the trainer's 1.0",
    "actor.optim.adam_beta1": "torch Adam default",
    "actor.optim.adam_beta2": "torch Adam default",
    "algorithm.hyperparameters.add_kl_coefficient": "Flow-DPPO paper value",
    "algorithm.hyperparameters.adv_clip_max": "paper value",
    "algorithm.hyperparameters.adv_soft_clip": "V-GRPO paper value",
    "algorithm.hyperparameters.advantage_scale": "DiffusionNFT paper value",
    "algorithm.hyperparameters.beta": "Diffusion-DPO paper value",
    "algorithm.hyperparameters.eps": "paper value",
    "algorithm.hyperparameters.kl_mask_threshold": "Flow-DPPO paper value",
    "algorithm.hyperparameters.nft_beta": "DiffusionNFT paper value",
    # Opt-in modes and diagnostics: off unless a run asks for them.
    "precision.rollout.quantization": "FP8/NVFP4 rollout kernels (docs/PRECISION.md)",
    "precision.training.quantization": "replay through the rollout's FP8/NVFP4 GEMM (f14c55928)",
    "rollout.torch_profiler": "profiling window",
    "trainer.torch_profiler": "profiling window",
    "trainer.deterministic": "cuDNN/algorithm determinism switch",
    "trainer.precision_correction": "TIS/RS for off-policy replay",
    "trainer.replay_parity.every_update": "overrides the derived not-continuous default",
    # Launch-time values that name a run's own inputs.
    "trainer.resume_from": "the checkpoint a launch resumes",
    "reward.inference.*.expected_model_version": "pin for an operator-owned http service",
    "distributed.training.fsdp.context_parallel": "set by base/distributed/training_fsdp_cp2",
    "distributed.training.fsdp.context_parallel.ulysses_degree": "same preset",
    # No bundled recipe sets these; each is a keep-or-delete decision.
    "data.max_train_samples": "Wan DPO subset cap (scripts/families/wan_2_1/train_dpo.py)",
    "data.preprocessing.reference_image": "i2v/v2w reference column (docs/CONFIGURATION.md)",
    "model.torch_compile.scope": "rollout-only compile; recipes went eager in b15dd5c28",
    "sampling.input_mode": "Qwen-Image-Layered source alpha (docs/qwen_image_layered.md)",
    "sampling.negative_prompt": "CFG negative prompt (docs/CONFIGURATION.md)",
    "sampling.output_mode": "Qwen-Image-2.1 RGBA decode (docs/qwen_image_21.md)",
}


def _fields(obj: Any) -> list[tuple[str, Any, Any]]:
    """``(name, default, value)`` of every field a config file can set."""

    if isinstance(obj, BaseModel):
        rows = []
        for name, info in type(obj).model_fields.items():
            if info.default is not PydanticUndefined:
                default = info.default
            elif info.default_factory is not None:
                default = info.default_factory()
            else:
                default = PydanticUndefined
            rows.append((name, default, getattr(obj, name)))
        return rows
    rows = []
    for field in dataclasses.fields(obj):
        if not field.init:  # filled by a builder, never a key
            continue
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        else:
            default = PydanticUndefined
        rows.append((field.name, default, getattr(obj, field.name)))
    return rows


def _walk(obj: Any, path: str, seen: set[str], moved: set[str]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            _walk(value, f"{path}.*", seen, moved)
        return
    if isinstance(obj, list | tuple):
        for value in obj:
            _walk(value, path, seen, moved)
        return
    if not isinstance(obj, BaseModel) and not (
        dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    ):
        return
    for name, default, value in _fields(obj):
        key = f"{path}.{name}" if path else name
        seen.add(key)
        if default is PydanticUndefined or value != default:
            moved.add(key)
        _walk(value, key, seen, moved)


def test_every_config_field_is_moved_by_a_recipe_or_listed_with_a_reason() -> None:
    names = list_bundled_configs("experiment")
    assert names
    seen: set[str] = set()
    moved: set[str] = set()
    for name in names:
        _walk(parse_config(load_config(name.removesuffix(".yaml"))), "", seen, moved)

    unmoved = seen - moved
    assert {
        "never moved, not listed": sorted(unmoved - _UNMOVED.keys()),
        "listed, now moved or gone": sorted(_UNMOVED.keys() - unmoved),
    } == {"never moved, not listed": [], "listed, now moved or gone": []}
