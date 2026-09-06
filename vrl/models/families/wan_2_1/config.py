"""Lightweight public config schema for the Wan model family."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import (
    checkpoint_identity_metadata,
    require_remote_checkpoint_source_pin,
)

if TYPE_CHECKING:
    from vrl.models.interfaces.runtime import ModelBuild

WanTransformerName = Literal["transformer", "transformer_2"]


class WanModelSection(ModelSection):
    """Wan-specific public model keys."""

    expert_lifecycle_profiling: bool = Field(
        default=False,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    offload_mode: Literal["none", "model", "sequential"] = Field(
        default="none",
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    trainable_transformers: (
        list[WanTransformerName] | Literal["all", "both"] | WanTransformerName | None
    ) = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "value",
            required=True,
            canonicalize="sorted_unique",
        ),
    )


def normalize_wan_boundary_ratio(value: Any, *, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float or null") from exc
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1, got {ratio!r}")
    return ratio


def normalize_wan_trainable_transformers(
    value: Any,
    *,
    dual_stage: bool,
) -> tuple[WanTransformerName, ...]:
    """Canonicalize Wan policy ownership independently of config spelling."""

    if value is None or value == "":
        requested = {"transformer_2"} if dual_stage else {"transformer"}
    elif isinstance(value, str):
        text = value.strip().lower()
        requested = {"transformer", "transformer_2"} if text in {"all", "both"} else {text}
    else:
        try:
            requested = {str(item).strip().lower() for item in value}
        except TypeError as exc:
            raise ValueError(
                "model.trainable_transformers must be a transformer name, "
                "'all'/'both', or a sequence of transformer names",
            ) from exc

    allowed = {"transformer", "transformer_2"} if dual_stage else {"transformer"}
    invalid = sorted(requested - allowed)
    if invalid:
        raise ValueError(
            f"invalid Wan trainable transformer(s) {invalid}; allowed={sorted(allowed)}",
        )
    if not requested:
        raise ValueError("Wan trainable_transformers must not be empty")
    return tuple(name for name in ("transformer", "transformer_2") if name in requested)


def normalize_wan_model_build(build: ModelBuild) -> ModelBuild:
    """Resolve immutable dual-stage topology before replay or rollout construction."""

    if build.family not in {"wan_2_1", "wan_2_1_i2v"}:
        raise ValueError(f"Wan build normalizer received family {build.family!r}")

    require_remote_checkpoint_source_pin(
        build.model_name_or_path,
        build.revision,
        field_name="model.path",
    )

    from diffusers import DiffusionPipeline

    source_config = DiffusionPipeline.load_config(
        build.model_name_or_path,
        **build.revision_kwargs,
    )
    if not isinstance(source_config, Mapping):
        raise TypeError(
            "DiffusionPipeline.load_config must return a mapping for Wan topology",
        )
    if bool(source_config.get("expand_timesteps", False)):
        raise NotImplementedError(
            "Wan RL does not yet support expand_timesteps pipelines.",
        )

    source_boundary = normalize_wan_boundary_ratio(
        source_config.get("boundary_ratio"),
        field_name="pipeline boundary_ratio",
    )
    model_config = dict(build.model_config or {})
    names = normalize_wan_trainable_transformers(
        model_config.get("trainable_transformers"),
        dual_stage=source_boundary is not None,
    )

    # Pipeline offload: move the public model.offload_mode onto the rollout build
    # only, so replay builds stay residency-neutral and the rollout worker owns
    # the CPU-offload decision. Kept in the same normalizer as topology so both
    # immutable projections happen once, before replay or rollout construction.
    from dataclasses import replace

    from vrl.models.interfaces.runtime import PipelineOffloadMode

    legacy = sorted(
        key
        for key in ("enable_model_cpu_offload", "enable_sequential_cpu_offload")
        if key in model_config
    )
    if legacy:
        raise ValueError(
            f"removed Wan model config key(s): {', '.join('model.' + key for key in legacy)}; "
            "use model.offload_mode='none', 'model', or 'sequential'",
        )
    mode = model_config.get("offload_mode")
    if "offload_mode" not in model_config and build.rollout is not None:
        mode = build.rollout.pipeline_offload_mode
    if mode is None or mode == "":
        mode = "none"
    mode = str(mode)
    allowed_modes = [m.value for m in PipelineOffloadMode]
    if mode not in allowed_modes:
        raise ValueError(
            f"model.offload_mode must be one of {allowed_modes}, got {mode!r}",
        )
    model_config.pop("offload_mode", None)

    model_config["boundary_ratio"] = source_boundary
    model_config["trainable_transformers"] = list(names)
    build.model_config = model_config
    if build.rollout is not None:
        build.rollout = replace(build.rollout, pipeline_offload_mode=mode)
    return build


def wan_topology_from_build(
    build: ModelBuild,
) -> tuple[float | None, tuple[WanTransformerName, ...]]:
    """Read the canonical topology installed by ``normalize_wan_model_build``."""

    model_config = build.model_config or {}
    if "boundary_ratio" not in model_config or "trainable_transformers" not in model_config:
        raise ValueError(
            "Wan ModelBuild is missing canonical topology; construct it through "
            "ModelFamilyEntry.resolve_model_build",
        )
    boundary_ratio = normalize_wan_boundary_ratio(
        model_config["boundary_ratio"],
        field_name="canonical Wan boundary_ratio",
    )
    raw_names = model_config["trainable_transformers"]
    if not isinstance(raw_names, list):
        raise TypeError(
            "canonical Wan model.trainable_transformers must be a list",
        )
    names = normalize_wan_trainable_transformers(
        raw_names,
        dual_stage=boundary_ratio is not None,
    )
    if raw_names != list(names):
        raise ValueError(
            "Wan ModelBuild trainable_transformers is not canonical: "
            f"expected={list(names)!r}, actual={raw_names!r}",
        )
    return boundary_ratio, names


__all__ = [
    "WanModelSection",
    "WanTransformerName",
    "normalize_wan_boundary_ratio",
    "normalize_wan_model_build",
    "normalize_wan_trainable_transformers",
    "wan_topology_from_build",
]
