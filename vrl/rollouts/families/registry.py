"""Canonical rollout family registry.

YAML owns experiment values and defaults. This registry owns rollout wiring:
runtime construction, gatherer construction, collector kind, and capability
metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from vrl.generation.capabilities import FamilyCapability
from vrl.generation.ray.launcher import (
    RayGenerationLauncher,
    RayGenerationLaunchInputs,
)
from vrl.models.ar.capabilities import (
    ar_continuous_family_capability,
    ar_discrete_family_capability,
)
from vrl.models.ar.janus_pro import JANUS_R1_SEGMENTS
from vrl.models.diffusion.capabilities import diffusion_family_capability

CollectorKind = Literal["diffusion", "ar_discrete", "ar_continuous", "ar_r1"]


_default_return_artifacts = (
    "output",
    "trajectory",
)


@dataclass(frozen=True, slots=True)
class CollectorMetadata:
    """Collector-facing family metadata shared by collector builders."""

    kind: CollectorKind
    request_prefix: str | None = None
    default_task_type: str | None = None
    return_artifacts: tuple[str, ...] = ()
    metadata_key: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutorKwargsMetadata:
    """Runtime executor kwargs that can be derived from a full rollout cfg."""

    include_samples_per_chunk: bool = False
    include_reference_image: bool = False


@dataclass(frozen=True, slots=True)
class GathererMetadata:
    """Driver-side chunk gatherer construction metadata."""

    import_path: str
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiffusionFamilyBuild:
    """Declarative build recipe for a descriptor-driven diffusion family.

    A family whose runtime construction is pure data — no per-call code such
    as variant resolution (wan), artifact resolution (anima), or adapter hooks
    (flux NFT) — records its build inputs here and points ``runtime_builder``
    / ``runtime_spec_extractor`` at the generic functions in
    ``vrl.models.diffusion.build``. Such a family ships NO builder functions:
    its ``runtime.py`` holds only the capability constant and the chunk
    executor. Families with per-call code keep their own thin stubs instead.
    """

    model_cls: str
    replay_cls: str
    transformer_classname: str
    task_variant: str
    memory_owner: str
    scheduler_classname: str | None = None
    # Verbatim runtime_caps override for both bundles; None keeps the generic
    # default ({family_capability, supports_reference_conditioning}).
    runtime_caps: Mapping[str, Any] | None = None
    # Non-None marks the family LoRA-only: the generic builders fail loud with
    # this reason BEFORE paying the transformer load (Cosmos Predict2.5's
    # DiffusionNFT needs the default+previous adapters, which only exist on
    # the LoRA path).
    requires_lora_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RolloutFamilyEntry:
    """Declarative binding for one canonical rollout family."""

    family: str
    task: str
    collector: CollectorMetadata
    executor_cls: str
    runtime_builder: str
    runtime_spec_extractor: str
    gatherer: GathererMetadata
    capability: FamilyCapability
    executor_kwargs: ExecutorKwargsMetadata = field(
        default_factory=ExecutorKwargsMetadata,
    )
    aliases: tuple[str, ...] = ()
    build: DiffusionFamilyBuild | None = None


FAMILY_REGISTRY: dict[str, RolloutFamilyEntry] = {}


def register_rollout_family(entry: RolloutFamilyEntry) -> RolloutFamilyEntry:
    """Register one canonical rollout family."""

    if entry.family in FAMILY_REGISTRY:
        raise ValueError(f"duplicate rollout family registration: {entry.family!r}")
    FAMILY_REGISTRY[entry.family] = entry
    return entry


def _diffusion_entry(
    *,
    family: str,
    task: str,
    aliases: tuple[str, ...],
    executor_cls: str,
    runtime_builder: str,
    runtime_spec_extractor: str,
    request_prefix: str,
    default_task_type: str,
    supports_reference_conditioning: bool = False,
    build: DiffusionFamilyBuild | None = None,
) -> RolloutFamilyEntry:
    return RolloutFamilyEntry(
        family=family,
        task=task,
        aliases=aliases,
        collector=CollectorMetadata(
            kind="diffusion",
            request_prefix=request_prefix,
            default_task_type=default_task_type,
            return_artifacts=_default_return_artifacts,
        ),
        executor_cls=executor_cls,
        runtime_builder=runtime_builder,
        runtime_spec_extractor=runtime_spec_extractor,
        build=build,
        gatherer=GathererMetadata(
            import_path="vrl.generation.diffusion.gather:DiffusionChunkGatherer",
        ),
        capability=diffusion_family_capability(
            family,
            task,
            supports_reference_conditioning=supports_reference_conditioning,
        ),
        executor_kwargs=ExecutorKwargsMetadata(
            include_samples_per_chunk=True,
            include_reference_image=supports_reference_conditioning,
        ),
    )


register_rollout_family(
    _diffusion_entry(
        family="sd3_5",
        task="t2i",
        aliases=(),
        executor_cls="vrl.models.diffusion.sd3_5.runtime:SD3_5ChunkExecutor",
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="sd3_5",
        default_task_type="text_to_image",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.sd3_5.model:SD3_5Model",
            replay_cls="vrl.models.diffusion.sd3_5.model:SD3_5ReplayModel",
            transformer_classname="SD3Transformer2DModel",
            task_variant="t2i",
            memory_owner="SD3.5 VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="flux",
        task="t2i",
        aliases=("flux_1_dev",),
        executor_cls="vrl.models.diffusion.flux.runtime:FluxChunkExecutor",
        runtime_builder="vrl.models.diffusion.flux.runtime:build_flux_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.flux.runtime:extract_flux_runtime_spec",
        request_prefix="flux",
        default_task_type="text_to_image",
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="qwen_image",
        task="t2i",
        aliases=("qwen-image",),
        executor_cls="vrl.models.diffusion.qwen_image.runtime:QwenImageChunkExecutor",
        # Descriptor-driven family: the generic functions in
        # vrl.models.diffusion.build read the recipe below, so qwen_image ships
        # no per-family builder/extractor functions.
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="qwen_image",
        default_task_type="text_to_image",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.qwen_image.model:QwenImageModel",
            replay_cls="vrl.models.diffusion.qwen_image.model:QwenImageReplayModel",
            transformer_classname="QwenImageTransformer2DModel",
            task_variant="t2i",
            memory_owner="Qwen-Image VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="wan_2_1",
        task="t2v",
        aliases=("wan",),
        executor_cls="vrl.models.diffusion.wan_2_1.runtime:Wan_2_1ChunkExecutor",
        runtime_builder="vrl.models.diffusion.wan_2_1.runtime:build_wan_2_1_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.wan_2_1.runtime:extract_wan_2_1_runtime_spec",
        request_prefix="wan_2_1",
        default_task_type="text_to_video",
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="wan_2_1_i2v",
        task="i2v",
        aliases=("wan_i2v",),
        executor_cls="vrl.models.diffusion.wan_2_1.runtime:Wan_2_1I2VChunkExecutor",
        runtime_builder="vrl.models.diffusion.wan_2_1.runtime:build_wan_2_1_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.wan_2_1.runtime:extract_wan_2_1_runtime_spec",
        request_prefix="wan_2_1_i2v",
        default_task_type="image_to_video",
        supports_reference_conditioning=True,
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="cosmos-predict2",
        task="v2w",
        aliases=("cosmos", "cosmos_predict2"),
        executor_cls="vrl.models.diffusion.cosmos.predict2.runtime:CosmosChunkExecutor",
        runtime_builder=(
            "vrl.models.diffusion.cosmos.predict2.runtime:"
            "build_cosmos_predict2_runtime_bundle"
        ),
        runtime_spec_extractor=(
            "vrl.models.diffusion.cosmos.predict2.runtime:"
            "extract_cosmos_predict2_runtime_spec"
        ),
        request_prefix="cosmos-predict2",
        default_task_type="video2world",
        supports_reference_conditioning=True,
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="cosmos-predict2.5",
        task="t2w",
        aliases=("cosmos_predict2_5",),
        executor_cls=(
            "vrl.models.diffusion.cosmos.predict2_5.runtime:"
            "CosmosPredict25ChunkExecutor"
        ),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="cosmos-predict2.5",
        default_task_type="text_to_video",
        build=DiffusionFamilyBuild(
            model_cls=(
                "vrl.models.diffusion.cosmos.predict2_5.model:CosmosPredict25Model"
            ),
            replay_cls=(
                "vrl.models.diffusion.cosmos.predict2_5.model:CosmosPredict25ReplayModel"
            ),
            transformer_classname="CosmosTransformer3DModel",
            # Upstream ships UniPC; replay must recompute log-probs under the
            # same schedule the rollout sampled with.
            scheduler_classname="UniPCMultistepScheduler",
            task_variant="text2world",
            memory_owner="Cosmos Predict2.5 VAE",
            requires_lora_reason=(
                "DiffusionNFT needs the trainable default + frozen previous "
                "adapters, which only exist on the LoRA path"
            ),
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="cosmos3",
        task="t2v",
        aliases=("cosmos3_omni", "cosmos_omni"),
        executor_cls="vrl.models.diffusion.cosmos.cosmos3.runtime:Cosmos3ChunkExecutor",
        runtime_builder=(
            "vrl.models.diffusion.cosmos.cosmos3.runtime:"
            "build_cosmos3_runtime_bundle"
        ),
        runtime_spec_extractor=(
            "vrl.models.diffusion.cosmos.cosmos3.runtime:"
            "extract_cosmos3_runtime_spec"
        ),
        request_prefix="cosmos3",
        default_task_type="text_to_video",
        supports_reference_conditioning=False,
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="cosmos-predict2-anima",
        task="t2i",
        aliases=("anima", "cosmos_anima"),
        executor_cls="vrl.models.diffusion.cosmos.anima.runtime:AnimaChunkExecutor",
        runtime_builder=(
            "vrl.models.diffusion.cosmos.anima.runtime:"
            "build_anima_runtime_bundle"
        ),
        runtime_spec_extractor=(
            "vrl.models.diffusion.cosmos.anima.runtime:"
            "extract_anima_runtime_spec"
        ),
        request_prefix="anima",
        default_task_type="text_to_image",
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="echo",
        task="t2v",
        aliases=("joyai_echo",),
        executor_cls="vrl.models.diffusion.echo.runtime:EchoChunkExecutor",
        runtime_builder="vrl.models.diffusion.echo.runtime:build_echo_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.echo.runtime:extract_echo_runtime_spec",
        request_prefix="echo",
        default_task_type="text_to_video",
    ),
)

register_rollout_family(
    RolloutFamilyEntry(
        family="janus_pro",
        task="ar_t2i",
        aliases=("janus",),
        collector=CollectorMetadata(
            kind="ar_discrete",
            request_prefix="janus_pro",
            return_artifacts=_default_return_artifacts,
        ),
        executor_cls="vrl.models.ar.janus_pro.runtime:JanusProChunkExecutor",
        runtime_builder="vrl.models.ar.janus_pro.runtime:build_janus_pro_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.ar.janus_pro.runtime:extract_janus_pro_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.models.ar.janus_pro.runtime:JanusProChunkGatherer",
        ),
        capability=ar_discrete_family_capability("janus_pro", "ar_t2i"),
    ),
)

register_rollout_family(
    RolloutFamilyEntry(
        family="janus_pro_r1",
        task="ar_t2i_r1",
        aliases=("janus_r1",),
        collector=CollectorMetadata(
            kind="ar_r1",
            request_prefix="janus_pro_r1",
            return_artifacts=_default_return_artifacts,
        ),
        executor_cls="vrl.models.ar.janus_pro.runtime:JanusProR1ChunkExecutor",
        runtime_builder="vrl.models.ar.janus_pro.runtime:build_janus_pro_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.ar.janus_pro.runtime:extract_janus_pro_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.models.ar.janus_pro.runtime:JanusProR1ChunkGatherer",
        ),
        capability=ar_discrete_family_capability(
            "janus_pro_r1",
            "ar_t2i_r1",
            trajectory_kind="multisegment",
            trainable_segments=JANUS_R1_SEGMENTS,
        ),
    ),
)

register_rollout_family(
    RolloutFamilyEntry(
        family="nextstep_1",
        task="ar_t2i",
        aliases=("nextstep",),
        collector=CollectorMetadata(
            kind="ar_continuous",
            request_prefix="nextstep_1",
            return_artifacts=_default_return_artifacts,
            metadata_key="rollout_metadata",
        ),
        executor_cls="vrl.models.ar.nextstep_1.runtime:NextStep1ChunkExecutor",
        runtime_builder="vrl.models.ar.nextstep_1.runtime:build_nextstep_1_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.ar.nextstep_1.runtime:extract_nextstep_1_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.models.ar.nextstep_1.runtime:NextStep1ChunkGatherer",
        ),
        capability=ar_continuous_family_capability("nextstep_1", "ar_t2i"),
    ),
)

_FAMILY_ALIASES: dict[str, str] = {
    alias: family
    for family, entry in FAMILY_REGISTRY.items()
    for alias in (family, *entry.aliases)
}


def normalize_rollout_family(family: str) -> str:
    """Return the canonical registry key for a rollout family or alias."""

    text = str(family)
    return _FAMILY_ALIASES.get(text, text)


def get_rollout_family_entry(family: str) -> RolloutFamilyEntry:
    """Return the canonical rollout family entry for ``family``."""

    normalized = normalize_rollout_family(family)
    try:
        return FAMILY_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported rollout family: {family!r}; registered={sorted(FAMILY_REGISTRY)}",
        ) from exc


def registered_rollout_families() -> tuple[str, ...]:
    """Return canonical rollout family keys."""

    return tuple(FAMILY_REGISTRY)


def build_ray_generation_inputs_for_family(
    cfg: Any,
    family: str,
    *,
    weight_dtype: Any,
    executor_kwargs: Mapping[str, Any] | None = None,
    policy_version: int = 0,
) -> RayGenerationLaunchInputs:
    """Build Ray generation launch inputs for a registered rollout family."""

    return RayGenerationLauncher.build_inputs(
        cfg,
        get_rollout_family_entry(family),
        weight_dtype=weight_dtype,
        executor_kwargs=executor_kwargs,
        policy_version=policy_version,
    )


__all__ = [
    "FAMILY_REGISTRY",
    "CollectorKind",
    "CollectorMetadata",
    "ExecutorKwargsMetadata",
    "FamilyCapability",
    "GathererMetadata",
    "RayGenerationLaunchInputs",
    "RolloutFamilyEntry",
    "build_ray_generation_inputs_for_family",
    "get_rollout_family_entry",
    "normalize_rollout_family",
    "register_rollout_family",
    "registered_rollout_families",
]
