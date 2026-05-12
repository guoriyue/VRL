"""Canonical rollout family registry.

The registry lives in ``vrl.rollouts`` because it wires training-time rollout
components: collector metadata, executor import paths, runtime builders, and
driver-side chunk gatherers. Distributed backends should consume the resolved
entry instead of branching on concrete model families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from vrl.engine.core.capabilities import (
    FamilyCapability,
    ar_continuous_family_capability,
    ar_discrete_family_capability,
    diffusion_family_capability,
)

CollectorKind = Literal["diffusion", "ar_discrete", "ar_continuous", "ar_r1"]


@dataclass(frozen=True, slots=True)
class CollectorMetadata:
    """Collector-facing family metadata shared by collector builders."""

    kind: CollectorKind
    config_cls: str
    request_prefix: str | None = None
    default_task_type: str | None = None
    error_prefix: str | None = None
    sampling_fields: tuple[str, ...] = ()
    return_artifacts: tuple[str, ...] = ()
    metadata_key: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutorKwargsMetadata:
    """Runtime executor kwargs that can be derived from a full rollout cfg."""

    include_sample_batch_size: bool = False
    include_reference_image: bool = False


@dataclass(frozen=True, slots=True)
class GathererMetadata:
    """Driver-side chunk gatherer construction metadata."""

    import_path: str
    kwargs: dict[str, object] = field(default_factory=dict)


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


DIFFUSION_RETURN_ARTIFACTS = (
    "output",
    "rollout_trajectory_data",
    "trajectory_timesteps",
    "trajectory_latents",
    "denoising_env",
)

DIFFUSION_COMMON_SAMPLING_FIELDS = (
    "num_steps",
    "guidance_scale",
    "height",
    "width",
    "cfg",
    "sample_batch_size",
    "sde_type",
    "sde_window_size",
    "sde_window_range",
    "same_latent",
    "max_sequence_length",
    "noise_level",
    "return_kl",
)

DIFFUSION_VIDEO_SAMPLING_FIELDS = (
    *DIFFUSION_COMMON_SAMPLING_FIELDS,
    "num_frames",
)


FAMILY_REGISTRY: dict[str, RolloutFamilyEntry] = {
    "sd3_5": RolloutFamilyEntry(
        family="sd3_5",
        task="t2i",
        aliases=("sd3.5", "sd35"),
        collector=CollectorMetadata(
            kind="diffusion",
            config_cls="vrl.rollouts.collector.configs:SD3_5CollectorConfig",
            request_prefix="sd3_5",
            default_task_type="text_to_image",
            error_prefix="SD3.5",
            sampling_fields=DIFFUSION_COMMON_SAMPLING_FIELDS,
            return_artifacts=DIFFUSION_RETURN_ARTIFACTS,
        ),
        executor_cls="vrl.models.families.sd3_5.runtime:SD3_5PipelineExecutor",
        runtime_builder="vrl.models.families.sd3_5.runtime:build_sd3_5_runtime_bundle",
        runtime_spec_extractor=("vrl.models.families.sd3_5.runtime:extract_sd3_5_runtime_spec"),
        gatherer=GathererMetadata(
            import_path="vrl.engine.gather:DiffusionChunkGatherer",
            kwargs={"model_family": "sd3_5"},
        ),
        capability=diffusion_family_capability("sd3_5", "t2i"),
        executor_kwargs=ExecutorKwargsMetadata(include_sample_batch_size=True),
    ),
    "wan_2_1": RolloutFamilyEntry(
        family="wan_2_1",
        task="t2v",
        aliases=("wan", "wan_2_1_1_3b", "wan_2_1_14b"),
        collector=CollectorMetadata(
            kind="diffusion",
            config_cls="vrl.rollouts.collector.configs:Wan_2_1CollectorConfig",
            request_prefix="wan_2_1",
            default_task_type="text_to_video",
            error_prefix="Wan 2.1",
            sampling_fields=DIFFUSION_VIDEO_SAMPLING_FIELDS,
            return_artifacts=DIFFUSION_RETURN_ARTIFACTS,
        ),
        executor_cls="vrl.models.families.wan_2_1.runtime:Wan_2_1PipelineExecutor",
        runtime_builder="vrl.models.families.wan_2_1.runtime:build_wan_2_1_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.families.wan_2_1.runtime:extract_wan_2_1_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.engine.gather:DiffusionChunkGatherer",
            kwargs={"model_family": "wan_2_1"},
        ),
        capability=diffusion_family_capability("wan_2_1", "t2v"),
        executor_kwargs=ExecutorKwargsMetadata(include_sample_batch_size=True),
    ),
    "cosmos-predict2": RolloutFamilyEntry(
        family="cosmos-predict2",
        task="v2w",
        aliases=("cosmos", "cosmos_predict2", "cosmos_predict2_2b"),
        collector=CollectorMetadata(
            kind="diffusion",
            config_cls="vrl.rollouts.collector.configs:CosmosPredict2CollectorConfig",
            request_prefix="cosmos-predict2",
            default_task_type="video2world",
            error_prefix="Cosmos",
            sampling_fields=(
                *DIFFUSION_VIDEO_SAMPLING_FIELDS,
                "fps",
            ),
            return_artifacts=DIFFUSION_RETURN_ARTIFACTS,
        ),
        executor_cls="vrl.models.families.cosmos.predict2.runtime:CosmosPipelineExecutor",
        runtime_builder=(
            "vrl.models.families.cosmos.predict2.runtime:"
            "build_cosmos_predict2_runtime_bundle"
        ),
        runtime_spec_extractor=(
            "vrl.models.families.cosmos.predict2.runtime:"
            "extract_cosmos_predict2_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.engine.gather:DiffusionChunkGatherer",
            kwargs={"model_family": "cosmos-predict2", "respect_cfg_flag": False},
        ),
        capability=diffusion_family_capability(
            "cosmos-predict2",
            "v2w",
            supports_reference_conditioning=True,
        ),
        executor_kwargs=ExecutorKwargsMetadata(
            include_sample_batch_size=True,
            include_reference_image=True,
        ),
    ),
    "cosmos-predict2.5": RolloutFamilyEntry(
        family="cosmos-predict2.5",
        task="t2w",
        aliases=("cosmos_predict25", "cosmos_predict2_5", "cosmos_predict2_5_2b"),
        collector=CollectorMetadata(
            kind="diffusion",
            config_cls="vrl.rollouts.collector.configs:CosmosPredict2CollectorConfig",
            request_prefix="cosmos-predict2.5",
            default_task_type="text_to_video",
            error_prefix="Cosmos Predict2.5",
            sampling_fields=(
                *DIFFUSION_VIDEO_SAMPLING_FIELDS,
                "fps",
            ),
            return_artifacts=DIFFUSION_RETURN_ARTIFACTS,
        ),
        executor_cls=(
            "vrl.models.families.cosmos.predict2_5.runtime:"
            "CosmosPredict25PipelineExecutor"
        ),
        runtime_builder=(
            "vrl.models.families.cosmos.predict2_5.runtime:"
            "build_cosmos_predict25_runtime_bundle"
        ),
        runtime_spec_extractor=(
            "vrl.models.families.cosmos.predict2_5.runtime:"
            "extract_cosmos_predict25_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.engine.gather:DiffusionChunkGatherer",
            kwargs={"model_family": "cosmos-predict2.5"},
        ),
        capability=diffusion_family_capability("cosmos-predict2.5", "t2w"),
        executor_kwargs=ExecutorKwargsMetadata(include_sample_batch_size=True),
    ),
    "janus_pro": RolloutFamilyEntry(
        family="janus_pro",
        task="ar_t2i",
        aliases=("janus", "janus_pro_1b"),
        collector=CollectorMetadata(
            kind="ar_discrete",
            config_cls="vrl.rollouts.collector.configs:JanusProCollectorConfig",
            request_prefix="janus_pro",
            sampling_fields=(
                "cfg_weight",
                "temperature",
                "image_token_num",
                "image_size",
                "max_text_length",
            ),
            return_artifacts=("output", "token_ids", "token_log_probs"),
        ),
        executor_cls="vrl.models.families.janus_pro.runtime:JanusProPipelineExecutor",
        runtime_builder=("vrl.models.families.janus_pro.runtime:build_janus_pro_runtime_bundle"),
        runtime_spec_extractor=(
            "vrl.models.families.janus_pro.runtime:extract_janus_pro_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.models.families.janus_pro.runtime:JanusProChunkGatherer",
        ),
        capability=ar_discrete_family_capability("janus_pro", "ar_t2i"),
    ),
    "janus_pro_r1": RolloutFamilyEntry(
        family="janus_pro_r1",
        task="ar_t2i_r1",
        aliases=("janus_r1", "janus_pro_1b_r1"),
        collector=CollectorMetadata(
            kind="ar_r1",
            config_cls="vrl.rollouts.collector.configs:JanusProR1CollectorConfig",
            request_prefix="janus_pro_r1",
            sampling_fields=(
                "cfg_weight",
                "temperature",
                "image_token_num",
                "image_size",
                "max_text_length",
                "max_reflect_len",
                "final_image_policy",
                "train_segments",
            ),
            return_artifacts=(
                "output",
                "r1_segments",
                "initial_image",
                "final_image",
                "selfcheck_text",
            ),
        ),
        executor_cls=(
            "vrl.models.families.janus_pro.runtime:JanusProR1PipelineExecutor"
        ),
        runtime_builder=("vrl.models.families.janus_pro.runtime:build_janus_pro_runtime_bundle"),
        runtime_spec_extractor=(
            "vrl.models.families.janus_pro.runtime:extract_janus_pro_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.models.families.janus_pro.runtime:JanusProR1ChunkGatherer",
        ),
        capability=ar_discrete_family_capability(
            "janus_pro_r1",
            "ar_t2i_r1",
            multisegment=True,
        ),
    ),
    "nextstep_1": RolloutFamilyEntry(
        family="nextstep_1",
        task="ar_t2i",
        aliases=("nextstep", "nextstep_1_1"),
        collector=CollectorMetadata(
            kind="ar_continuous",
            config_cls="vrl.rollouts.collector.configs:NextStep1CollectorConfig",
            request_prefix="nextstep_1",
            sampling_fields=(
                "cfg_scale",
                "num_flow_steps",
                "noise_level",
                "image_token_num",
                "image_size",
                "max_text_length",
                "rescale_to_unit",
            ),
            return_artifacts=("output", "rollout_trajectory_data"),
            metadata_key="rollout_metadata",
        ),
        executor_cls="vrl.models.families.nextstep_1.runtime:NextStep1PipelineExecutor",
        runtime_builder=("vrl.models.families.nextstep_1.runtime:build_nextstep_1_runtime_bundle"),
        runtime_spec_extractor=(
            "vrl.models.families.nextstep_1.runtime:extract_nextstep_1_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.models.families.nextstep_1.runtime:NextStep1ChunkGatherer",
        ),
        capability=ar_continuous_family_capability("nextstep_1", "ar_t2i"),
    ),
}

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


__all__ = [
    "DIFFUSION_COMMON_SAMPLING_FIELDS",
    "DIFFUSION_RETURN_ARTIFACTS",
    "DIFFUSION_VIDEO_SAMPLING_FIELDS",
    "FAMILY_REGISTRY",
    "CollectorKind",
    "CollectorMetadata",
    "ExecutorKwargsMetadata",
    "FamilyCapability",
    "GathererMetadata",
    "RolloutFamilyEntry",
    "get_rollout_family_entry",
    "normalize_rollout_family",
    "registered_rollout_families",
]
