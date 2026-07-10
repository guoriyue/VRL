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
from vrl.models.ar.emu3.runtime import EMU3_FAMILY_CAPABILITY
from vrl.models.ar.glm_image.runtime import GLM_IMAGE_FAMILY_CAPABILITY
from vrl.models.ar.janus_pro.runtime import (
    JANUS_PRO_FAMILY_CAPABILITY,
    JANUS_PRO_R1_FAMILY_CAPABILITY,
)
from vrl.models.ar.llamagen.runtime import LLAMAGEN_FAMILY_CAPABILITY
from vrl.models.ar.nextstep_1.runtime import NEXTSTEP_1_FAMILY_CAPABILITY
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
    task_variant: str
    memory_owner: str
    # Replay recipe; None marks a family whose replay builder stays hand-written
    # (wan's multi-transformer replay) — the generic replay builder fails loud.
    replay_cls: str | None = None
    transformer_classname: str | None = None
    scheduler_classname: str | None = None
    # LoRA-only family: the generic builders fail loud BEFORE paying the
    # transformer load. The per-family WHY belongs in a comment on the entry
    # (and in the model's own apply_full_finetune error), not in runtime data.
    requires_lora: bool = False



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
    aliases: tuple[str, ...] = ()
    build: DiffusionFamilyBuild | None = None


FAMILY_REGISTRY: dict[str, RolloutFamilyEntry] = {}


def register_rollout_family(entry: RolloutFamilyEntry) -> RolloutFamilyEntry:
    """Register one canonical rollout family."""

    if entry.family in FAMILY_REGISTRY:
        raise ValueError(f"duplicate rollout family registration: {entry.family!r}")
    FAMILY_REGISTRY[entry.family] = entry
    return entry


# One generic executor serves every family whose chunk executor is pure data
# (no build_chunk_encoded / encode_prompt_for_chunk override). Such families
# leave ``executor_cls`` unset (defaulting here) and put their executor config
# in the model yaml's ``executor`` block.
GENERIC_DIFFUSION_EXECUTOR = (
    "vrl.generation.diffusion.executor:DiffusionChunkExecutor"
)


def _diffusion_entry(
    *,
    family: str,
    task: str,
    aliases: tuple[str, ...],
    runtime_builder: str,
    runtime_spec_extractor: str,
    request_prefix: str,
    default_task_type: str,
    executor_cls: str | None = None,
    supports_reference_conditioning: bool = False,
    build: DiffusionFamilyBuild | None = None,
) -> RolloutFamilyEntry:
    # Default dispatch: the shared generic executor. Families with real
    # per-chunk logic pass their own executor_cls. Per-family executor config
    # (num_frames / max_sequence_length / ...) lives in the model yaml's
    # ``executor`` block, read wholesale at launch — not here.
    if executor_cls is None:
        executor_cls = GENERIC_DIFFUSION_EXECUTOR
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
    )


register_rollout_family(
    _diffusion_entry(
        family="sd3_5",
        task="t2i",
        aliases=(),
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
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="flux",
        default_task_type="text_to_image",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.flux.model:FluxModel",
            task_variant="t2i",
            memory_owner="FLUX VAE",
            replay_cls="vrl.models.diffusion.flux.model:FluxReplayModel",
            transformer_classname="FluxTransformer2DModel",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="qwen_image",
        task="t2i",
        aliases=("qwen-image",),
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
        family="sana",
        task="t2i",
        aliases=("sana_1600m",),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="sana",
        default_task_type="text_to_image",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.sana.model:SanaModel",
            replay_cls="vrl.models.diffusion.sana.model:SanaReplayModel",
            transformer_classname="SanaTransformer2DModel",
            task_variant="t2i",
            memory_owner="SANA DC-AE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="lumina2",
        task="t2i",
        aliases=("lumina_image_2",),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="lumina2",
        default_task_type="text_to_image",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.lumina2.model:Lumina2Model",
            replay_cls="vrl.models.diffusion.lumina2.model:Lumina2ReplayModel",
            transformer_classname="Lumina2Transformer2DModel",
            task_variant="t2i",
            memory_owner="Lumina2 VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="hunyuan_video",
        task="t2v",
        aliases=("hunyuanvideo",),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="hunyuan_video",
        default_task_type="text_to_video",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.hunyuan_video.model:HunyuanVideoModel",
            replay_cls="vrl.models.diffusion.hunyuan_video.model:HunyuanVideoReplayModel",
            transformer_classname="HunyuanVideoTransformer3DModel",
            task_variant="t2v",
            memory_owner="HunyuanVideo VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="mochi",
        task="t2v",
        aliases=("mochi_1",),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="mochi",
        default_task_type="text_to_video",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.mochi.model:MochiModel",
            replay_cls="vrl.models.diffusion.mochi.model:MochiReplayModel",
            transformer_classname="MochiTransformer3DModel",
            task_variant="t2v",
            memory_owner="Mochi VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="hunyuan_image",
        task="t2i",
        aliases=("hunyuanimage_2_1",),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="hunyuan_image",
        default_task_type="text_to_image",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.hunyuan_image.model:HunyuanImageModel",
            replay_cls="vrl.models.diffusion.hunyuan_image.model:HunyuanImageReplayModel",
            transformer_classname="HunyuanImageTransformer2DModel",
            task_variant="t2i",
            memory_owner="HunyuanImage VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="pixart_sigma",
        task="t2i",
        aliases=("pixart",),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="pixart_sigma",
        default_task_type="text_to_image",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.pixart_sigma.model:PixArtSigmaModel",
            replay_cls="vrl.models.diffusion.pixart_sigma.model:PixArtSigmaReplayModel",
            transformer_classname="PixArtTransformer2DModel",
            # Epsilon DDPM family (sde_type=ddim): load a DDIMScheduler so the
            # shipped beta config survives into prepare_replay, which rebuilds
            # the rollout's DDIM ladder via pixart_ddim_scheduler.
            scheduler_classname="DDIMScheduler",
            task_variant="t2i",
            memory_owner="PixArt VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="cogvideox",
        task="t2v",
        aliases=("cogvideox_2b", "cogvideox_5b"),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="cogvideox",
        default_task_type="text_to_video",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.cogvideox.model:CogVideoXModel",
            replay_cls="vrl.models.diffusion.cogvideox.model:CogVideoXReplayModel",
            transformer_classname="CogVideoXTransformer3DModel",
            # v-prediction DDPM family: replay recomputes log-probs on the
            # same ladder the rollout sampled (sde_type=ddim).
            scheduler_classname="CogVideoXDDIMScheduler",
            task_variant="t2v",
            memory_owner="CogVideoX VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="wan_2_1",
        task="t2v",
        aliases=("wan",),
        # The two wan entries carry their own per-variant recipes, so the
        # t2v/i2v resolution is decided here, once, by family selection. The
        # dual-stage transformer_2 late-load lives in the replay model's
        # prepare_replay, so replay is generic too.
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="wan_2_1",
        default_task_type="text_to_video",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.wan_2_1.model:WanT2VDiffusersModel",
            replay_cls="vrl.models.diffusion.wan_2_1.model:WanT2VReplayModel",
            transformer_classname="WanTransformer3DModel",
            # Replay recomputes log-probs on the schedule the rollout sampled.
            scheduler_classname="UniPCMultistepScheduler",
            task_variant="t2v",
            memory_owner="Wan VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="wan_2_1_i2v",
        task="i2v",
        aliases=("wan_i2v",),
        executor_cls="vrl.models.diffusion.wan_2_1.runtime:Wan_2_1I2VChunkExecutor",
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="wan_2_1_i2v",
        default_task_type="image_to_video",
        supports_reference_conditioning=True,
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.wan_2_1.model:WanI2VDiffusersModel",
            replay_cls="vrl.models.diffusion.wan_2_1.model:WanI2VReplayModel",
            transformer_classname="WanTransformer3DModel",
            scheduler_classname="UniPCMultistepScheduler",
            task_variant="i2v",
            memory_owner="Wan VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="cosmos-predict2",
        task="v2w",
        aliases=("cosmos", "cosmos_predict2"),
        executor_cls="vrl.models.diffusion.cosmos.predict2.runtime:CosmosChunkExecutor",
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="cosmos-predict2",
        default_task_type="video2world",
        supports_reference_conditioning=True,
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.cosmos.predict2.model:CosmosPredict2Model",
            task_variant="video2world",
            memory_owner="Cosmos Predict2 VAE",
            replay_cls="vrl.models.diffusion.cosmos.predict2.model:CosmosPredict2ReplayModel",
            transformer_classname="CosmosTransformer3DModel",
        ),
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
            # DiffusionNFT needs the trainable default + frozen previous
            # adapters, which only exist on the LoRA path.
            requires_lora=True,
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="cosmos3",
        task="t2v",
        aliases=("cosmos3_omni", "cosmos_omni"),
        executor_cls="vrl.models.diffusion.cosmos.cosmos3.runtime:Cosmos3ChunkExecutor",
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="cosmos3",
        default_task_type="text_to_video",
        supports_reference_conditioning=False,
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.cosmos.cosmos3.model:Cosmos3Model",
            task_variant="text2world",
            memory_owner="Cosmos3 VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="cosmos-predict2-anima",
        task="t2i",
        aliases=("anima", "cosmos_anima"),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="anima",
        default_task_type="text_to_image",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.cosmos.anima.model:AnimaModel",
            task_variant="t2i",
            memory_owner="Anima VAE",
        ),
    ),
)

register_rollout_family(
    _diffusion_entry(
        family="echo",
        task="t2v",
        aliases=("joyai_echo",),
        executor_cls="vrl.models.diffusion.echo.runtime:EchoChunkExecutor",
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.build:extract_family_runtime_spec",
        request_prefix="echo",
        default_task_type="text_to_video",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.echo.model:EchoModel",
            task_variant="text2video",
            memory_owner="Echo video VAE",
        ),
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
            import_path="vrl.generation.ar.executor:ARDiscreteChunkGatherer",
        ),
        capability=JANUS_PRO_FAMILY_CAPABILITY,
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
        capability=JANUS_PRO_R1_FAMILY_CAPABILITY,
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
        capability=NEXTSTEP_1_FAMILY_CAPABILITY,
    ),
)

register_rollout_family(
    RolloutFamilyEntry(
        family="emu3",
        task="ar_t2i",
        aliases=("emu3_gen",),
        collector=CollectorMetadata(
            kind="ar_discrete",
            request_prefix="emu3",
            return_artifacts=_default_return_artifacts,
        ),
        executor_cls="vrl.models.ar.emu3.runtime:Emu3ChunkExecutor",
        runtime_builder="vrl.models.ar.emu3.runtime:build_emu3_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.ar.emu3.runtime:extract_emu3_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.generation.ar.executor:ARDiscreteChunkGatherer",
        ),
        capability=EMU3_FAMILY_CAPABILITY,
    ),
)

register_rollout_family(
    RolloutFamilyEntry(
        family="glm_image",
        task="ar_t2i",
        aliases=("glm_image_t2i",),
        collector=CollectorMetadata(
            kind="ar_discrete",
            request_prefix="glm_image",
            return_artifacts=_default_return_artifacts,
        ),
        executor_cls="vrl.models.ar.glm_image.runtime:GlmImageChunkExecutor",
        runtime_builder="vrl.models.ar.glm_image.runtime:build_glm_image_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.ar.glm_image.runtime:extract_glm_image_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.generation.ar.executor:ARDiscreteChunkGatherer",
        ),
        capability=GLM_IMAGE_FAMILY_CAPABILITY,
    ),
)

register_rollout_family(
    RolloutFamilyEntry(
        family="llamagen",
        task="ar_t2i",
        aliases=("llamagen_t2i",),
        collector=CollectorMetadata(
            kind="ar_discrete",
            request_prefix="llamagen",
            return_artifacts=_default_return_artifacts,
        ),
        executor_cls="vrl.models.ar.llamagen.runtime:LlamaGenChunkExecutor",
        runtime_builder="vrl.models.ar.llamagen.runtime:build_llamagen_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.ar.llamagen.runtime:extract_llamagen_runtime_spec"
        ),
        gatherer=GathererMetadata(
            import_path="vrl.generation.ar.executor:ARDiscreteChunkGatherer",
        ),
        capability=LLAMAGEN_FAMILY_CAPABILITY,
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
