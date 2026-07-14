"""Canonical model-family registry.

YAML owns experiment values and defaults. This composition boundary owns the
single family table shared by model construction, generation, and collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args

from vrl.families.names import (
    normalize_model_family,
    validate_model_family_aliases,
)

CollectorKind = Literal["diffusion", "ar_discrete", "ar_continuous", "ar_r1"]

# Import-path protocol value shared by registry dispatch and generation workers.
# Keeping it here avoids making the neutral family table import a runtime module.
GENERIC_DIFFUSION_EXECUTOR = "vrl.generation.diffusion.executor:DiffusionChunkExecutor"


@dataclass(frozen=True, slots=True)
class DiffusionFamilyBuild:
    """Declarative build recipe for a descriptor-driven diffusion family.

    A family whose runtime construction is pure data records its build inputs
    here. ``ModelFamilyEntry`` derives the shared resolver and worker builder
    from whether this descriptor or an ``ARFamilyBuild`` is present. Only real
    family-specific replay assembly remains as an explicit override.
    """

    model_cls: str
    # Replay recipe; None marks a family whose registry entry points directly to
    # its hand-written replay builder (echo/cosmos3/anima).
    replay_cls: str | None = None
    transformer_classname: str | None = None
    scheduler_classname: str | None = None
    # Only families whose replay construction cannot use the generic descriptor
    # path declare an override (echo/cosmos3/anima).
    replay_runtime_builder: str | None = None
    # Base transformer parameter dtype required by the model family, independent
    # of rollout/replay autocast. Keep this on the build descriptor only for a
    # genuine model invariant; ordinary families inherit the training dtype.
    base_parameter_dtype: str | None = None
    # LoRA-only family: the generic builders fail loud BEFORE paying the
    # transformer load. The per-family WHY belongs in a comment on the entry
    # (and in the model's own apply_full_finetune error), not in runtime data.
    requires_lora: bool = False

    def __post_init__(self) -> None:
        generic_replay = self.replay_cls is not None and self.transformer_classname is not None
        partial_generic_replay = (self.replay_cls is None) != (self.transformer_classname is None)
        custom_replay = self.replay_runtime_builder is not None
        if partial_generic_replay or generic_replay == custom_replay:
            raise ValueError(
                "diffusion family build must declare either replay_cls plus "
                "transformer_classname, or one custom replay_runtime_builder",
            )
        if custom_replay and self.scheduler_classname is not None:
            raise ValueError(
                "scheduler_classname belongs to the generic replay builder and "
                "cannot accompany replay_runtime_builder",
            )


@dataclass(frozen=True, slots=True)
class ARFamilyBuild:
    """Declarative model construction for a descriptor-driven AR family."""

    model_cls: str
    replay_cls: str
    config_cls: str
    config_builder: str
    default_model_path: str
    # NextStep's upstream loader accepts checkpointing only during construction;
    # ordinary AR families do not project the trainer knob into ModelBuild.
    gradient_checkpointing_at_load: bool = False


@dataclass(frozen=True, slots=True)
class ModelFamilyEntry:
    """Declarative runtime binding for one canonical model family."""

    family: str
    task: str
    collector_kind: CollectorKind
    executor_cls: str
    family_build: DiffusionFamilyBuild | ARFamilyBuild

    def __post_init__(self) -> None:
        if self.collector_kind not in get_args(CollectorKind):
            raise ValueError(f"unsupported collector kind: {self.collector_kind!r}")
        diffusion_build = isinstance(self.family_build, DiffusionFamilyBuild)
        if diffusion_build != (self.collector_kind == "diffusion"):
            raise ValueError(
                f"model family {self.family!r} collector kind "
                f"{self.collector_kind!r} does not match its family build",
            )

    def resolve_model_build(
        self,
        cfg: Any,
        device: Any,
        *,
        for_rollout: bool = True,
        parameter_dtype_override: Any | None = None,
    ) -> Any:
        """Project user config into the model inputs owned by this family.

        This is the only config-to-``ModelBuild`` boundary. Family identity was
        already selected by ``get_model_family_entry`` and is never read from
        the config again; ordinary precision comes from the top-level precision
        policy, while genuine family invariants stay on the build descriptor.
        """

        from vrl.config.precision import resolve_precision_policy
        from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
        from vrl.utils.config import plain_mapping

        model_config = plain_mapping(cfg.model, field_name="model")
        configured_dtype = model_config.get("dtype")
        model_path = model_config.get("path")

        if isinstance(self.family_build, DiffusionFamilyBuild):
            family_dtype = self.family_build.base_parameter_dtype
            if configured_dtype is not None:
                reason = (
                    f"its base parameter dtype is fixed to {family_dtype!r} by the "
                    "family build descriptor"
                    if family_dtype is not None
                    else "parameter precision follows the top-level precision block"
                )
                raise ValueError(
                    f"model.dtype is not configurable for diffusion family "
                    f"{self.family!r}; {reason}. Remove model.dtype.",
                )
            if family_dtype is not None and parameter_dtype_override is not None:
                from vrl.models.dtypes import dtype_to_wire_name

                if dtype_to_wire_name(parameter_dtype_override) != dtype_to_wire_name(
                    family_dtype,
                ):
                    raise ValueError(
                        f"diffusion family {self.family!r} requires base parameter "
                        f"dtype {family_dtype!r}; explicit parameter_dtype_override="
                        f"{parameter_dtype_override!r} conflicts with that invariant.",
                    )
            parameter_dtype_override = family_dtype or parameter_dtype_override
        else:
            if configured_dtype is not None:
                raise ValueError(
                    "model.dtype is not configurable for AR families; remove it and "
                    "use the top-level precision block so rollout and replay cannot "
                    "diverge",
                )
            model_path = model_path or self.family_build.default_model_path

        for routing_key in ("executor", "family", "path"):
            model_config.pop(routing_key, None)
        sampling = cfg.get("sampling")
        sampling_config = (
            None if sampling is None else plain_mapping(sampling, field_name="sampling")
        )
        precision = resolve_precision_policy(cfg)
        role_parameter_dtype = precision.rollout.dtype if for_rollout else precision.training.dtype
        parameter_dtype = (
            parameter_dtype_override
            if parameter_dtype_override is not None
            else role_parameter_dtype
        )
        rollout = None
        if for_rollout:
            quantization = precision.rollout.quantization
            rollout = RolloutBuildOptions(
                autocast_dtype=precision.rollout.dtype,
                prompt_encoder_dtype=precision.prompt_encoder_dtype,
                quantization_format=(quantization.format if quantization is not None else None),
                quantization_recipe=(quantization.recipe if quantization is not None else None),
            )
        build = ModelBuild(
            model_name_or_path=str(model_path),
            device=device,
            parameter_dtype=parameter_dtype,
            family=self.family,
            model_config=model_config,
            sampling_config=sampling_config,
            rollout=rollout,
        )

        if (
            isinstance(self.family_build, ARFamilyBuild)
            and self.family_build.gradient_checkpointing_at_load
        ):
            # NextStep's upstream loader accepts this only during construction.
            # Rollout inference never checkpoints, and its bool-only API cannot
            # represent the trainer's selective checkpointing policy.
            from vrl.trainers.activation_checkpointing import (
                resolve_gradient_checkpointing_mode,
            )

            checkpointing = resolve_gradient_checkpointing_mode(cfg)
            if checkpointing == "selective" and not for_rollout:
                raise ValueError(
                    "nextstep_1 replay does not support selective gradient "
                    "checkpointing; use actor.gradient_checkpointing=full or off",
                )
            if build.model_config is not None:
                build.model_config["gradient_checkpointing"] = (
                    not for_rollout and checkpointing == "full"
                )
        return build

    def build_replay(self, build: Any) -> Any:
        """Construct trainer replay through this entry's registered builder."""

        if build.family != self.family:
            raise ValueError(
                f"replay build family {build.family!r} does not match entry {self.family!r}",
            )

        if isinstance(self.family_build, DiffusionFamilyBuild):
            if self.family_build.replay_runtime_builder is not None:
                from vrl.utils.config import import_from_path

                return import_from_path(self.family_build.replay_runtime_builder)(build)
            from vrl.models.diffusion.build import build_family_replay_runtime_bundle

            return build_family_replay_runtime_bundle(build, entry=self)
        from vrl.models.ar.build import build_family_ar_bundle

        return build_family_ar_bundle(build, entry=self, replay=True)

    def build_rollout(self, build: Any) -> Any:
        """Construct a rollout bundle from this entry's resolved model build."""

        if build.family != self.family:
            raise ValueError(
                f"rollout build family {build.family!r} does not match entry {self.family!r}",
            )
        if isinstance(self.family_build, DiffusionFamilyBuild):
            from vrl.models.diffusion.build import build_family_runtime_bundle

            return build_family_runtime_bundle(build, entry=self)
        from vrl.models.ar.build import build_family_ar_bundle

        return build_family_ar_bundle(build, entry=self, replay=False)

    def new_gatherer(self) -> Any:
        """Construct the gatherer implied by the collector protocol."""

        if self.collector_kind == "diffusion":
            from vrl.generation.diffusion.gather import DiffusionChunkGatherer

            return DiffusionChunkGatherer()
        if self.collector_kind == "ar_continuous":
            from vrl.models.ar.nextstep_1.runtime import NextStep1ChunkGatherer

            return NextStep1ChunkGatherer()
        if self.collector_kind == "ar_r1":
            from vrl.models.ar.janus_pro.runtime import JanusProR1ChunkGatherer

            return JanusProR1ChunkGatherer()
        from vrl.generation.ar.executor import ARDiscreteChunkGatherer

        return ARDiscreteChunkGatherer()


FAMILY_REGISTRY: dict[str, ModelFamilyEntry] = {}


def _register_model_family(entry: ModelFamilyEntry) -> ModelFamilyEntry:
    """Register one canonical model family."""

    if entry.family in FAMILY_REGISTRY:
        raise ValueError(f"duplicate model family registration: {entry.family!r}")
    FAMILY_REGISTRY[entry.family] = entry
    return entry


def _diffusion_entry(
    *,
    family: str,
    task: str,
    executor_cls: str | None = None,
    build: DiffusionFamilyBuild,
) -> ModelFamilyEntry:
    # Default dispatch: the shared generic executor. Families with real
    # per-chunk logic pass their own executor_cls. Per-family executor config
    # (num_frames / max_sequence_length / ...) lives in the model yaml's
    # ``executor`` block, read wholesale at launch — not here.
    if executor_cls is None:
        executor_cls = GENERIC_DIFFUSION_EXECUTOR
    return ModelFamilyEntry(
        family=family,
        task=task,
        collector_kind="diffusion",
        executor_cls=executor_cls,
        family_build=build,
    )


def _ar_entry(
    *,
    family: str,
    collector_kind: CollectorKind,
    executor_cls: str,
    ar_build: ARFamilyBuild,
    task: str = "ar_t2i",
) -> ModelFamilyEntry:
    """Construct the uniform registry wiring shared by AR families."""

    return ModelFamilyEntry(
        family=family,
        task=task,
        collector_kind=collector_kind,
        executor_cls=executor_cls,
        family_build=ar_build,
    )


_register_model_family(
    _diffusion_entry(
        family="sd3_5",
        task="t2i",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.sd3_5.model:SD3_5Model",
            replay_cls="vrl.models.diffusion.sd3_5.model:SD3_5ReplayModel",
            transformer_classname="SD3Transformer2DModel",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="flux",
        task="t2i",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.flux.model:FluxModel",
            replay_cls="vrl.models.diffusion.flux.model:FluxReplayModel",
            transformer_classname="FluxTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="qwen_image",
        task="t2i",
        # Descriptor-driven family: the generic functions in
        # vrl.models.diffusion.build read the recipe below, so qwen_image ships
        # no per-family builder/resolver functions.
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.qwen_image.model:QwenImageModel",
            replay_cls="vrl.models.diffusion.qwen_image.model:QwenImageReplayModel",
            transformer_classname="QwenImageTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="sana",
        task="t2i",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.sana.model:SanaModel",
            replay_cls="vrl.models.diffusion.sana.model:SanaReplayModel",
            transformer_classname="SanaTransformer2DModel",
            # SANA linear attention is mantissa-sensitive: fp16 parameters are
            # required even when bf16 autocast supplies forward activation range.
            base_parameter_dtype="fp16",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="lumina2",
        task="t2i",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.lumina2.model:Lumina2Model",
            replay_cls="vrl.models.diffusion.lumina2.model:Lumina2ReplayModel",
            transformer_classname="Lumina2Transformer2DModel",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="hunyuan_video",
        task="t2v",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.hunyuan_video.model:HunyuanVideoModel",
            replay_cls="vrl.models.diffusion.hunyuan_video.model:HunyuanVideoReplayModel",
            transformer_classname="HunyuanVideoTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="mochi",
        task="t2v",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.mochi.model:MochiModel",
            replay_cls="vrl.models.diffusion.mochi.model:MochiReplayModel",
            transformer_classname="MochiTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="hunyuan_image",
        task="t2i",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.hunyuan_image.model:HunyuanImageModel",
            replay_cls="vrl.models.diffusion.hunyuan_image.model:HunyuanImageReplayModel",
            transformer_classname="HunyuanImageTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="pixart_sigma",
        task="t2i",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.pixart_sigma.model:PixArtSigmaModel",
            replay_cls="vrl.models.diffusion.pixart_sigma.model:PixArtSigmaReplayModel",
            transformer_classname="PixArtTransformer2DModel",
            # Epsilon DDPM family (sde_type=ddim): load a DDIMScheduler so the
            # shipped beta config survives into prepare_replay, which rebuilds
            # the rollout's DDIM ladder via pixart_ddim_scheduler.
            scheduler_classname="DDIMScheduler",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="cogvideox",
        task="t2v",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.cogvideox.model:CogVideoXModel",
            replay_cls="vrl.models.diffusion.cogvideox.model:CogVideoXReplayModel",
            transformer_classname="CogVideoXTransformer3DModel",
            # v-prediction DDPM family: replay recomputes log-probs on the
            # same ladder the rollout sampled (sde_type=ddim).
            scheduler_classname="CogVideoXDDIMScheduler",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="wan_2_1",
        task="t2v",
        # The two wan entries carry their own per-variant recipes, so the
        # t2v/i2v resolution is decided here, once, by family selection. The
        # dual-stage transformer_2 late-load lives in the replay model's
        # prepare_replay, so replay is generic too.
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.wan_2_1.model:WanT2VDiffusersModel",
            replay_cls="vrl.models.diffusion.wan_2_1.model:WanT2VReplayModel",
            transformer_classname="WanTransformer3DModel",
            # Replay recomputes log-probs on the schedule the rollout sampled.
            scheduler_classname="UniPCMultistepScheduler",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="wan_2_1_i2v",
        task="i2v",
        executor_cls="vrl.models.diffusion.wan_2_1.runtime:Wan_2_1I2VChunkExecutor",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.wan_2_1.model:WanI2VDiffusersModel",
            replay_cls="vrl.models.diffusion.wan_2_1.model:WanI2VReplayModel",
            transformer_classname="WanTransformer3DModel",
            scheduler_classname="UniPCMultistepScheduler",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="cosmos-predict2",
        task="v2w",
        executor_cls="vrl.models.diffusion.cosmos.predict2.runtime:CosmosChunkExecutor",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.cosmos.predict2.model:CosmosPredict2Model",
            replay_cls="vrl.models.diffusion.cosmos.predict2.model:CosmosPredict2ReplayModel",
            transformer_classname="CosmosTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="cosmos-predict2.5",
        task="t2w",
        executor_cls=(
            "vrl.models.diffusion.cosmos.predict2_5.runtime:CosmosPredict25ChunkExecutor"
        ),
        build=DiffusionFamilyBuild(
            model_cls=("vrl.models.diffusion.cosmos.predict2_5.model:CosmosPredict25Model"),
            replay_cls=("vrl.models.diffusion.cosmos.predict2_5.model:CosmosPredict25ReplayModel"),
            transformer_classname="CosmosTransformer3DModel",
            # Upstream ships UniPC; replay must recompute log-probs under the
            # same schedule the rollout sampled with.
            scheduler_classname="UniPCMultistepScheduler",
            # DiffusionNFT needs the trainable default + frozen previous
            # adapters, which only exist on the LoRA path.
            requires_lora=True,
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="cosmos3",
        task="t2v",
        executor_cls="vrl.models.diffusion.cosmos.cosmos3.runtime:Cosmos3ChunkExecutor",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.cosmos.cosmos3.model:Cosmos3Model",
            replay_runtime_builder=(
                "vrl.models.diffusion.cosmos.cosmos3.runtime:build_cosmos3_replay_runtime_bundle"
            ),
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="cosmos-predict2-anima",
        task="t2i",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.cosmos.anima.model:AnimaModel",
            replay_runtime_builder=(
                "vrl.models.diffusion.cosmos.anima.runtime:build_anima_replay_runtime_bundle"
            ),
        ),
    ),
)

_register_model_family(
    _diffusion_entry(
        family="echo",
        task="t2v",
        executor_cls="vrl.models.diffusion.echo.runtime:EchoChunkExecutor",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.echo.model:EchoModel",
            replay_runtime_builder=(
                "vrl.models.diffusion.echo.runtime:build_echo_replay_runtime_bundle"
            ),
        ),
    ),
)

_JANUS_PRO_BUILD = ARFamilyBuild(
    model_cls="vrl.models.ar.janus_pro.model:JanusProModel",
    replay_cls="vrl.models.ar.janus_pro.model:JanusProReplayModel",
    config_cls="vrl.models.ar.janus_pro.model:JanusProConfig",
    config_builder="vrl.models.ar.janus_pro.runtime:janus_config_from_build",
    default_model_path="deepseek-ai/Janus-Pro-1B",
)

_register_model_family(
    _ar_entry(
        family="janus_pro",
        collector_kind="ar_discrete",
        executor_cls="vrl.models.ar.janus_pro.runtime:JanusProChunkExecutor",
        ar_build=_JANUS_PRO_BUILD,
    ),
)

_register_model_family(
    _ar_entry(
        family="janus_pro_r1",
        collector_kind="ar_r1",
        task="ar_t2i_r1",
        executor_cls="vrl.models.ar.janus_pro.runtime:JanusProR1ChunkExecutor",
        ar_build=_JANUS_PRO_BUILD,
    ),
)

_register_model_family(
    _ar_entry(
        family="nextstep_1",
        collector_kind="ar_continuous",
        executor_cls="vrl.models.ar.nextstep_1.runtime:NextStep1ChunkExecutor",
        ar_build=ARFamilyBuild(
            model_cls="vrl.models.ar.nextstep_1.model:NextStep1Model",
            replay_cls="vrl.models.ar.nextstep_1.model:NextStep1ReplayModel",
            config_cls="vrl.models.ar.nextstep_1.model:NextStep1Config",
            config_builder=("vrl.models.ar.nextstep_1.runtime:nextstep_config_from_build"),
            default_model_path="stepfun-ai/NextStep-1.1",
            gradient_checkpointing_at_load=True,
        ),
    ),
)

_register_model_family(
    _ar_entry(
        family="emu3",
        collector_kind="ar_discrete",
        executor_cls="vrl.models.ar.emu3.runtime:Emu3ChunkExecutor",
        ar_build=ARFamilyBuild(
            model_cls="vrl.models.ar.emu3.model:Emu3Model",
            replay_cls="vrl.models.ar.emu3.model:Emu3ReplayModel",
            config_cls="vrl.models.ar.emu3.model:Emu3Config",
            config_builder="vrl.models.ar.emu3.runtime:emu3_config_from_build",
            default_model_path="BAAI/Emu3-Gen-hf",
        ),
    ),
)

_register_model_family(
    _ar_entry(
        family="glm_image",
        collector_kind="ar_discrete",
        executor_cls="vrl.models.ar.glm_image.runtime:GlmImageChunkExecutor",
        ar_build=ARFamilyBuild(
            model_cls="vrl.models.ar.glm_image.model:GlmImageModel",
            replay_cls="vrl.models.ar.glm_image.model:GlmImageReplayModel",
            config_cls="vrl.models.ar.glm_image.model:GlmImageConfig",
            config_builder="vrl.models.ar.glm_image.runtime:glm_image_config_from_build",
            default_model_path="zai-org/GLM-Image",
        ),
    ),
)

_register_model_family(
    _ar_entry(
        family="llamagen",
        collector_kind="ar_discrete",
        executor_cls="vrl.models.ar.llamagen.runtime:LlamaGenChunkExecutor",
        ar_build=ARFamilyBuild(
            model_cls="vrl.models.ar.llamagen.model:LlamaGenModel",
            replay_cls="vrl.models.ar.llamagen.model:LlamaGenReplayModel",
            config_cls="vrl.models.ar.llamagen.model:LlamaGenConfig",
            config_builder="vrl.models.ar.llamagen.runtime:llamagen_config_from_build",
            default_model_path="peizesun/llamagen_t2i",
        ),
    ),
)


validate_model_family_aliases(FAMILY_REGISTRY)


def get_model_family_entry(family: str) -> ModelFamilyEntry:
    """Return the canonical model-family entry for ``family``."""

    normalized = normalize_model_family(family)
    try:
        return FAMILY_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported model family: {family!r}; registered={sorted(FAMILY_REGISTRY)}",
        ) from exc


__all__ = [
    "FAMILY_REGISTRY",
    "GENERIC_DIFFUSION_EXECUTOR",
    "CollectorKind",
    "ModelFamilyEntry",
    "get_model_family_entry",
]
