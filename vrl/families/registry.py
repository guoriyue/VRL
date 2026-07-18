"""Canonical model-family registry.

YAML owns experiment values and defaults. This composition boundary owns the
single family table shared by model construction, generation, and collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vrl.families.names import (
    normalize_model_family,
    validate_model_family_aliases,
)
from vrl.families.semantics import PolicySemantics, TrajectoryLayout

# Import-path protocol value shared by registry dispatch and generation workers.
# Keeping it here avoids making the neutral family table import a runtime module.
GENERIC_DENOISE_EXECUTOR = "vrl.generation.bindings.joint_denoise.executor:DiffusionChunkExecutor"


@dataclass(frozen=True, slots=True)
class GenerationRuntimeCapabilities:
    """Concrete executor/runtime behaviors, separate from model semantics."""

    supports_torch_compile: bool = False
    accepts_samples_per_chunk: bool = False
    supports_cumem_pool: bool = False
    requires_frozen_component_parking: bool = False


@dataclass(frozen=True, slots=True)
class DenoiseFamilyBuild:
    """Declarative build recipe for a descriptor-driven denoise policy.

    A family whose runtime construction is pure data records its build inputs
    here. ``ModelFamilyEntry`` derives the shared resolver and worker builder
    from whether this descriptor or a ``TokenFamilyBuild`` is present. Only real
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
                "denoise family build must declare either replay_cls plus "
                "transformer_classname, or one custom replay_runtime_builder",
            )
        if custom_replay and self.scheduler_classname is not None:
            raise ValueError(
                "scheduler_classname belongs to the generic replay builder and "
                "cannot accompany replay_runtime_builder",
            )


@dataclass(frozen=True, slots=True)
class TokenFamilyBuild:
    """Declarative model construction for a descriptor-driven token policy."""

    model_cls: str
    replay_cls: str
    config_cls: str
    config_builder: str
    default_model_path: str
    # NextStep's upstream loader accepts checkpointing only during construction;
    # ordinary token families do not project the trainer knob into ModelBuild.
    gradient_checkpointing_at_load: bool = False


@dataclass(frozen=True, slots=True)
class ModelFamilyEntry:
    """Declarative runtime binding for one canonical model family."""

    family: str
    task: str
    policy_semantics: PolicySemantics
    executor_cls: str
    gatherer_cls: str
    family_build: DenoiseFamilyBuild | TokenFamilyBuild
    runtime_capabilities: GenerationRuntimeCapabilities = field(
        default_factory=GenerationRuntimeCapabilities,
    )
    request_metadata_namespace: str | None = None

    def __post_init__(self) -> None:
        denoise_build = isinstance(self.family_build, DenoiseFamilyBuild)
        if denoise_build != (self.policy_semantics.step_kind == "denoise"):
            raise ValueError(
                f"model family {self.family!r} policy semantics "
                f"{self.policy_semantics!r} does not match its family build",
            )
        if not self.gatherer_cls:
            raise ValueError(f"model family {self.family!r} requires a gatherer binding")
        if self.request_metadata_namespace == "":
            raise ValueError("request_metadata_namespace must be non-empty when set")

    def resolve_model_build(
        self,
        cfg: Any,
        device: Any,
        *,
        for_rollout: bool = True,
        precision_role: Literal["training", "rollout"] | None = None,
        parameter_dtype_override: Any | None = None,
    ) -> Any:
        """Project user config into the model inputs owned by this family.

        This is the only config-to-``ModelBuild`` boundary. Family identity was
        already selected by ``get_model_family_entry`` and is never read from
        the config again; execution precision comes from the top-level precision
        policy.
        """

        from vrl.config.precision import resolve_precision_policy
        from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
        from vrl.utils.config import plain_mapping

        model_config = plain_mapping(cfg.model, field_name="model")
        configured_dtype = model_config.get("dtype")
        model_path = model_config.get("path")

        if isinstance(self.family_build, DenoiseFamilyBuild):
            if configured_dtype is not None:
                raise ValueError(
                    f"model.dtype is not configurable for denoise family "
                    f"{self.family!r}; parameter precision follows the top-level "
                    "precision block. Remove model.dtype.",
                )
        else:
            if configured_dtype is not None:
                raise ValueError(
                    "model.dtype is not configurable for token families; remove it and "
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
        resolved_precision_role = precision_role or ("rollout" if for_rollout else "training")
        if resolved_precision_role not in ("training", "rollout"):
            raise ValueError(
                f"precision_role must be 'training' or 'rollout'; got {resolved_precision_role!r}",
            )
        role_precision = getattr(precision, resolved_precision_role)
        role_parameter_dtype = role_precision.dtype
        parameter_dtype = (
            parameter_dtype_override
            if parameter_dtype_override is not None
            else role_parameter_dtype
        )
        rollout = None
        if for_rollout:
            rollout = RolloutBuildOptions(
                prompt_encoder_dtype=precision.prompt_encoder_dtype,
            )
        build = ModelBuild(
            model_name_or_path=str(model_path),
            device=device,
            parameter_dtype=parameter_dtype,
            family=self.family,
            precision=role_precision,
            model_config=model_config,
            sampling_config=sampling_config,
            rollout=rollout,
        )

        if (
            isinstance(self.family_build, TokenFamilyBuild)
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

        if isinstance(self.family_build, DenoiseFamilyBuild):
            if self.family_build.replay_runtime_builder is not None:
                from vrl.utils.config import import_from_path

                return import_from_path(self.family_build.replay_runtime_builder)(build)
            from vrl.models.steps.denoise.build import build_family_replay_runtime_bundle

            return build_family_replay_runtime_bundle(build, entry=self)
        from vrl.models.steps.token.build import build_token_family_bundle

        return build_token_family_bundle(build, entry=self, replay=True)

    def build_rollout(self, build: Any) -> Any:
        """Construct a rollout bundle from this entry's resolved model build."""

        if build.family != self.family:
            raise ValueError(
                f"rollout build family {build.family!r} does not match entry {self.family!r}",
            )
        if isinstance(self.family_build, DenoiseFamilyBuild):
            from vrl.models.steps.denoise.build import build_family_runtime_bundle

            return build_family_runtime_bundle(build, entry=self)
        from vrl.models.steps.token.build import build_token_family_bundle

        return build_token_family_bundle(build, entry=self, replay=False)

    def new_gatherer(self) -> Any:
        """Construct the explicitly bound driver-side gatherer lazily."""

        from vrl.utils.config import import_from_path

        return import_from_path(self.gatherer_cls)()


FAMILY_REGISTRY: dict[str, ModelFamilyEntry] = {}


def _register_model_family(entry: ModelFamilyEntry) -> ModelFamilyEntry:
    """Register one canonical model family."""

    if entry.family in FAMILY_REGISTRY:
        raise ValueError(f"duplicate model family registration: {entry.family!r}")
    FAMILY_REGISTRY[entry.family] = entry
    return entry


def _joint_denoise_entry(
    *,
    family: str,
    task: str,
    executor_cls: str | None = None,
    build: DenoiseFamilyBuild,
    runtime_capabilities: GenerationRuntimeCapabilities | None = None,
) -> ModelFamilyEntry:
    # Default dispatch: the shared generic executor. Families with real
    # per-chunk logic pass their own executor_cls. Per-family executor config
    # (num_frames / max_sequence_length / ...) lives in the model yaml's
    # ``executor`` block, read wholesale at launch — not here.
    if executor_cls is None:
        executor_cls = GENERIC_DENOISE_EXECUTOR
    if runtime_capabilities is None:
        runtime_capabilities = GenerationRuntimeCapabilities(
            supports_torch_compile=True,
            accepts_samples_per_chunk=True,
            supports_cumem_pool=True,
            requires_frozen_component_parking=True,
        )
    return ModelFamilyEntry(
        family=family,
        task=task,
        policy_semantics=PolicySemantics(
            temporal_organization="joint",
            step_kind="denoise",
            action_distribution="continuous",
            trajectory_layout="denoise",
        ),
        executor_cls=executor_cls,
        gatherer_cls="vrl.generation.bindings.joint_denoise.gather:DiffusionChunkGatherer",
        family_build=build,
        runtime_capabilities=runtime_capabilities,
    )


def _causal_token_entry(
    *,
    family: str,
    action_distribution: Literal["categorical", "continuous"],
    executor_cls: str,
    build: TokenFamilyBuild,
    task: str = "ar_t2i",
    trajectory_layout: TrajectoryLayout = "token",
    gatherer_cls: str = "vrl.generation.bindings.causal_token.executor:ARDiscreteChunkGatherer",
    request_metadata_namespace: str | None = None,
) -> ModelFamilyEntry:
    """Construct common wiring for current causal token-policy variants."""

    return ModelFamilyEntry(
        family=family,
        task=task,
        policy_semantics=PolicySemantics(
            temporal_organization="causal",
            step_kind="token",
            action_distribution=action_distribution,
            trajectory_layout=trajectory_layout,
        ),
        executor_cls=executor_cls,
        gatherer_cls=gatherer_cls,
        family_build=build,
        request_metadata_namespace=request_metadata_namespace,
    )


_register_model_family(
    _joint_denoise_entry(
        family="sd3_5",
        task="t2i",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.sd3_5.model:SD3_5Model",
            replay_cls="vrl.models.families.sd3_5.model:SD3_5ReplayModel",
            transformer_classname="SD3Transformer2DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="flux",
        task="t2i",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.flux.model:FluxModel",
            replay_cls="vrl.models.families.flux.model:FluxReplayModel",
            transformer_classname="FluxTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="qwen_image",
        task="t2i",
        # Descriptor-driven family: the generic functions in
        # vrl.models.steps.denoise.build read the recipe below, so qwen_image ships
        # no per-family builder/resolver functions.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.qwen_image.model:QwenImageModel",
            replay_cls="vrl.models.families.qwen_image.model:QwenImageReplayModel",
            transformer_classname="QwenImageTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="sana",
        task="t2i",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.sana.model:SanaModel",
            replay_cls="vrl.models.families.sana.model:SanaReplayModel",
            transformer_classname="SanaTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="lumina2",
        task="t2i",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.lumina2.model:Lumina2Model",
            replay_cls="vrl.models.families.lumina2.model:Lumina2ReplayModel",
            transformer_classname="Lumina2Transformer2DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="hunyuan_video",
        task="t2v",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.hunyuan_video.model:HunyuanVideoModel",
            replay_cls="vrl.models.families.hunyuan_video.model:HunyuanVideoReplayModel",
            transformer_classname="HunyuanVideoTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="mochi",
        task="t2v",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.mochi.model:MochiModel",
            replay_cls="vrl.models.families.mochi.model:MochiReplayModel",
            transformer_classname="MochiTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="hunyuan_image",
        task="t2i",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.hunyuan_image.model:HunyuanImageModel",
            replay_cls="vrl.models.families.hunyuan_image.model:HunyuanImageReplayModel",
            transformer_classname="HunyuanImageTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="pixart_sigma",
        task="t2i",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.pixart_sigma.model:PixArtSigmaModel",
            replay_cls="vrl.models.families.pixart_sigma.model:PixArtSigmaReplayModel",
            transformer_classname="PixArtTransformer2DModel",
            # Epsilon DDPM family (sde_type=ddim): load a DDIMScheduler so the
            # shipped beta config survives into prepare_replay, which rebuilds
            # the rollout's DDIM ladder via pixart_ddim_scheduler.
            scheduler_classname="DDIMScheduler",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="cogvideox",
        task="t2v",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cogvideox.model:CogVideoXModel",
            replay_cls="vrl.models.families.cogvideox.model:CogVideoXReplayModel",
            transformer_classname="CogVideoXTransformer3DModel",
            # v-prediction DDPM family: replay recomputes log-probs on the
            # same ladder the rollout sampled (sde_type=ddim).
            scheduler_classname="CogVideoXDDIMScheduler",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="wan_2_1",
        task="t2v",
        # The two wan entries carry their own per-variant recipes, so the
        # t2v/i2v resolution is decided here, once, by family selection. The
        # dual-stage transformer_2 late-load lives in the replay model's
        # prepare_replay, so replay is generic too.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.wan_2_1.model:WanT2VDiffusersModel",
            replay_cls="vrl.models.families.wan_2_1.model:WanT2VReplayModel",
            transformer_classname="WanTransformer3DModel",
            # Replay recomputes log-probs on the schedule the rollout sampled.
            scheduler_classname="UniPCMultistepScheduler",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="wan_2_1_i2v",
        task="i2v",
        executor_cls="vrl.models.families.wan_2_1.runtime:Wan_2_1I2VChunkExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.wan_2_1.model:WanI2VDiffusersModel",
            replay_cls="vrl.models.families.wan_2_1.model:WanI2VReplayModel",
            transformer_classname="WanTransformer3DModel",
            scheduler_classname="UniPCMultistepScheduler",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="cosmos-predict2",
        task="v2w",
        executor_cls="vrl.models.families.cosmos.predict2.runtime:CosmosChunkExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cosmos.predict2.model:CosmosPredict2Model",
            replay_cls="vrl.models.families.cosmos.predict2.model:CosmosPredict2ReplayModel",
            transformer_classname="CosmosTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="cosmos-predict2.5",
        task="t2w",
        executor_cls=(
            "vrl.models.families.cosmos.predict2_5.runtime:CosmosPredict25ChunkExecutor"
        ),
        build=DenoiseFamilyBuild(
            model_cls=("vrl.models.families.cosmos.predict2_5.model:CosmosPredict25Model"),
            replay_cls=("vrl.models.families.cosmos.predict2_5.model:CosmosPredict25ReplayModel"),
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
    _joint_denoise_entry(
        family="cosmos3",
        task="t2v",
        executor_cls="vrl.models.families.cosmos.cosmos3.runtime:Cosmos3ChunkExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cosmos.cosmos3.model:Cosmos3Model",
            replay_runtime_builder=(
                "vrl.models.families.cosmos.cosmos3.runtime:build_cosmos3_replay_runtime_bundle"
            ),
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="cosmos-predict2-anima",
        task="t2i",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cosmos.anima.model:AnimaModel",
            replay_runtime_builder=(
                "vrl.models.families.cosmos.anima.runtime:build_anima_replay_runtime_bundle"
            ),
        ),
    ),
)

_register_model_family(
    _joint_denoise_entry(
        family="echo",
        task="t2v",
        executor_cls="vrl.models.families.echo.runtime:EchoChunkExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.echo.model:EchoModel",
            replay_runtime_builder=(
                "vrl.models.families.echo.runtime:build_echo_replay_runtime_bundle"
            ),
        ),
    ),
)

_JANUS_PRO_BUILD = TokenFamilyBuild(
    model_cls="vrl.models.families.janus_pro.model:JanusProModel",
    replay_cls="vrl.models.families.janus_pro.model:JanusProReplayModel",
    config_cls="vrl.models.families.janus_pro.model:JanusProConfig",
    config_builder="vrl.models.families.janus_pro.runtime:janus_config_from_build",
    default_model_path="deepseek-ai/Janus-Pro-1B",
)

_register_model_family(
    _causal_token_entry(
        family="janus_pro",
        action_distribution="categorical",
        executor_cls="vrl.models.families.janus_pro.runtime:JanusProChunkExecutor",
        build=_JANUS_PRO_BUILD,
    ),
)

_register_model_family(
    _causal_token_entry(
        family="janus_pro_r1",
        action_distribution="categorical",
        task="ar_t2i_r1",
        executor_cls="vrl.models.families.janus_pro.runtime:JanusProR1ChunkExecutor",
        gatherer_cls="vrl.models.families.janus_pro.runtime:JanusProR1ChunkGatherer",
        build=_JANUS_PRO_BUILD,
        trajectory_layout="multisegment_token",
    ),
)

_register_model_family(
    _causal_token_entry(
        family="nextstep_1",
        action_distribution="continuous",
        executor_cls="vrl.models.families.nextstep_1.runtime:NextStep1ChunkExecutor",
        gatherer_cls="vrl.models.families.nextstep_1.runtime:NextStep1ChunkGatherer",
        build=TokenFamilyBuild(
            model_cls="vrl.models.families.nextstep_1.model:NextStep1Model",
            replay_cls="vrl.models.families.nextstep_1.model:NextStep1ReplayModel",
            config_cls="vrl.models.families.nextstep_1.model:NextStep1Config",
            config_builder=("vrl.models.families.nextstep_1.runtime:nextstep_config_from_build"),
            default_model_path="stepfun-ai/NextStep-1.1",
            gradient_checkpointing_at_load=True,
        ),
        request_metadata_namespace="rollout_metadata",
    ),
)

_register_model_family(
    _causal_token_entry(
        family="emu3",
        action_distribution="categorical",
        executor_cls="vrl.models.families.emu3.runtime:Emu3ChunkExecutor",
        build=TokenFamilyBuild(
            model_cls="vrl.models.families.emu3.model:Emu3Model",
            replay_cls="vrl.models.families.emu3.model:Emu3ReplayModel",
            config_cls="vrl.models.families.emu3.model:Emu3Config",
            config_builder="vrl.models.families.emu3.runtime:emu3_config_from_build",
            default_model_path="BAAI/Emu3-Gen-hf",
        ),
    ),
)

_register_model_family(
    _causal_token_entry(
        family="glm_image",
        action_distribution="categorical",
        executor_cls="vrl.models.families.glm_image.runtime:GlmImageChunkExecutor",
        build=TokenFamilyBuild(
            model_cls="vrl.models.families.glm_image.model:GlmImageModel",
            replay_cls="vrl.models.families.glm_image.model:GlmImageReplayModel",
            config_cls="vrl.models.families.glm_image.model:GlmImageConfig",
            config_builder="vrl.models.families.glm_image.runtime:glm_image_config_from_build",
            default_model_path="zai-org/GLM-Image",
        ),
    ),
)

_register_model_family(
    _causal_token_entry(
        family="llamagen",
        action_distribution="categorical",
        executor_cls="vrl.models.families.llamagen.runtime:LlamaGenChunkExecutor",
        build=TokenFamilyBuild(
            model_cls="vrl.models.families.llamagen.model:LlamaGenModel",
            replay_cls="vrl.models.families.llamagen.model:LlamaGenReplayModel",
            config_cls="vrl.models.families.llamagen.model:LlamaGenConfig",
            config_builder="vrl.models.families.llamagen.runtime:llamagen_config_from_build",
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
    "GENERIC_DENOISE_EXECUTOR",
    "GenerationRuntimeCapabilities",
    "ModelFamilyEntry",
    "PolicySemantics",
    "get_model_family_entry",
]
