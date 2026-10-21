"""Canonical model-family registry.

YAML owns experiment values and defaults. This composition boundary owns the
single family table shared by model construction, generation, and collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from vrl.models.families.names import (
    normalize_model_family,
)
from vrl.models.families.semantics import GenerationRegime, Task

if TYPE_CHECKING:
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import RootConfig

# Import-path protocol value shared by registry dispatch and generation workers.
# Keeping it here avoids making the neutral family table import a runtime module.
GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR = (
    "vrl.generation.bindings.full_sequence.executor:GenericDenoiseBatchExecutor"
)
# Lazy public-sampling schema protocol values. Families share a path only when
# they expose the same real request vocabulary; no per-family empty class is
# needed merely to make registry entries look unique.
TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS = (
    "vrl.config.sampling_schema:TextEncodedImageSamplingSection"
)
VIDEO_SAMPLING_SECTION_CLS = "vrl.config.sampling_schema:VideoSamplingSection"
TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS = (
    "vrl.config.sampling_schema:TextEncodedVideoSamplingSection"
)


@dataclass(frozen=True, slots=True)
class GenerationRuntimeCapabilities:
    """Concrete executor/runtime behaviors, separate from model semantics."""

    supports_torch_compile: bool = False
    # Dotted path of the family's sequence-parallel installer
    # (``module:function`` taking (transformer, process_group)). Its presence
    # IS the multi-GPU engine capability — one source, no capability bool to
    # drift. The launch preflight gate and the rank program both read it.
    sequence_parallel_installer: str | None = None

    @property
    def supports_multi_gpu_engine(self) -> bool:
        """Derived: a family is multi-GPU capable iff it ships an installer."""

        return self.sequence_parallel_installer is not None


@dataclass(frozen=True, slots=True)
class DenoiseFamilyBuild:
    """Declarative build recipe for a descriptor-driven denoise policy.

    A family whose runtime construction is pure data records its build inputs
    here. ``ModelFamilyEntry`` derives the shared resolver and worker builder
    from this descriptor.
    """

    model_cls: str
    # Replay model class. With ``transformer_classname`` the shared builder loads
    # its transformer and scheduler; without it the class loads itself through
    # ``replay_cls.from_build`` (non-diffusers loaders, extra schedulers, grafts).
    replay_cls: str
    # A diffusers export name, or ``module:Class`` for a component the checkpoint
    # ships as remote code and a family package vendors (``resolve_component_class``).
    transformer_classname: str | None = None
    scheduler_classname: str | None = None
    # Driver-side normalization for deployment inputs that must be resolved
    # before ModelBuild is serialized into a Ray launch config.
    model_build_normalizer: str | None = None

    def __post_init__(self) -> None:
        if self.transformer_classname is None and self.scheduler_classname is not None:
            raise ValueError("scheduler_classname requires transformer_classname")


@dataclass(frozen=True, slots=True)
class ModelFamilyEntry:
    """Declarative runtime binding for one canonical model family."""

    family: str
    task: Task
    generation_regime: GenerationRegime
    executor_cls: str
    gatherer_cls: str
    # Lazy public-YAML schema boundary.
    model_section_cls: str
    # Lazy request-vocabulary boundary selected by model.family. This remains
    # separate from mutable runtime sampling state and wire request protocols.
    sampling_section_cls: str
    family_build: DenoiseFamilyBuild
    runtime_capabilities: GenerationRuntimeCapabilities = field(
        default_factory=GenerationRuntimeCapabilities,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.family_build, DenoiseFamilyBuild):
            raise ValueError(
                f"model family {self.family!r} needs a DenoiseFamilyBuild, "
                f"got {type(self.family_build).__name__}",
            )
        if not self.gatherer_cls:
            raise ValueError(f"model family {self.family!r} requires a gatherer binding")

    def validate_gpus_per_engine(self, gpus_per_engine: int) -> None:
        """Gate multi-GPU engines on this family's sequence-parallel install.

        Without one, N ranks would each redundantly compute the full batch —
        N times the GPU cost for zero benefit. The knob therefore either works
        or fails loud here, before any actor launches.
        """

        if gpus_per_engine <= 1:
            return
        if not self.runtime_capabilities.supports_multi_gpu_engine:
            raise ValueError(
                f"model family {self.family!r} does not support multi-GPU "
                "engines, but distributed.resources.rollout.gpus_per_engine="
                f"{gpus_per_engine}. Remove the key (single-GPU engines) or "
                "use a family whose runtime capabilities declare a "
                "sequence_parallel_installer.",
            )

    def validate_model_executor(
        self,
        *,
        executor_config: Any | None,
    ) -> None:
        """Reject executor options when the family does not use the shared executor."""

        from vrl.utils.config import plain_mapping

        executor = (
            {}
            if executor_config is None
            else plain_mapping(executor_config, field_name="model.executor")
        )
        if executor and self.executor_cls != GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR:
            raise ValueError(
                f"model family {self.family!r} does not support model.executor",
            )

    def executor_kwargs(self, root: RootConfig) -> dict[str, Any]:
        """Return this family's complete serializable executor arguments.

        The generic full-sequence executor serves many families, so it also
        receives the family identity and task here; only the live gatherer is
        added where the executor is constructed.
        """

        from vrl.utils.config import plain_mapping

        if root.model is None:
            raise ValueError("validated root is missing model configuration")
        if self.executor_cls != GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR:
            return {}
        kwargs: dict[str, Any] = {"family": self.family, "task": self.task}
        if root.model.executor is not None:
            kwargs.update(plain_mapping(root.model.executor, field_name="model.executor"))
        return kwargs

    def resolve_model_build(
        self,
        root: RootConfig,
        device: Any,
        *,
        precision: PrecisionPolicy,
        for_rollout: bool = True,
        precision_role: Literal["training", "rollout"] | None = None,
        parameter_dtype_override: Any | None = None,
    ) -> Any:
        """Project validated config into the model inputs owned by this family.

        This is the only config-to-``ModelBuild`` boundary. Family identity was
        already selected by ``get_model_family_entry`` and must match the typed
        root. Execution precision is resolved once by the caller and cannot be
        reinterpreted at this boundary.
        """

        from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions

        if root.model is None:
            raise ValueError("validated root is missing model configuration")
        configured_family = normalize_model_family(str(root.model.family))
        if configured_family != self.family:
            raise ValueError(
                f"model config family {configured_family!r} does not match "
                f"registry entry {self.family!r}",
            )

        model_config = root.model.model_dump(
            mode="python",
            exclude_unset=True,
        )
        model_path = model_config.pop("path", None)
        model_revision = model_config.pop("revision", None)
        model_config.pop("family", None)
        model_config.pop("executor", None)
        model_config.pop("memory", None)
        # Rollout residency; replay placement reads ``actor.offload`` instead.
        offload = model_config.pop("offload", None)
        if model_path is None or not str(model_path).strip():
            raise ValueError("config missing required field: model.path")

        sampling_config = (
            None
            if root.sampling is None
            else root.sampling.model_dump(
                mode="python",
                exclude_unset=True,
            )
        )
        resolved_precision_role = precision_role or ("rollout" if for_rollout else "training")
        role_precision = getattr(precision, resolved_precision_role)
        role_parameter_dtype = role_precision.dtype
        parameter_dtype = (
            parameter_dtype_override
            if parameter_dtype_override is not None
            else role_parameter_dtype
        )
        rollout = None
        generation_memory = None
        if for_rollout:
            rollout = RolloutBuildOptions(
                prompt_encoder_dtype=precision.prompt_encoder_dtype,
                # Full-finetune sync replaces base parameters; LoRA sync only
                # sends adapters, so only the former needs retained masters.
                base_weight_sync=not root.model.use_lora,
                offload=offload,
            )
            model_memory = root.model.memory
            if model_memory is not None and model_memory.model_fields_set:
                from vrl.models.interfaces.generation_memory import (
                    GenerationMemoryPolicy,
                    VaeDecodeMemory,
                )

                vae_decode = None
                if "vae_decode" in model_memory.model_fields_set:
                    vae_decode = VaeDecodeMemory(
                        **(
                            {}
                            if model_memory.vae_decode is None
                            else model_memory.vae_decode.model_dump(
                                mode="python",
                                exclude_none=True,
                            )
                        ),
                    )
                generation_memory = GenerationMemoryPolicy(
                    vae_decode=vae_decode,
                    cpu_resident=tuple(model_memory.cpu_resident or ()),
                )
        build = ModelBuild(
            model_name_or_path=str(model_path),
            revision=None if model_revision is None else str(model_revision),
            device=device,
            parameter_dtype=parameter_dtype,
            family=self.family,
            precision=role_precision,
            model_config=model_config,
            sampling_config=sampling_config,
            generation_memory=generation_memory,
            rollout=rollout,
        )

        if self.family_build.model_build_normalizer is not None:
            from vrl.utils.config import import_from_path

            build = import_from_path(self.family_build.model_build_normalizer)(build)
        return build

    def build_replay(
        self,
        build: Any,
        *,
        expected_identity: dict[str, Any] | None,
    ) -> Any:
        """Construct trainer replay through this entry's ``replay_cls``.

        ``expected_identity`` pins the checkpoint the bundle must load (see
        ``ModelBuild.pinned_to``); ``None`` builds without that proof.
        """

        if build.family != self.family:
            raise ValueError(
                f"replay build family {build.family!r} does not match entry {self.family!r}",
            )
        from vrl.models.steps.denoise.build import build_family_replay_runtime_bundle

        with build.pinned_to(expected_identity):
            bundle = build_family_replay_runtime_bundle(build, entry=self)
        return bundle

    def build_rollout(self, build: Any, *, expected_identity: dict[str, Any] | None) -> Any:
        """Construct a rollout bundle from this entry's resolved model build.

        ``expected_identity`` pins the checkpoint the bundle must load (see
        ``ModelBuild.pinned_to``); ``None`` builds without that proof.
        """

        if build.family != self.family:
            raise ValueError(
                f"rollout build family {build.family!r} does not match entry {self.family!r}",
            )
        from vrl.models.steps.denoise.build import build_denoise_runtime_bundle
        from vrl.utils.config import import_from_path

        with build.pinned_to(expected_identity):
            bundle = build_denoise_runtime_bundle(
                build, model_cls=import_from_path(self.family_build.model_cls)
            )
        return bundle

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


def _full_sequence_entry(
    *,
    family: str,
    task: str,
    model_section_cls: str,
    sampling_section_cls: str,
    executor_cls: str | None = None,
    build: DenoiseFamilyBuild,
    runtime_capabilities: GenerationRuntimeCapabilities | None = None,
) -> ModelFamilyEntry:
    # Default dispatch: the shared generic executor. Families with real
    # per-batch logic pass their own executor_cls. Per-family executor config
    # (num_frames / max_sequence_length / ...) lives in the model yaml's
    # ``executor`` block, read wholesale at launch — not here.
    if executor_cls is None:
        executor_cls = GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR
    if runtime_capabilities is None:
        runtime_capabilities = GenerationRuntimeCapabilities(
            supports_torch_compile=True,
        )
    return ModelFamilyEntry(
        family=family,
        task=task,
        generation_regime="full_sequence",
        executor_cls=executor_cls,
        gatherer_cls="vrl.generation.bindings.full_sequence.gather:DenoiseBatchGatherer",
        model_section_cls=model_section_cls,
        sampling_section_cls=sampling_section_cls,
        family_build=build,
        runtime_capabilities=runtime_capabilities,
    )


_register_model_family(
    _full_sequence_entry(
        family="sd3_5",
        task="t2i",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.sd3_5.model:SD3_5PipelineModel",
            replay_cls="vrl.models.families.sd3_5.model:SD3_5Model",
            transformer_classname="SD3Transformer2DModel",
        ),
        # MMDiT joint attention has a verified Ulysses install (numeric
        # N=2-vs-reference equivalence in tests/models/test_sequence_parallel.py);
        # declaring it here is what opens gpus_per_engine > 1.
        runtime_capabilities=GenerationRuntimeCapabilities(
            supports_torch_compile=True,
            sequence_parallel_installer=(
                "vrl.models.sequence_parallel:install_sd3_sequence_parallel"
            ),
        ),
    ),
)


_register_model_family(
    _full_sequence_entry(
        family="flux",
        task="t2i",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.flux.model:FluxPipelineModel",
            replay_cls="vrl.models.families.flux.model:FluxModel",
            transformer_classname="FluxTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="qwen_image",
        task="t2i",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        # Descriptor-driven family: the generic functions in
        # vrl.models.steps.denoise.build read the recipe below, so qwen_image ships
        # no per-family builder/resolver functions.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.qwen_image.model:QwenImagePipelineModel",
            replay_cls="vrl.models.families.qwen_image.model:QwenImageModel",
            transformer_classname="QwenImageTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="qwen_image_layered",
        task="t2i",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls="vrl.config.sampling_schema:QwenImageLayeredSamplingSection",
        executor_cls="vrl.models.families.qwen_image.layered_runtime:QwenImageLayeredBatchExecutor",
        # Qwen-Image-Layered: the qwen_image MMDiT and video VAE loaded through
        # the checkpoint's QwenImageLayeredPipeline; one source image conditions
        # a packed sequence of ``layers`` RGBA layers (see qwen_image/layered.py).
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.qwen_image.layered:QwenImageLayeredPipelineModel",
            replay_cls="vrl.models.families.qwen_image.layered:QwenImageLayeredModel",
            transformer_classname="QwenImageTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="qwen_image_21",
        task="t2i",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls="vrl.config.sampling_schema:QwenImage21SamplingSection",
        executor_cls="vrl.models.families.qwen_image_21.runtime:QwenImage21BatchExecutor",
        # Qwen-Image-2.1: 7B single-stream DiT (diffusers ``QwenImage21*``), a
        # different transformer/VAE/pipeline from the 20B ``qwen_image`` MMDiT.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.qwen_image_21.model:QwenImage21PipelineModel",
            replay_cls="vrl.models.families.qwen_image_21.model:QwenImage21Model",
            transformer_classname="QwenImage21Transformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="krea2",
        task="t2i",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        # Krea 2 (Raw): 12B single-stream MMDiT in the Qwen-Image VAE latent space
        # conditioned on stacked Qwen3-VL hidden states; diffusers-native classes.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.krea2.model:Krea2PipelineModel",
            replay_cls="vrl.models.families.krea2.model:Krea2Model",
            transformer_classname="Krea2Transformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="moss_image",
        task="t2i",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        # MOSS-Image: OpenMOSS 5B single-stream DiT in the FLUX.2 latent space with
        # a Qwen3.5-9B encoder. The checkpoint ships its DiT and scheduler as
        # remote code, vendored in moss_image/pipeline_moss_image.py and named here by path.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.moss_image.model:MossImagePipelineModel",
            replay_cls="vrl.models.families.moss_image.model:MossImageModel",
            transformer_classname="vrl.models.families.moss_image.pipeline_moss_image:MOSSImageTransformer",
            # Replay recomputes log-probs on the training-grid schedule the rollout
            # sampled; the plain flow-match scheduler would rebuild a different grid.
            scheduler_classname="vrl.models.families.moss_image.pipeline_moss_image:MOSSImageScheduler",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="sana",
        task="t2i",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.sana.model:SanaPipelineModel",
            replay_cls="vrl.models.families.sana.model:SanaModel",
            transformer_classname="SanaTransformer2DModel",
        ),
    ),
)


_register_model_family(
    _full_sequence_entry(
        family="wan_2_1",
        task="t2v",
        model_section_cls="vrl.models.families.wan_2_1.config:WanModelSection",
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        # The two wan entries carry their own per-variant recipes, so the
        # t2v/i2v resolution is decided here, once, by family selection. The
        # dual-stage transformer_2 late-load lives in the replay model's
        # prepare_replay, so replay is generic too.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.wan_2_1.model:WanT2VPipelineModel",
            replay_cls="vrl.models.families.wan_2_1.model:WanT2VModel",
            transformer_classname="WanTransformer3DModel",
            # Replay recomputes log-probs on the schedule the rollout sampled.
            scheduler_classname="UniPCMultistepScheduler",
            model_build_normalizer=(
                "vrl.models.families.wan_2_1.config:normalize_wan_model_build"
            ),
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="wan_2_1_i2v",
        task="i2v",
        model_section_cls="vrl.models.families.wan_2_1.config:WanModelSection",
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        executor_cls="vrl.models.families.wan_2_1.runtime:Wan_2_1I2VBatchExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.wan_2_1.model:WanI2VPipelineModel",
            replay_cls="vrl.models.families.wan_2_1.model:WanI2VModel",
            transformer_classname="WanTransformer3DModel",
            scheduler_classname="UniPCMultistepScheduler",
            model_build_normalizer=(
                "vrl.models.families.wan_2_1.config:normalize_wan_model_build"
            ),
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="cosmos-predict2",
        task="v2w",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls=VIDEO_SAMPLING_SECTION_CLS,
        executor_cls="vrl.models.families.cosmos.predict2.runtime:CosmosBatchExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cosmos.predict2.model:CosmosPredict2PipelineModel",
            replay_cls="vrl.models.families.cosmos.predict2.model:CosmosPredict2Model",
            transformer_classname="CosmosTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="cosmos-predict2.5",
        task="t2w",
        model_section_cls=(
            "vrl.models.families.cosmos.predict2_5.config:CosmosPredict25ModelSection"
        ),
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        executor_cls=(
            "vrl.models.families.cosmos.predict2_5.runtime:CosmosPredict25BatchExecutor"
        ),
        build=DenoiseFamilyBuild(
            model_cls=("vrl.models.families.cosmos.predict2_5.model:CosmosPredict25PipelineModel"),
            replay_cls=("vrl.models.families.cosmos.predict2_5.model:CosmosPredict25Model"),
            transformer_classname="CosmosTransformer3DModel",
            # Upstream ships UniPC; replay must recompute log-probs under the
            # same schedule the rollout sampled with.
            scheduler_classname="UniPCMultistepScheduler",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="vdn_h3",
        task="t2v",
        model_section_cls="vrl.models.families.vdn_h3.config:VDNH3ModelSection",
        sampling_section_cls="vrl.config.sampling_schema:MiniMaxH3SamplingSection",
        executor_cls="vrl.models.families.vdn_h3.runtime:VDNH3BatchExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.vdn_h3.model:VDNH3PipelineModel",
            replay_cls="vrl.models.families.vdn_h3.model:VDNH3Model",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="minimax_h3",
        task="t2v",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls="vrl.config.sampling_schema:MiniMaxH3SamplingSection",
        executor_cls="vrl.models.families.minimax_h3.runtime:MiniMaxH3BatchExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.minimax_h3.model:MiniMaxH3PipelineModel",
            replay_cls="vrl.models.families.minimax_h3.model:MiniMaxH3Model",
        ),
    ),
)

_register_model_family(
    _full_sequence_entry(
        family="cosmos3",
        task="t2v",
        model_section_cls="vrl.config.model_schema:ModelSection",
        sampling_section_cls=VIDEO_SAMPLING_SECTION_CLS,
        executor_cls="vrl.models.families.cosmos.cosmos3.runtime:Cosmos3BatchExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cosmos.cosmos3.model:Cosmos3PipelineModel",
            replay_cls="vrl.models.families.cosmos.cosmos3.model:Cosmos3Model",
        ),
    ),
)


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
    "GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR",
    "DenoiseFamilyBuild",
    "GenerationRuntimeCapabilities",
    "ModelFamilyEntry",
    "get_model_family_entry",
]
