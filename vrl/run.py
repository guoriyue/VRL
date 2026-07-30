"""THE single composition seam for training-run config resolution.

Every training entrypoint used to hand-wire the same choreography:
``build_configs`` -> family registry lookup -> distributed resource resolution
-> trainer device (and, for online recipes, run/generation/collector
projections). That choreography lives here, once. New entrypoints call
``resolve_run`` (or ``resolve_online_run``) and read fields off the returned
aggregate -- never re-wire the chain inline.

Model materialization is the same story: the hand-wired chain
``resolve_model_build`` -> ``resolve_checkpoint_model_identity`` -> bundle
build -> identity recheck lives here as ``resolve_model`` + ``materialize``.
The two stages are deliberately separate: the online recipe runs its
checkpoint-compatibility preflight between identity resolution and heavy
bundle construction, so the seam must not fuse them.

Ray worker composition follows the same ownership rule:
``resolve_ray_generation_launch_inputs`` projects one resolved online run plus
its replay-model identity into a typed, serializable launch payload. The Ray
launcher consumes that payload and topology only; it never reinterprets trainer
schedule or model YAML.

The composers deliberately call their dependencies through the source modules
(``builders.build_configs``, ``registry.get_model_family_entry``,
``ray_resources.resolve_distributed_resources``,
``checkpoint_identity.resolve_checkpoint_model_identity``) instead of
importing the bare names. Attribute lookup happens at call time, so tests that
stub a seam at its owning module (the established pattern in the recipe test
suites) reach the composer without patching this module too.

Deliberately NOT absorbed here: ``load_training_checkpoint_for_resume`` does
checkpoint file I/O, not config resolution -- it stays in the recipes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import torch
from omegaconf import DictConfig

from vrl.config import builders
from vrl.config.builders import BuiltConfigs
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import RootConfig
from vrl.families import registry
from vrl.families.names import normalize_model_family
from vrl.families.registry import ModelFamilyEntry
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.models import checkpoint_identity
from vrl.models.dtypes import dtype_to_wire_name
from vrl.models.interfaces import ModelBuild, RuntimeBundle
from vrl.ray import resources as ray_resources
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.utils.config import cfg_path, require_exact_int, to_builtin_deep


@dataclass(frozen=True, slots=True)
class OnlineRunConfig:
    """Controller-owned epoch, checkpoint cadence, and prompt RNG policy."""

    total_epochs: int
    save_freq: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        require_exact_int(self.total_epochs, path="trainer.total_epochs", minimum=0)
        require_exact_int(self.save_freq, path="trainer.save_freq", minimum=0)
        require_exact_int(self.seed, path="trainer.seed")

    @classmethod
    def from_root(cls, root: RootConfig) -> OnlineRunConfig:
        trainer = root.trainer
        total_epochs = None if trainer is None else trainer.total_epochs
        if total_epochs is None:
            raise ValueError("config missing required key: trainer.total_epochs")
        values: dict[str, Any] = {"total_epochs": total_epochs}
        if trainer.save_freq is not None:
            values["save_freq"] = trainer.save_freq
        if trainer.seed is not None:
            values["seed"] = trainer.seed
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """Core resolution shared by every training entrypoint."""

    built: BuiltConfigs
    family: ModelFamilyEntry
    resources: ResolvedDistributedResources
    device: torch.device


@dataclass(frozen=True, slots=True)
class ResolvedOnlineRun(ResolvedRun):
    """Online-recipe resolution: the shared core plus run/generation/collector."""

    run: OnlineRunConfig
    generation: RayGenerationConfig
    collector: RolloutCollectorConfig


def _model_family(built: BuiltConfigs) -> ModelFamilyEntry:
    """Registry entry for the configured model family (canonical name)."""

    if built.root.model is None:
        raise ValueError("online recipe requires model configuration")
    return registry.get_model_family_entry(
        normalize_model_family(str(built.root.model.family)),
    )


def resolve_run(cfg: DictConfig) -> ResolvedRun:
    """Resolve the shared training-run core from one merged config."""

    built = builders.build_configs(cfg)
    family = _model_family(built)
    resources = ray_resources.resolve_distributed_resources(
        cfg,
        reward_inference=built.reward.inference_configs if built.reward else None,
    )
    device = torch.device(resources.trainer_torch_device)
    return ResolvedRun(
        built=built,
        family=family,
        resources=resources,
        device=device,
    )


def resolve_online_run(cfg: DictConfig) -> ResolvedOnlineRun:
    """Resolve everything the online recipe reads before heavy construction.

    The composition order mirrors the historical inline order in
    ``run_online_recipe`` (built -> run -> family -> resources -> generation ->
    device, collector last) so every fail-fast validation fires in the same
    relative order it always did.
    """

    built = builders.build_configs(cfg)
    run = OnlineRunConfig.from_root(built.root)
    family = _model_family(built)
    resources = ray_resources.resolve_distributed_resources(
        cfg,
        reward_inference=built.reward.inference_configs if built.reward else None,
    )
    generation = RayGenerationConfig.from_cfg(
        built.root,
        resources=resources,
    )
    device = torch.device(resources.trainer_torch_device)
    collector = RolloutCollectorConfig.from_cfg(built.root)
    return ResolvedOnlineRun(
        built=built,
        family=family,
        resources=resources,
        device=device,
        run=run,
        generation=generation,
        collector=collector,
    )


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Cheap model projection: family build inputs plus checkpoint identity."""

    entry: ModelFamilyEntry
    build: ModelBuild
    identity: dict[str, Any]


def resolve_model(
    entry: ModelFamilyEntry,
    root: RootConfig,
    device: torch.device,
    *,
    precision: PrecisionPolicy,
    for_rollout: bool,
    precision_role: Literal["training", "rollout"] | None = None,
) -> ResolvedModel:
    """Project validated config into a ``ModelBuild`` and resolve its identity.

    This is the cheap stage of model materialization. Callers run their own
    preflights (checkpoint-compatibility, prompt validation) against the
    returned identity before paying for ``materialize``.
    """

    build = entry.resolve_model_build(
        root,
        device,
        precision=precision,
        for_rollout=for_rollout,
        precision_role=precision_role,
    )
    identity = checkpoint_identity.resolve_checkpoint_model_identity(build)
    return ResolvedModel(entry=entry, build=build, identity=identity)


def materialize(
    resolved: ResolvedModel,
    *,
    replay: bool = False,
    context: str,
) -> RuntimeBundle:
    """Build the heavy runtime bundle, then re-verify checkpoint identity.

    ``context`` names the construction step in the mismatch error (for example
    ``"replay bundle construction"``). Each call site's historical wording is
    pinned by its tests, so the caller owns the fragment.
    """

    entry = resolved.entry
    bundle = entry.build_replay(resolved.build) if replay else entry.build_rollout(resolved.build)
    loaded_identity = checkpoint_identity.resolve_checkpoint_model_identity(resolved.build)
    if loaded_identity != resolved.identity:
        raise RuntimeError(
            f"model checkpoint source changed during {context}; "
            f"before={resolved.identity!r}, after={loaded_identity!r}",
        )
    return bundle


def resolve_ray_generation_launch_inputs(
    run: ResolvedOnlineRun,
    *,
    replay_model: ResolvedModel,
) -> RayGenerationLaunchInputs:
    """Resolve validated run state into the serializable Ray worker contract."""

    if not isinstance(run, ResolvedOnlineRun):
        raise TypeError(
            f"run must be a ResolvedOnlineRun, got {type(run).__name__}",
        )
    if not isinstance(replay_model, ResolvedModel):
        raise TypeError(
            f"replay_model must be a ResolvedModel, got {type(replay_model).__name__}",
        )
    if replay_model.entry.family != run.family.family:
        raise ValueError(
            "replay model family does not match the resolved online run: "
            f"{replay_model.entry.family!r} != {run.family.family!r}",
        )
    replay_model.build.require_replay()
    trainer = run.built.trainer
    if trainer is None:
        raise ValueError("online generation launch requires a trainer config")

    generation = run.generation
    runtime_device = torch.device(
        "cuda" if generation.resources.rollout_gpus_per_worker > 0 else "cpu",
    )
    build = run.family.resolve_model_build(
        run.built.root,
        runtime_device,
        precision=run.built.precision,
    )
    if build.rollout is not None:
        build.rollout = replace(
            build.rollout,
            # Full-finetune sync replaces base parameters; LoRA sync only sends
            # adapters, so only the former needs retained base-weight masters.
            base_weight_sync=(generation.worker.sync_trainable_state and not build.use_lora),
        )
    if (
        build.torch_compile is not None
        and not run.family.runtime_capabilities.supports_torch_compile
    ):
        raise ValueError(
            f"{run.family.family} does not support torch compile but "
            "model.torch_compile.enable is set",
        )

    rollout_model_identity = checkpoint_identity.resolve_checkpoint_model_identity(build)
    if rollout_model_identity != replay_model.identity:
        raise ValueError(
            "rollout model identity does not match the driver replay model "
            "identity before Ray worker launch: "
            f"replay={replay_model.identity!r}, rollout={rollout_model_identity!r}",
        )

    return RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=run.family.family,
            model_build=_model_build_payload(build),
            expected_model_identity=replay_model.identity,
            executor_kwargs=run.family.resolve_executor_kwargs(run.built.root),
            policy_version=0,
            torch_profiler=_runtime_profiler(run.built.root),
            # The typed trainer schedule is the source of truth for whether a
            # worker may retain an older LoRA slot across non-draining sync.
            versioned_weight_sync=(
                generation.worker.sync_trainable_state
                and trainer.rollout_orchestration.schedule_mode == "continuous"
                and build.use_lora
            ),
        ),
        gatherer=run.family.new_gatherer(),
    )


def _model_build_payload(build: ModelBuild) -> dict[str, Any]:
    payload = asdict(build)
    # Family is the launch contract identity and is restored worker-side. Do not
    # serialize it again inside the nested model-build payload.
    payload.pop("family", None)
    payload["device"] = str(payload["device"])
    payload["parameter_dtype"] = dtype_to_wire_name(payload["parameter_dtype"])
    rollout = payload.get("rollout")
    if rollout is not None:
        rollout["prompt_encoder_dtype"] = dtype_to_wire_name(
            rollout["prompt_encoder_dtype"],
        )
        # Absence is the wire representation of the universal no-offload
        # default. Only a selected pipeline residency mode needs to cross Ray.
        if rollout.get("pipeline_offload_mode") == "none":
            rollout.pop("pipeline_offload_mode")
    return payload


def _runtime_profiler(root: RootConfig) -> dict[str, Any]:
    profiler_cfg = cfg_path(root, "rollout.torch_profiler", None)
    if profiler_cfg is None:
        return {}
    profiler = to_builtin_deep(profiler_cfg)
    if not isinstance(profiler, dict):
        return {}
    # trainer.output_dir is required by the public config, so the worker payload
    # needs no independent path fallback.
    profiler["output_dir"] = str(cfg_path(root, "trainer.output_dir"))
    return profiler


__all__ = [
    "OnlineRunConfig",
    "ResolvedModel",
    "ResolvedOnlineRun",
    "ResolvedRun",
    "materialize",
    "resolve_model",
    "resolve_online_run",
    "resolve_ray_generation_launch_inputs",
    "resolve_run",
]
