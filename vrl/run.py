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

from dataclasses import dataclass
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
from vrl.generation.ray.config import RayGenerationConfig
from vrl.models import checkpoint_identity
from vrl.models.interfaces import ModelBuild, RuntimeBundle
from vrl.ray import resources as ray_resources
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.utils.config import require_exact_int


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
    device = torch.device(ray_resources.trainer_torch_device(resources))
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
    device = torch.device(ray_resources.trainer_torch_device(resources))
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


__all__ = [
    "OnlineRunConfig",
    "ResolvedModel",
    "ResolvedOnlineRun",
    "ResolvedRun",
    "materialize",
    "resolve_model",
    "resolve_online_run",
    "resolve_run",
]
