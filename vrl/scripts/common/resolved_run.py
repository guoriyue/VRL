"""THE single composition seam for training-run config resolution.

Every training entrypoint used to hand-wire the same choreography:
``build_configs`` -> family registry lookup -> distributed resource resolution
-> trainer device (and, for online recipes, run/generation/collector
projections). That choreography lives here, once. New entrypoints call
``resolve_run`` (or ``resolve_online_run``) and read fields off the returned
aggregate -- never re-wire the chain inline.

The composers deliberately call their dependencies through the source modules
(``builders.build_configs``, ``registry.get_model_family_entry``,
``ray_resources.resolve_distributed_resources``) instead of importing the bare
names. Attribute lookup happens at call time, so tests that stub a seam at its
owning module (the established pattern in the recipe test suites) reach the
composer without patching this module too.

Deliberately NOT absorbed here: ``load_training_checkpoint_for_resume`` does
checkpoint file I/O, not config resolution -- it stays in the recipes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from vrl.config import builders
from vrl.config.builders import BuiltConfigs
from vrl.config.schema import RootConfig
from vrl.families import registry
from vrl.families.names import normalize_model_family
from vrl.families.registry import ModelFamilyEntry
from vrl.generation.ray.config import RayGenerationConfig
from vrl.ray import resources as ray_resources
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rollouts.collector.config import RolloutCollectorConfig


def _run_integer(value: object, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer (got {value!r})")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum} (got {value})")
    return value


@dataclass(frozen=True, slots=True)
class OnlineRunConfig:
    """Controller-owned epoch, checkpoint cadence, and prompt RNG policy."""

    total_epochs: int
    save_freq: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        _run_integer(self.total_epochs, path="trainer.total_epochs", minimum=0)
        _run_integer(self.save_freq, path="trainer.save_freq", minimum=0)
        _run_integer(self.seed, path="trainer.seed")

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


__all__ = [
    "OnlineRunConfig",
    "ResolvedOnlineRun",
    "ResolvedRun",
    "resolve_online_run",
    "resolve_run",
]
