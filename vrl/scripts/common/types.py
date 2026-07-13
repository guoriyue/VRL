"""Typed contracts for common online training recipes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig

from vrl.models.interfaces import ModelBuild, RuntimeBundle

CollectorKwargsGetter = Callable[[DictConfig, Sequence[Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class OnlineRecipeDefinition:
    """Family-specific hooks kept outside the common online recipe glue."""

    # Trainer-side builder for the config-resolved replay ModelBuild. The
    # rollout (generation) model is built inside Ray workers from the registry
    # launch contract, so the driver never builds a rollout bundle of its own.
    build_replay_bundle: Callable[[ModelBuild], RuntimeBundle]
    reference_model_getter: Callable[[Any, DictConfig], Any | None] | None = None
    export_modules_getter: Callable[[Any, DictConfig], dict[str, Any] | None] | None = None
    after_bundle_built: Callable[[Any, DictConfig], None] | None = None
    collector_kwargs_getter: CollectorKwargsGetter | None = None


@dataclass(slots=True)
class OnlineRecipeStack:
    """Wired runtime objects consumed by the recipe's IO controller.

    ``OnlineRecipeRun`` (vrl/scripts/common/online.py) reads these for the
    metrics-CSV rows and the checkpoint save; nothing else consumes the stack.
    """

    cfg: DictConfig
    definition: OnlineRecipeDefinition
    bundle: Any
    reward_fn: Any
    trainer: Any
    # Training strategy seam (vrl/trainers/strategy.py): the owner of trainable
    # -state export for both checkpointing and rollout weight sync. Typed Any to
    # match the other wired runtime objects and avoid importing the protocol.
    strategy: Any
    family: str
    component_names: tuple[str, ...] = field(default_factory=tuple)
