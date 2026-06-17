"""Typed contracts for common online training recipes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from vrl.models.interfaces import RuntimeModel, require_runtime_model

MetricRowHook = Callable[[dict[str, Any], Any], None]
StepHook = Callable[["OnlineRecipeStack", int, Sequence[Any]], Awaitable[None] | None]
TrainerConfigHook = Callable[[DictConfig, Any], None]
WeightDTypeGetter = Callable[[DictConfig, Any, Any], Any]
CollectorKwargsGetter = Callable[[DictConfig, Sequence[Any]], dict[str, Any]]


def default_model_getter(bundle: Any) -> RuntimeModel:
    """Return the model object exposed by runtime bundles."""

    return require_runtime_model(bundle.model, owner="RuntimeBundle.model")


def default_scheduler_getter(bundle: Any) -> Any | None:
    """Return the scheduler used by flow-matching evaluators when present."""

    return getattr(bundle, "scheduler", None)


@dataclass(frozen=True, slots=True)
class RecipeDeviceContext:
    """Resolved device placement inputs shared by recipe factory and runner."""

    device: Any
    weight_dtype: Any
    distributed_resources: Any | None = None


@dataclass(frozen=True, slots=True)
class OnlineRecipeDefinition:
    """Family-specific hooks kept outside the common online recipe glue."""

    family: str
    build_bundle: Callable[[DictConfig, Any, Any], Any]
    build_replay_bundle: Callable[[DictConfig, Any, Any], Any] | None = None
    model_getter: Callable[[Any], RuntimeModel] = default_model_getter
    scheduler_getter: Callable[[Any], Any | None] = default_scheduler_getter
    reference_model_getter: Callable[[Any, DictConfig], Any | None] | None = None
    export_modules_getter: Callable[[Any, DictConfig], dict[str, Any] | None] | None = None
    after_bundle_built: Callable[[Any, DictConfig], None] | None = None
    configure_trainer: TrainerConfigHook | None = None
    weight_dtype_getter: WeightDTypeGetter | None = None
    collector_kwargs_getter: CollectorKwargsGetter | None = None
    before_step: StepHook | None = None
    after_step: StepHook | None = None
    metric_row_hook: MetricRowHook | None = None
    description: str = ""


@dataclass(slots=True)
class OnlineRecipeStack:
    """Fully wired runtime objects for one online training run."""

    cfg: DictConfig
    definition: OnlineRecipeDefinition
    bundle: Any
    model: RuntimeModel
    reward_fn: Any
    collector: Any
    algorithm: Any
    evaluator: Any | None
    trainer: Any
    # Training strategy seam (vrl/trainers/strategy.py): the owner of trainable
    # -state export for both checkpointing and rollout weight sync. Typed Any to
    # match the other wired runtime objects and avoid importing the protocol.
    strategy: Any
    trainer_config: Any
    collector_config: Any
    family: str
    output_dir: Path
    component_names: tuple[str, ...] = field(default_factory=tuple)
