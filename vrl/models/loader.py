"""Shared Diffusers component loading.

Family builders own LoRA/full-finetune setup; ``vrl.nn.optimization`` owns the
rollout passes (quantization, device placement, compilation, memory hooks).
"""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces.runtime import ModelBuild


def resolve_component_class(class_name: str) -> Any:
    """A diffusers class by bare name, or a family-vendored one by ``module:Class``.

    Registry descriptors name upstream components by their diffusers export.
    A checkpoint whose model code ships as remote code (MOSS-Image) is vendored
    under a family package instead, and its descriptor names that class by the
    import-path grammar every other config-declared callable uses.
    """

    if ":" in class_name:
        from vrl.utils.config import import_from_path

        return import_from_path(class_name)
    import diffusers

    return getattr(diffusers, class_name)


def load_diffusers_transformer(
    build: ModelBuild,
    class_name: str,
    *,
    subfolder: str = "transformer",
    materialize_weights: bool = True,
) -> Any:
    """Load only a diffusers transformer component from a model repository.

    ``materialize_weights=False`` builds the module from the repository's
    config alone, with every parameter on the ``meta`` device and buffers
    computed normally: the shape a sharded training strategy fills from the
    primary rank's weights, so only one process ever reads the checkpoint into
    host memory. Nothing under ``subfolder`` but its config is opened.
    """

    transformer_cls = resolve_component_class(class_name)
    if materialize_weights:
        return transformer_cls.from_pretrained(
            build.model_name_or_path,
            subfolder=subfolder,
            torch_dtype=build.parameter_dtype,
            **build.pretrained_kwargs,
        )
    # A skeleton: meta parameters, real buffers, parameter dtypes as a plain
    # from_pretrained cast would set them. A strategy that later fills it from
    # another rank re-syncs any per-parameter exception (diffusers' fp32 pins)
    # before materializing storage.
    from accelerate import init_empty_weights

    config = transformer_cls.load_config(
        build.model_name_or_path,
        subfolder=subfolder,
        **build.pretrained_kwargs,
    )
    with init_empty_weights(include_buffers=False):
        model = transformer_cls.from_config(config)
    return model.to(dtype=build.parameter_dtype)


def load_diffusers_scheduler(
    build: ModelBuild,
    class_name: str,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load only a diffusers scheduler component from a model repository."""

    scheduler_cls = resolve_component_class(class_name)
    scheduler = scheduler_cls.from_pretrained(
        build.model_name_or_path,
        subfolder=subfolder,
        **build.pretrained_kwargs,
    )
    num_steps = build.num_steps
    # Dynamic-shifting schedulers (e.g. FLUX's FlowMatchEulerDiscreteScheduler)
    # derive their timestep/sigma schedule from a resolution-dependent ``mu`` that
    # is unknown here. Eager-setting without ``mu`` raises; defer to the family,
    # which sets the dynamic timesteps once the resolution is known (rollout:
    # prepare_sampling; replay: prepare_replay). Only eager-set the
    # static schedules (SD3.5 / Wan), whose sigmas depend solely on num_steps.
    if num_steps is not None and not getattr(scheduler.config, "use_dynamic_shifting", False):
        scheduler.set_timesteps(num_steps, device=build.device)
    return scheduler


def load_flow_match_scheduler(
    build: ModelBuild,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load the lightweight FlowMatch scheduler needed for replay log-prob math."""

    scheduler = load_diffusers_scheduler(
        build,
        "FlowMatchEulerDiscreteScheduler",
        subfolder=subfolder,
    )
    flow_shift = getattr(scheduler.config, "flow_shift", None)
    if flow_shift is None:
        return scheduler
    # SANA stores the rectified-flow shift under the DPM-facing ``flow_shift``
    # name. Rebuild the lightweight replay scheduler with the same value used by
    # the rollout conversion instead of FlowMatch's unrelated default shift=1.
    scheduler_cls = type(scheduler)
    rebuilt = scheduler_cls.from_config(dict(scheduler.config), shift=float(flow_shift))
    num_steps = build.num_steps
    if num_steps is not None:
        rebuilt.set_timesteps(num_steps, device=build.device)
    return rebuilt


__all__ = [
    "load_diffusers_scheduler",
    "load_diffusers_transformer",
    "load_flow_match_scheduler",
    "resolve_component_class",
]
