"""Shared Diffusers component loading.

Family builders own LoRA/full-finetune setup; ``vrl.nn.optimization`` owns the
rollout passes (quantization, device placement, compilation, memory hooks).
"""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces.runtime import ModelBuild


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

    import diffusers

    transformer_cls = getattr(diffusers, class_name)
    if materialize_weights:
        return transformer_cls.from_pretrained(
            build.model_name_or_path,
            subfolder=subfolder,
            torch_dtype=build.parameter_dtype,
            **build.pretrained_kwargs,
        )
    return skeleton_from_config(
        transformer_cls,
        transformer_cls.load_config(
            build.model_name_or_path,
            subfolder=subfolder,
            **build.pretrained_kwargs,
        ),
        dtype=build.parameter_dtype,
    )


def skeleton_from_config(model_cls: Any, config: Any, *, dtype: Any) -> Any:
    """Construct ``model_cls`` from ``config`` with meta parameters and real buffers.

    The parameter dtypes follow ``dtype`` the way a plain ``from_pretrained``
    cast would; a strategy that later fills the skeleton from another rank
    re-syncs any per-parameter exception (diffusers' fp32 pins) before
    materializing storage.
    """

    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = model_cls.from_config(config)
    return model.to(dtype=dtype)


def load_diffusers_scheduler(
    build: ModelBuild,
    class_name: str,
    *,
    subfolder: str = "scheduler",
) -> Any:
    """Load only a diffusers scheduler component from a model repository."""

    import diffusers

    scheduler_cls = getattr(diffusers, class_name)
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
    # prepare_sampling; replay: build_*_replay_runtime_bundle). Only eager-set the
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
]
