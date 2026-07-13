"""Cosmos Predict2 Anima family runtime."""

from __future__ import annotations

from typing import Any

from vrl.models.diffusion.cosmos.anima.model import _resolve_artifact
from vrl.models.interfaces.runtime import (
    ModelBuild,
    RuntimeBundle,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def resolve_anima_replay_model_build(
    cfg: Any,
    device: Any,
    *,
    parameter_dtype_override: Any | None = None,
) -> ModelBuild:
    """Resolve trainer replay-only Anima model-build inputs.

    With the whole ``cfg.model`` block carried wholesale, the replay model only
    reads the artifact paths / scheduler_shift / torch_compile it needs; the
    remaining fields ride along inertly, so no trimming is required.
    """

    # WHY keep this thin pass-through: it is the *named* replay-path contract,
    # referenced by symbol in train.py and by "module:function" string in the
    # e2e replay test (test_real_checkpoint_rl.py). It intentionally diverges
    # from full-generation extraction — it previously trimmed fields
    # (commit 571277787) and may diverge again; the stable entry point lets the
    # replay build change without touching callers/tests. Do not inline.
    from vrl.models.diffusion.build import resolve_family_model_build

    return resolve_family_model_build(
        cfg,
        device,
        for_rollout=False,
        parameter_dtype_override=parameter_dtype_override,
    )


def build_anima_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Build the trainer replay bundle without Anima generation-only modules."""

    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.models.diffusion.cosmos.anima.model import AnimaReplayModel

    logger.info("Building Anima replay runtime bundle from %s", build.model_name_or_path)

    model_config = build.model_config or {}
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=float(model_config.get("scheduler_shift", 3.0)),
    )
    scheduler.register_to_config(sigma_data=1.0, sigma_max=1.0)
    num_steps = build.num_steps
    if num_steps is not None:
        scheduler.set_timesteps(int(num_steps), device=build.device)

    model = AnimaReplayModel(
        transformer=load_anima_transformer(build),
        scheduler=scheduler,
        device=build.device,
        dtype=build.parameter_dtype,
    )

    # Trainer replay reads bundle.scheduler directly (no prepare_sampling),
    # so the timestep table must be set here.
    if num_steps is not None:
        model.set_num_steps(int(num_steps))

    from vrl.models.diffusion.build import assemble_replay_bundle

    return assemble_replay_bundle(model, build)


def load_anima_transformer(build: ModelBuild) -> Any:
    from safetensors.torch import load_file

    from vrl.models.diffusion.cosmos.anima.model import (
        _load_anima_transformer,
    )
    from vrl.models.loader import model_revision_kwargs

    model_config = build.model_config or {}
    path = model_config.get("transformer_path") or _resolve_artifact(
        str(build.model_name_or_path or ""),
        explicit_path="",
        relative_file=model_config.get("transformer_file", ""),
        field_name="transformer_path",
        **model_revision_kwargs(build),
    )
    if not path:
        raise ValueError("Anima replay model build is missing transformer_path")
    dtype = build.parameter_dtype
    return _load_anima_transformer(load_file(path, device="cpu"), dtype=dtype).to(
        build.device, dtype=dtype
    )


__all__ = [
    "build_anima_replay_runtime_bundle",
    "load_anima_transformer",
    "resolve_anima_replay_model_build",
]
