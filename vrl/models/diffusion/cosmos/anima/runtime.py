"""Cosmos Predict2 Anima family runtime."""

from __future__ import annotations

from typing import Any

from vrl.models.diffusion.cosmos.anima.model import _resolve_artifact
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

ANIMA_FAMILY = "cosmos-predict2-anima"
def extract_anima_replay_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBuildSpec:
    """Trainer replay-only Anima runtime spec.

    With the whole ``cfg.model`` block carried wholesale, the replay model only
    reads the artifact paths / scheduler_shift / torch_compile it needs; the
    remaining fields ride along inertly, so no trimming is required.
    """

    # WHY keep this thin pass-through: it is the *named* replay-path contract,
    # referenced by symbol in train.py and by "module:function" string in the
    # e2e replay test (test_real_checkpoint_rl.py). It intentionally diverges
    # from full-generation extraction — it previously trimmed fields
    # (commit 571277787) and may diverge again; the stable entry point lets the
    # replay spec change without touching callers/tests. Do not inline.
    from vrl.models.diffusion.build import extract_family_runtime_spec

    return extract_family_runtime_spec(cfg, device, weight_dtype)


def build_anima_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without Anima generation-only modules."""

    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.models.diffusion.cosmos.anima.model import AnimaReplayModel
    from vrl.models.dtypes import resolve_torch_dtype

    logger.info("Building Anima replay runtime bundle from %s", spec.model_name_or_path)

    model_config = spec.model_config or {}
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=float(model_config.get("scheduler_shift", 3.0)),
    )
    scheduler.register_to_config(sigma_data=1.0, sigma_max=1.0)
    num_steps = spec.num_steps
    if num_steps is not None:
        scheduler.set_timesteps(int(num_steps), device=spec.device)

    model = AnimaReplayModel(
        transformer=load_anima_transformer(spec),
        scheduler=scheduler,
        device=spec.device,
        dtype=resolve_torch_dtype(spec.dtype),
    )

    # Trainer replay reads bundle.scheduler directly (no prepare_sampling),
    # so the timestep table must be set here.
    if num_steps is not None:
        model.set_num_steps(int(num_steps))

    from vrl.models.diffusion.build import assemble_replay_bundle

    return assemble_replay_bundle(model, spec, family=ANIMA_FAMILY)
def load_anima_transformer(spec: RuntimeBuildSpec) -> Any:
    from safetensors.torch import load_file

    from vrl.models.diffusion.cosmos.anima.model import (
        _load_anima_transformer,
    )
    from vrl.models.dtypes import resolve_torch_dtype

    model_config = spec.model_config or {}
    path = model_config.get("transformer_path") or _resolve_artifact(
        str(spec.model_name_or_path or ""),
        explicit_path="",
        relative_file=model_config.get("transformer_file", ""),
        field_name="transformer_path",
    )
    if not path:
        raise ValueError("Anima replay runtime spec is missing transformer_path")
    dtype = resolve_torch_dtype(spec.dtype)
    return _load_anima_transformer(load_file(path, device="cpu"), dtype=dtype).to(spec.device, dtype=dtype)


__all__ = [
    "ANIMA_FAMILY",
    "build_anima_replay_runtime_bundle",
    "extract_anima_replay_runtime_spec",
    "load_anima_transformer",
]
