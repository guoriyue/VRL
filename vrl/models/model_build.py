"""Resolve uniform model-build inputs from the whole training config.

One resolver carries the runtime-relevant config blocks (``model`` +
``sampling``) wholesale as plain dicts. No per-field curation: each family
reads what it needs from ``build.model_config`` / ``build.sampling_config`` in
its ``from_build`` / ``apply_lora`` / runtime builder; unused fields ride along
harmlessly.

The old curated build fields (``lora_config``, ``scheduler_config`` ...) carried
TRANSFORMED data. The transforms now live in the shared read helpers below so
the per-family read sites stay thin and behaviorally identical.
"""

from __future__ import annotations

from typing import Any

from vrl.config.precision import resolve_precision_policy
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
from vrl.utils.config import plain_mapping


def resolve_model_build(
    cfg: Any,
    device: Any,
    *,
    model_name_or_path: Any | None = None,
    family: str | None = None,
    for_rollout: bool = True,
    parameter_dtype_override: Any | None = None,
) -> ModelBuild:
    """Resolve the uniform ``ModelBuild`` from a whole training config.

    ``model_config`` carries the entire ``cfg.model`` block (path / lora /
    memory / torch_compile / family-specific knobs); ``sampling_config``
    carries the entire ``cfg.sampling`` block when present. Both are deep
    OmegaConf->plain converted so the Ray launch contract serializes cleanly.

    ``model_name_or_path`` defaults to ``cfg.model.path`` but families with a
    fallback model id pass an explicit default.

    ``for_rollout=False`` is the trainer replay boundary: generation-only
    precision, quantization, prompt encoders, and weight-sync build state are
    omitted rather than copied into a path that must never consume them. Parameter
    storage follows ``precision.training.dtype`` on replay and
    ``precision.rollout.dtype`` on rollout. ``parameter_dtype_override``
    is reserved for direct tools and genuine family invariants; production callers
    leave it unset.
    """

    model_config = plain_mapping(cfg.model, field_name="model")
    path = model_name_or_path if model_name_or_path is not None else model_config.get("path")
    sampling_config = _optional_block(cfg, "sampling")
    policy = resolve_precision_policy(cfg)
    role_parameter_precision = policy.rollout.dtype if for_rollout else policy.training.dtype
    parameter_dtype = (
        parameter_dtype_override
        if parameter_dtype_override is not None
        else role_parameter_precision
    )
    rollout = None
    if for_rollout:
        # Quantization is layered on rollout.dtype rather than masquerading as a
        # dtype. Unswapped projections, attention score kernels, norms, residuals,
        # and the outer autocast keep this ordinary fp16/bf16/fp32 base dtype.
        quantization = policy.rollout.quantization
        rollout = RolloutBuildOptions(
            autocast_dtype=policy.rollout.dtype,
            prompt_encoder_dtype=policy.prompt_encoder_dtype,
            quantization_format=(quantization.format if quantization is not None else None),
            quantization_recipe=(quantization.recipe if quantization is not None else None),
        )
    return ModelBuild(
        model_name_or_path=str(path),
        device=device,
        parameter_dtype=parameter_dtype,
        family=family,
        model_config=model_config,
        sampling_config=sampling_config,
        rollout=rollout,
    )


def _optional_block(cfg: Any, field_name: str) -> dict[str, Any] | None:
    block = getattr(cfg, "get", lambda *_: None)(field_name)
    if block is None:
        return None
    return plain_mapping(block, field_name=field_name)


# Read views live on ``ModelBuild`` as properties (``build.memory`` /
# ``build.lora`` / ``build.num_steps`` / ...), so consumers read them directly
# instead of through free-function wrappers. AR families merge their defaults
# via ``vrl.models.ar.build.ar_model_config_base``.


__all__ = [
    "resolve_model_build",
]
