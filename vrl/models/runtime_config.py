"""Uniform runtime-spec extraction + shared read helpers.

One extractor carries the runtime-relevant config blocks (``model`` +
``sampling``) wholesale as plain dicts. No per-field curation: each family
reads what it needs from ``spec.model_config`` / ``spec.sampling_config`` in
its ``from_spec`` / ``apply_lora`` / runtime builder; unused fields ride along
harmlessly.

The OLD curated spec fields (``lora_config``, ``scheduler_config`` ...) carried
TRANSFORMED data. The transforms now live in the shared read helpers below so
the per-family read sites stay thin and behaviorally identical.
"""

from __future__ import annotations

from typing import Any

from vrl.config.precision import resolve_precision_policy
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.interfaces.runtime import RuntimeBuildSpec
from vrl.utils.config import plain_mapping


def extract_runtime_spec(
    cfg: Any,
    device: Any,
    dtype: Any,
    *,
    task_variant: str,
    model_name_or_path: Any | None = None,
) -> RuntimeBuildSpec:
    """Build the uniform ``RuntimeBuildSpec`` from a whole training config.

    ``model_config`` carries the entire ``cfg.model`` block (path / lora /
    memory / torch_compile / family-specific knobs); ``sampling_config``
    carries the entire ``cfg.sampling`` block when present. Both are deep
    OmegaConf->plain converted so the Ray launch contract serializes cleanly.

    ``model_name_or_path`` defaults to ``cfg.model.path`` but families with a
    fallback model id pass an explicit default.
    """

    model_config = plain_mapping(cfg.model, field_name="model")
    path = model_name_or_path if model_name_or_path is not None else model_config.get("path")
    sampling_config = _optional_block(cfg, "sampling")
    policy = resolve_precision_policy(cfg)
    frozen_dtype = resolve_torch_dtype(policy.frozen)
    # fp8/fp4 rollout is a quantized GEMM, not a storage dtype: ``dtype`` stays the
    # bf16 master (set by the caller) and the builder swaps linears to fp8 from
    # this token. None for the usual bf16/fp16/fp32 rollout.
    rollout_quantization = policy.rollout if policy.rollout in ("fp8", "fp4") else None
    return RuntimeBuildSpec(
        model_name_or_path=str(path),
        device=device,
        dtype=dtype,
        task_variant=task_variant,
        model_config=model_config,
        sampling_config=sampling_config,
        frozen_dtype=frozen_dtype,
        rollout_quantization=rollout_quantization,
    )


def _optional_block(cfg: Any, field_name: str) -> dict[str, Any] | None:
    block = getattr(cfg, "get", lambda *_: None)(field_name)
    if block is None:
        return None
    return plain_mapping(block, field_name=field_name)


# Read views moved onto ``RuntimeBuildSpec`` as properties (``spec.memory`` /
# ``spec.lora`` / ``spec.num_steps`` / ...), so consumers read them directly
# instead of through free-function wrappers. AR families keep their own
# ``_resolve_lora_block`` for family-default merging.


__all__ = [
    "extract_runtime_spec",
]
