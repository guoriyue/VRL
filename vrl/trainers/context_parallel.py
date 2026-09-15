"""Training-side context parallelism through diffusers' ``enable_parallelism``.

A CP group of ranks holds one replica of the (dp-sharded) transformer and
splits the token axis of every sample between them. diffusers owns the plan:
``_cp_plan`` names the module whose input is split (``blocks.0``), the rotary
tables that follow it, and the projection whose output is all-gathered
(``proj_out``), so every rank leaves the transformer with the full-sequence
prediction and the log-prob / loss math runs unchanged on the whole sample.
Both the split and the gather are autograd functions: each rank's backward
carries its own token shard. Parameters shard over the whole world (CP peers
sit inside FSDP's shard axis), so FSDP's
reduce-scatter already sums the shards' contributions; ``FSDPStrategy.backward``
scales the loss by the CP size to undo the extra 1/cp in that mean.
"""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


def enable_context_parallel(
    module: nn.Module,
    *,
    mesh: Any,
    ulysses_degree: int,
    ring_degree: int,
) -> None:
    """Install the diffusers CP hooks on one unwrapped trainable transformer.

    Runs before ``fully_shard``: the hooks bind the module's own ``forward``
    and inspect the class signature, both of which are simplest on the plain
    diffusers class. Families without a ``_cp_plan`` (SD3, Cosmos) fail here
    by name instead of silently training without sequence sharding.
    """

    from diffusers import ContextParallelConfig
    from diffusers.models.modeling_utils import ModelMixin

    if not isinstance(module, ModelMixin):
        raise TypeError(
            f"context parallelism needs a diffusers ModelMixin, got {type(module).__name__}",
        )
    if getattr(module, "_cp_plan", None) is None:
        raise ValueError(
            f"{type(module).__name__} declares no _cp_plan; training-side context "
            "parallelism is only available for diffusers transformers with a plan "
            "(Wan, Flux, Qwen-Image, LTX, ...). Set context_parallel degrees to 1.",
        )
    if int(ring_degree) > 1:
        # Ring attention needs the log-sum-exp from the attention kernel; the
        # default SDPA path cannot return it, cuDNN's can.
        module.set_attention_backend("_native_cudnn")
    install_autocast_safe_attention_ops()
    module.enable_parallelism(
        config=ContextParallelConfig(
            ulysses_degree=int(ulysses_degree),
            ring_degree=int(ring_degree),
            mesh=mesh,
        ),
    )
    logger.info(
        "context parallel enabled on %s: ulysses=%d ring=%d",
        type(module).__name__,
        int(ulysses_degree),
        int(ring_degree),
    )


def _common_attention_dtype(query: Any, key: Any, value: Any) -> Any | None:
    """The dtype the three projections should share, or None when they already do.

    Under an outer autocast the norms that feed q/k run in fp32 while the v
    projection stays in the autocast dtype; the autocast dtype is the intended
    compute dtype. Without autocast, the lowest-precision operand wins, which
    is what the fused kernels would have done.
    """

    import torch

    dtypes = {query.dtype, key.dtype, value.dtype}
    if len(dtypes) == 1:
        return None
    device_type = query.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return min(dtypes, key=lambda dtype: torch.finfo(dtype).bits)


def autocast_safe_attention_forward_op(original: Any) -> Any:
    """Wrap a diffusers attention forward op so its saved q/k/v share one dtype.

    The context-parallel attention path is a custom autograd Function whose
    backward recomputes attention from the tensors saved in forward. Inside the
    forward an outer autocast cast q/k/v to one dtype at the kernel boundary;
    the backward runs outside autocast and hands the raw saved tensors to the
    kernel, which refuses mixed dtypes. Casting before the op saves and runs
    keeps forward numerics identical and makes the backward see what the
    forward saw. Gradients flow back in the cast dtype; autograd restores each
    input's own dtype.
    """

    import functools

    @functools.wraps(original)
    def wrapped(ctx: Any, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        target = _common_attention_dtype(query, key, value)
        if target is not None:
            query, key, value = (x.to(target) for x in (query, key, value))
        return original(ctx, query, key, value, *args, **kwargs)

    wrapped._vrl_autocast_safe = True  # type: ignore[attr-defined]
    return wrapped


def install_autocast_safe_attention_ops() -> None:
    """Patch the native attention forward op once per process (idempotent)."""

    from diffusers.models import attention_dispatch

    current = attention_dispatch._native_attention_forward_op
    if getattr(current, "_vrl_autocast_safe", False):
        return
    attention_dispatch._native_attention_forward_op = autocast_safe_attention_forward_op(current)


__all__ = [
    "autocast_safe_attention_forward_op",
    "enable_context_parallel",
    "install_autocast_safe_attention_ops",
]
