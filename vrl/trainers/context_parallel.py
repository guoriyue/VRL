"""Training-side context parallelism through diffusers' ``enable_parallelism``.

A CP group of ranks holds one replica of the (dp-sharded) transformer and
splits the token axis of every sample between them. diffusers owns the plan:
``_cp_plan`` names the module whose input is split (``blocks.0``), the rotary
tables that follow it, and the projection whose output is all-gathered
(``proj_out``), so every rank leaves the transformer with the full-sequence
prediction and the log-prob / loss math runs unchanged on the whole sample.
Both the split and the gather are autograd functions: each rank's backward
carries its own token shard. Parameters shard over the whole world (CP peers
sit inside FSDP's shard axis, the miles_diffusion layout), so FSDP's
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


__all__ = ["enable_context_parallel"]
