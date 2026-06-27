"""Activation (gradient) checkpointing appliers: off | full | selective.

A general training-setup concern (sibling to precision.py / fsdp.py), not specific
to the online recipe runner: turn the ``actor.gradient_checkpointing`` knob into the
right recompute policy on a trainable transformer.

``full`` recomputes every block (diffusers' default, ~1.3-2x slower backward).
``selective`` uses PyTorch SAC (selective activation checkpointing) to save the
expensive ops whose recompute is pure waste (GEMM + attention) and recompute only
the cheap norm/pointwise/dropout -- the native-PyTorch equivalent of Megatron's
selective recompute. Measured win on SD3.5-medium 1024² / 5090: selective recovers
~2/3 of full's recompute tax and reaches batches that no-checkpointing OOMs on. See
docs/sprints/planned/SPRINT_training_mfu_selective_checkpointing.md (P0 results).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

logger = logging.getLogger(__name__)

# Ops whose recompute is pure waste (GEMM + attention): SAC saves their outputs so
# backward reads them back instead of re-running them; everything cheap (norm,
# pointwise, dropout) is recomputed.
_GRADIENT_CHECKPOINT_SAVE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
}


def _selective_checkpoint_policy(ctx, op, *args, **kwargs):  # noqa: ARG001
    if op in _GRADIENT_CHECKPOINT_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def selective_checkpoint_func(module, *args):
    """Drop-in for diffusers' gradient_checkpointing_func that applies SAC.

    Public so the backward_mfu_probe can measure the exact production policy
    instead of duplicating it.
    """
    return checkpoint(
        module.__call__,
        *args,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(_selective_checkpoint_policy),
    )


def _normalize_gradient_checkpointing(value: Any) -> str:
    """Map the actor.gradient_checkpointing knob to off | full | selective.

    Back-compat: the historical knob was a bool — True -> full, False -> off.
    Strings off/full/selective select the policy explicitly.
    """

    if isinstance(value, bool):
        return "full" if value else "off"
    text = str(value).strip().lower()
    if text in ("", "none", "false", "off", "0"):
        return "off"
    if text in ("true", "full", "1"):
        return "full"
    if text == "selective":
        return "selective"
    raise ValueError(
        f"actor.gradient_checkpointing={value!r} invalid; "
        "use off | full | selective (or a bool: true=full, false=off)",
    )


def enable_transformer_gradient_checkpointing(
    bundle: Any,
    cfg: DictConfig,
    *,
    require_method: bool = True,
) -> None:
    """Enable transformer gradient checkpointing while preserving family policy.

    Mode is off | full | selective (see ``_normalize_gradient_checkpointing``).
    ``selective`` uses SAC to recompute only the cheap ops; ``full`` recomputes
    every block. If a module's ``enable_gradient_checkpointing`` cannot accept a
    custom func, selective falls back to full for that module with a warning —
    never silently, so the run log states what it actually got.
    """

    from vrl.trainers.core.types import TrainerConfig

    # Optional key: base yaml no longer restates the dataclass default, so an
    # absent key means "use the TrainerConfig default" — derived, not copied.
    enabled = OmegaConf.select(cfg, "actor.gradient_checkpointing")
    if enabled is None:
        enabled = TrainerConfig.__dataclass_fields__["gradient_checkpointing"].default
    mode = _normalize_gradient_checkpointing(enabled)
    if mode == "off":
        return

    # compile + manual checkpointing collide: torch.compile traces
    # torch.utils.checkpoint into an InternalTorchDynamoError (measured for both
    # full and selective/SAC), and inductor's AOTAutograd min-cut partitioner
    # already does automatic selective recompute -- so compiling makes manual
    # checkpointing both broken and redundant. Fail at config time with an
    # actionable message instead of a cryptic mid-run dynamo crash. Mirrors the
    # FSDP+compile guard in trainers/strategy.py.
    if bool(OmegaConf.select(cfg, "model.torch_compile.enable")):
        raise ValueError(
            f"actor.gradient_checkpointing={mode!r} cannot combine with "
            "model.torch_compile.enable=true: torch.compile traces "
            "torch.utils.checkpoint into an InternalTorchDynamoError, and its min-cut "
            "partitioner already does automatic selective recompute. Pick one — compile "
            "alone (preferred when it fits memory), or eager + checkpointing.",
        )

    trainable_modules = getattr(bundle, "trainable_modules", None) or {
        "transformer": bundle.model.transformer,
    }
    for name, module in trainable_modules.items():
        enable = getattr(module, "enable_gradient_checkpointing", None)
        if enable is None:
            if require_method:
                raise AttributeError(
                    f"trainable module {name!r} does not expose enable_gradient_checkpointing",
                )
            continue
        if mode == "selective":
            try:
                enable(gradient_checkpointing_func=selective_checkpoint_func)
                continue
            except TypeError:
                logger.warning(
                    "trainable module %r does not accept a custom gradient_checkpointing_func; "
                    "falling back to full checkpointing for it",
                    name,
                )
        enable()


__all__ = [
    "enable_transformer_gradient_checkpointing",
    "selective_checkpoint_func",
]
