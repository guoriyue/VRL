"""Activation checkpointing appliers: off | full | full_cpu | selective | selective_attention.

A general training-setup concern (sibling to precision.py / fsdp.py), not specific
to the online recipe runner: turn the ``actor.gradient_checkpointing`` knob into the
right recompute policy on a trainable transformer.

``full`` recomputes every block (diffusers' default, ~1.3-2x slower backward).
``full_cpu`` also stores checkpoint inputs in pinned CPU memory; this reduces
cross-block GPU activation retention at the cost of host memory and transfers.
``selective`` uses PyTorch SAC (selective activation checkpointing) to save the
expensive ops whose recompute is pure waste (GEMM + attention) and recompute only
the cheap norm/pointwise/dropout -- the native-PyTorch equivalent of Megatron's
selective recompute. Measured win on SD3.5-medium 1024² / 5090: selective recovers
~2/3 of full's recompute tax and reaches batches that no-checkpointing OOMs on. See
archive/main-20261016:docs/sprints/parked/SPRINT_training_mfu_selective_checkpointing.md (P0 results).
``selective_attention`` saves the outputs whose recompute costs the most time
per byte kept -- attention, then the token GEMMs that reduce width (an MLP down
projection: 2K flops per output element for a K wider than the output) -- and
recomputes every other op. Measured on fused Qwen-Image-Layered blocks (17.6k
tokens, batch 1, LoRA r32, 5090), fwd+bwd per block and bytes kept per block:
no checkpointing 110.6 ms / 1846 MB, full 151.4 / 70, attention only
134.0 / 175, attention + down projection 128.1 / 279, every GEMM 115.6 / 1105.
Per MB kept, attention saves 166 us, the down projection 57 us, the remaining
projections 15 us. Over a two-update layered GRPO run the down projection cut
the backward 7% for about 3 GB more peak memory, replay parity still bitwise.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


logger = logging.getLogger(__name__)

# Ops whose recompute is pure waste (GEMM + attention): SAC saves their outputs so
# backward reads them back instead of re-running them; everything cheap (norm,
# pointwise, dropout) is recomputed.
_ATTENTION_SAVE_OPS = {
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
}
_GEMM_OPS = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}
_GRADIENT_CHECKPOINT_SAVE_OPS = _GEMM_OPS | {torch.ops.aten.bmm.default} | _ATTENTION_SAVE_OPS


def _selective_checkpoint_policy(ctx, op, *args, **kwargs):
    if op in _GRADIENT_CHECKPOINT_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def selective_checkpoint_func(module, *args):
    """Drop-in for diffusers' gradient_checkpointing_func that applies SAC.

    ``context_fn`` is a partial, not a lambda: when torch.compile traces this
    call (a root-compiled transformer forward loops over its blocks through
    it), dynamo accepts only a plain function or a partial and raises
    "checkpoint not implemented for NestedUserFunctionVariable context_fn" on
    a lambda created in the traced frame.
    """
    return checkpoint(
        module.__call__,
        *args,
        use_reentrant=False,
        context_fn=functools.partial(
            create_selective_checkpoint_contexts, _selective_checkpoint_policy
        ),
    )


# Fewer rows than this are per-sample conditioning vectors or a short text
# stream: their GEMMs are microseconds, not worth a saved tensor.
_MIN_SAVED_GEMM_ROWS = 64


def _attention_saving_policy(ctx, op, *args, **kwargs):
    if op in _ATTENTION_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    if op in _GEMM_OPS:
        mat1, mat2 = (args[1], args[2]) if op is torch.ops.aten.addmm.default else args[:2]
        width_in, width_out = mat1.shape[-1], mat2.shape[-1]
        if width_in > width_out and mat1.numel() // width_in >= _MIN_SAVED_GEMM_ROWS:
            return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def attention_saving_checkpoint_func(module, *args):
    """``selective_checkpoint_func`` saving attention and width-reducing GEMM outputs."""
    return checkpoint(
        module.__call__,
        *args,
        use_reentrant=False,
        context_fn=functools.partial(
            create_selective_checkpoint_contexts, _attention_saving_policy
        ),
    )


def cpu_checkpoint_func(module, *args):
    """Recompute blocks while storing their checkpoint inputs in pinned CPU memory."""
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        return checkpoint(module.__call__, *args, use_reentrant=False)


def enable_transformer_gradient_checkpointing(bundle: Any, root: RootConfig) -> None:
    """Enable transformer gradient checkpointing while preserving family policy.

    Mode is off | full | full_cpu | selective | selective_attention
    (``ActorSection.gradient_checkpointing_mode``). ``selective`` uses SAC to
    recompute only the cheap ops; ``selective_attention`` saves attention and
    width-reducing GEMM outputs; ``full`` recomputes
    every block. If a module's ``enable_gradient_checkpointing`` cannot accept a
    custom func, either SAC mode falls back to full for that module with a
    warning — never silently, so the run log states what it actually got.
    """

    mode = "off" if root.actor is None else root.actor.gradient_checkpointing_mode
    if mode == "off":
        return
    for name, module in bundle.trainable_modules.items():
        enable = getattr(module, "enable_gradient_checkpointing", None)
        if enable is None:
            raise AttributeError(
                f"trainable module {name!r} does not expose enable_gradient_checkpointing",
            )
        if mode == "full_cpu":
            try:
                enable(gradient_checkpointing_func=cpu_checkpoint_func)
            except TypeError as exc:
                raise ValueError(
                    f"trainable module {name!r} cannot install full_cpu checkpointing; "
                    "a custom gradient_checkpointing_func is required",
                ) from exc
            continue
        if mode in ("selective", "selective_attention"):
            func = (
                selective_checkpoint_func
                if mode == "selective"
                else attention_saving_checkpoint_func
            )
            try:
                inspect.signature(enable).bind_partial(gradient_checkpointing_func=func)
            except TypeError:
                logger.warning(
                    "trainable module %r does not accept a custom gradient_checkpointing_func; "
                    "falling back to full checkpointing for it",
                    name,
                )
            else:
                enable(gradient_checkpointing_func=func)
                continue
        enable()


__all__ = [
    "attention_saving_checkpoint_func",
    "enable_transformer_gradient_checkpointing",
    "selective_checkpoint_func",
]
