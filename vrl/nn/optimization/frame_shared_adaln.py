"""Cosmos AdaLN evaluated once per frame instead of once per token.

Cosmos Predict2 / 2.5 condition every latent frame on its own timestep (the
conditioning frames sit at ``conditional_frame_timestep``, the sampled frames at
the step's sigma), so the runner hands the transformer a ``[B, 1, T, 1, 1]``
timestep. The diffusers forward embeds those ``B*T`` timesteps and then expands
the result to one row per TOKEN -- ``[B, THW, C]`` for ``embedded_timestep`` and
``[B, THW, 3C]`` for ``temb`` -- before the blocks run. Every AdaLN site (three
per block plus the output norm) then runs its whole conditioning chain over all
``THW`` tokens::

    silu(embedded_timestep)      # [B, THW, C]
    linear_1                     # C   -> 256, M = B*THW rows
    linear_2                     # 256 -> 3C,  M = B*THW rows
    + temb                       # [B, THW, 3C]
    chunk -> shift, scale, gate  # [B, THW, C] each

although only ``T`` distinct rows exist per sample (``T`` is 9-24 where ``THW``
is 14k-40k on the video shapes VRL trains). Per site at 480p/33f, batch 4 with
CFG, that is ~25 full-width activation passes of memory traffic and a
``[B*THW, C] x [C, 256]`` plus ``[B*THW, 256] x [256, 3C]`` GEMM pair; over
85 sites it is a measurable share of an eager step and is not removed by
inductor, which fuses the elementwise chain but still runs the GEMMs and the
SiLU over every row.

The swap runs the chain on the ``[B, T, C]`` frame rows -- one row per (sample,
frame) picked out of the expanded tensors as a strided view -- and applies the
resulting ``[B, T, 1, C]`` shift / scale to a ``[B, T, HW, C]`` view of the
normalized hidden states by broadcasting. The one full-width write that remains
is the gate, which the block consumes as a ``[B, THW, C]`` operand.

The per-token layout is not recoverable from the ``[B, THW, C]`` tensors the
norms receive, so the transformer root records it per forward: a pre-hook reads
``(T, H, W)`` off the 5-D latent input and the model's patch size into a
:class:`FrameLayout` shared by the root and every swapped norm. A norm whose
sequence length does not match the recorded layout (a 1-D timestep, a call that
bypassed the root) takes the reference forward, which is always correct.

Numerics: the elementwise chain (SiLU, ``+ temb``, modulate) is the same
operation on the same values, so it rounds identically; only the two small
GEMMs run over fewer rows, and a GEMM's per-row result may differ at the last
bit across row counts (kernel selection). Rollout and replay both apply the swap
from ``model.frame_shared_adaln``, so the two roles keep one path and the
log-prob ratio sees no new drift.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import nn


class FrameLayout:
    """The token layout of the current forward, written by the root's pre-hook.

    ``seq_len`` is the flattened token count the layout was recorded for, so a
    norm can tell a stale or foreign input apart from the one the hook saw.
    ``None`` means the current forward is not frame-conditioned.
    """

    __slots__ = ("seq_len", "tokens_per_frame")

    def __init__(self) -> None:
        self.seq_len: int | None = None
        self.tokens_per_frame: int | None = None


def _cosmos_classes() -> tuple[type[nn.Module], type[nn.Module], type[nn.Module]] | None:
    try:
        from diffusers.models.transformers.transformer_cosmos import (
            CosmosAdaLayerNorm,
            CosmosAdaLayerNormZero,
            CosmosTransformer3DModel,
        )
    except ImportError:  # pragma: no cover - diffusers is a hard dependency of the family
        return None
    return CosmosTransformer3DModel, CosmosAdaLayerNormZero, CosmosAdaLayerNorm


def _frame_rows(
    expanded: torch.Tensor, batch: int, frames: int, tokens_per_frame: int
) -> torch.Tensor:
    """The ``[B, T, *]`` rows of a tensor expanded to ``[B, THW, *]`` frame-major."""

    return expanded.view(batch, frames, tokens_per_frame, -1)[:, :, 0]


@functools.cache
def _frame_shared_classes() -> tuple[type[nn.Module], type[nn.Module]]:
    classes = _cosmos_classes()
    assert classes is not None
    _, zero_base, plain_base = classes

    class FrameSharedAdaLayerNormZero(zero_base):  # type: ignore[misc,valid-type]
        """Cosmos ``AdaLayerNormZero`` whose conditioning chain runs per frame.

        Installed by re-classing the diffusers instance: ``linear_1`` /
        ``linear_2`` / ``norm`` and their state_dict names are untouched, and the
        reference forward stays available for inputs the layout does not cover.
        """

        frame_layout: FrameLayout

        def forward(
            self,
            hidden_states: torch.Tensor,
            embedded_timestep: torch.Tensor,
            temb: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            layout = self.frame_layout
            if embedded_timestep.ndim != 3 or hidden_states.shape[1] != layout.seq_len:
                return super().forward(hidden_states, embedded_timestep, temb)
            batch, seq_len, _ = hidden_states.shape
            tokens_per_frame = layout.tokens_per_frame
            frames = seq_len // tokens_per_frame

            rows = _frame_rows(embedded_timestep, batch, frames, tokens_per_frame)
            rows = self.linear_2(self.linear_1(self.activation(rows)))
            if temb is not None:
                rows = rows + _frame_rows(temb, batch, frames, tokens_per_frame)
            shift, scale, gate = rows.unsqueeze(2).chunk(3, dim=-1)  # [B, T, 1, C]

            normed = self.norm(hidden_states).view(batch, frames, tokens_per_frame, -1)
            hidden_states = (normed * (1 + scale) + shift).view(batch, seq_len, -1)
            gate = gate.expand(-1, -1, tokens_per_frame, -1).reshape(batch, seq_len, -1)
            return hidden_states, gate

    class FrameSharedAdaLayerNorm(plain_base):  # type: ignore[misc,valid-type]
        """Cosmos output ``AdaLayerNorm`` whose conditioning chain runs per frame."""

        frame_layout: FrameLayout

        def forward(
            self,
            hidden_states: torch.Tensor,
            embedded_timestep: torch.Tensor,
            temb: torch.Tensor | None = None,
        ) -> torch.Tensor:
            layout = self.frame_layout
            if embedded_timestep.ndim != 3 or hidden_states.shape[1] != layout.seq_len:
                return super().forward(hidden_states, embedded_timestep, temb)
            batch, seq_len, _ = hidden_states.shape
            tokens_per_frame = layout.tokens_per_frame
            frames = seq_len // tokens_per_frame

            rows = _frame_rows(embedded_timestep, batch, frames, tokens_per_frame)
            rows = self.linear_2(self.linear_1(self.activation(rows)))
            if temb is not None:
                temb_rows = _frame_rows(temb, batch, frames, tokens_per_frame)
                rows = rows + temb_rows[..., : 2 * self.embedding_dim]
            shift, scale = rows.unsqueeze(2).chunk(2, dim=-1)  # [B, T, 1, C]

            normed = self.norm(hidden_states).view(batch, frames, tokens_per_frame, -1)
            return (normed * (1 + scale) + shift).view(batch, seq_len, -1)

    return FrameSharedAdaLayerNormZero, FrameSharedAdaLayerNorm


def _record_frame_layout(root: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Root pre-hook: note the frame layout of a 5-D-timestep forward, clear it otherwise."""

    layout: FrameLayout = root.frame_layout
    layout.seq_len = layout.tokens_per_frame = None
    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
    if not isinstance(timestep, torch.Tensor) or timestep.ndim != 5:
        return
    _, _, frames, height, width = hidden_states.shape
    p_t, p_h, p_w = root.config.patch_size
    layout.tokens_per_frame = (height // p_h) * (width // p_w)
    layout.seq_len = (frames // p_t) * layout.tokens_per_frame


def share_adaln_across_frames(root: nn.Module) -> int:
    """Re-class every Cosmos AdaLN under each Cosmos transformer in ``root``; return the count.

    Zero is a valid result: a family without a Cosmos transformer has no
    per-frame conditioning to share.
    """

    classes = _cosmos_classes()
    if classes is None:
        return 0
    transformer_cls, zero_base, plain_base = classes
    shared_zero, shared_plain = _frame_shared_classes()
    count = 0
    for transformer in [m for m in root.modules() if type(m) is transformer_cls]:
        layout = FrameLayout()
        transformer.frame_layout = layout
        transformer.register_forward_pre_hook(_record_frame_layout, with_kwargs=True)
        for module in transformer.modules():
            if type(module) is zero_base:
                module.__class__ = shared_zero
            elif type(module) is plain_base:
                module.__class__ = shared_plain
            else:
                continue
            module.frame_layout = layout
            count += 1
    return count
