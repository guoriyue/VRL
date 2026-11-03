"""MiniMax-H3 transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``MiniMaxH3TransformerBlock`` of a
root to :class:`FusedMiniMaxH3TransformerBlock` and, where the block still has
the stock attention, its processor to :class:`FusedMiniMaxH3AttnProcessor`.
Modules, parameters and state-dict names stay; only the elementwise chains
between the projections change (``vrl.nn.kernels.dit_elementwise``):

- RMSNorm and the per-row AdaLN modulation in one pass, the table row of
  every packed row read through ``adaln_indices`` inside the kernel instead
  of three ``index_select`` copies of the sequence;
- the gated residual adds, gated per row the same way;
- the per-head q/k RMSNorm and the rotate-half RoPE of the leading rotary
  channels in one pass (the stock path rounds ``cos``/``sin`` to bf16 and
  rotates in bf16);
- the SwiGLU product ``up * silu(gate)`` straight off the packed projection.

``model.fused_kernels`` installs them in the rollout and the replay alike, so
both sides keep running one computation. The swap is not bitwise to the stock
block: it rounds once where the stock chains round two to four times.

VDN-H3 replaces every block's attention with its hybrid module; those blocks
fuse their norms, modulation, residuals and feed-forward and keep the hybrid
attention as it is.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.activations import SwiGLU
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    MiniMaxH3TransformerBlock,
)

from vrl.nn.kernels.batch_invariant import per_sample_attention
from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    rms_norm_modulate,
    rms_norm_rotate_half,
    swiglu,
)


class FusedMiniMaxH3AttnProcessor(MiniMaxH3AttnProcessor):
    """The stock packed self-attention with the q/k RMSNorm and RoPE fused."""

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn.fused_projections or rotary_emb is None:
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        cos, sin = rotary_emb
        query = rms_norm_rotate_half(
            attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)),
            attn.norm_q.weight,
            cos,
            sin,
            eps=attn.norm_q.eps,
        )
        key = rms_norm_rotate_half(
            attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)),
            attn.norm_k.weight,
            cos,
            sin,
            eps=attn.norm_k.eps,
        )
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        hidden_states = per_sample_attention(
            dispatch_attention_fn,
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class FusedMiniMaxH3TransformerBlock(MiniMaxH3TransformerBlock):
    """``MiniMaxH3TransformerBlock`` whose norms, modulation, residuals and SwiGLU run fused."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        attn_output = self.attn(
            rms_norm_modulate(
                hidden_states,
                self.norm1.weight,
                scale_msa,
                shift_msa,
                eps=self.norm1.eps,
                rows=adaln_indices,
            ),
            rotary_emb,
            attention_mask,
        )
        hidden_states = gated_residual(hidden_states, gate_msa, attn_output, rows=adaln_indices)

        swiglu_module, _, proj_out = self.ff.net
        ff_output = proj_out(
            swiglu(
                swiglu_module.proj(
                    rms_norm_modulate(
                        hidden_states,
                        self.norm2.weight,
                        scale_mlp,
                        shift_mlp,
                        eps=self.norm2.eps,
                        rows=adaln_indices,
                    )
                )
            )
        )
        return gated_residual(hidden_states, gate_mlp, ff_output, rows=adaln_indices)


def _is_weighted_rms_norm(norm: Any) -> bool:
    return isinstance(norm, nn.RMSNorm) and norm.weight is not None and norm.eps is not None


def _check_fusable(block: MiniMaxH3TransformerBlock) -> None:
    """Refuse a block whose norms or feed-forward the fused kernels do not compute."""

    if not (_is_weighted_rms_norm(block.norm1) and _is_weighted_rms_norm(block.norm2)):
        raise ValueError(
            "fused MiniMax-H3 blocks need weighted RMSNorms with an eps at norm1/norm2"
        )
    net = block.ff.net
    if (
        len(net) != 3
        or not isinstance(net[0], SwiGLU)
        or not isinstance(net[1], nn.Dropout)
        or net[1].p != 0.0
    ):
        raise ValueError("fused MiniMax-H3 blocks need the SwiGLU feed-forward without dropout")


def _fuses_attention(attn: Any) -> bool:
    """Whether ``attn`` is the stock attention the fused processor replaces."""

    return (
        isinstance(attn, MiniMaxH3Attention)
        and type(attn.processor) is MiniMaxH3AttnProcessor
        and _is_weighted_rms_norm(attn.norm_q)
        and _is_weighted_rms_norm(attn.norm_k)
    )


def _prepare(block: MiniMaxH3TransformerBlock) -> None:
    _check_fusable(block)
    if _fuses_attention(block.attn):
        swap_processor(block.attn, FusedMiniMaxH3AttnProcessor)


def install_fused_blocks(root: nn.Module) -> int:
    """Swap every MiniMax-H3 block under ``root`` to the fused one; return how many are fused."""

    return swap_block_class(
        root, MiniMaxH3TransformerBlock, FusedMiniMaxH3TransformerBlock, prepare=_prepare
    )


__all__ = [
    "FusedMiniMaxH3AttnProcessor",
    "FusedMiniMaxH3TransformerBlock",
    "install_fused_blocks",
]
