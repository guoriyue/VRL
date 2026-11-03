"""Krea 2 transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``Krea2TransformerBlock`` of a root
to :class:`FusedKrea2TransformerBlock` and its attention processor to
:class:`FusedKrea2AttnProcessor`. Modules, parameters and state-dict names
stay; only the elementwise chains between the projections change
(``vrl.nn.kernels.dit_elementwise``):

- the zero-centred fp32 RMSNorm and its modulation in one pass;
- the gated residual adds;
- the grouped-query q/k RMSNorm and the RoPE in one pass;
- the attention's sigmoid output gate ``out * sigmoid(gate)`` in one pass;
- the SwiGLU product ``silu(gate) * up``.

The text-fusion and refiner blocks keep the stock modules.
``model.fused_kernels`` installs the blocks in the rollout and the replay
alike, so both sides keep running one computation. The swap is not bitwise
to the stock block: it rounds once where the stock chains round two or three
times.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_krea2 import (
    Krea2Attention,
    Krea2AttnProcessor,
    Krea2RMSNorm,
    Krea2SwiGLU,
    Krea2TransformerBlock,
)

from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    joint_rms_norm_rope,
    rms_norm_modulate,
    sigmoid_mul,
    silu_mul,
)


def _scale(norm: Krea2RMSNorm) -> torch.Tensor:
    """The effective multiplier of a zero-centred ``Krea2RMSNorm``."""

    return norm.weight + 1.0


class FusedKrea2AttnProcessor(Krea2AttnProcessor):
    """The stock attention with the q/k RMSNorm, RoPE and output gate fused."""

    def __call__(
        self,
        attn: Krea2Attention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if image_rotary_emb is None:
            return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)
        # The repeat-interleaved [S, D] tables as [S, D/2] rotations.
        cos, sin = image_rotary_emb
        freqs = torch.complex(cos[:, 0::2].float(), sin[:, 0::2].float())
        query = joint_rms_norm_rope(
            [attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))],
            [_scale(attn.norm_q)],
            [freqs],
            eps=attn.norm_q.eps,
        )
        key = joint_rms_norm_rope(
            [attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))],
            [_scale(attn.norm_k)],
            [freqs],
            eps=attn.norm_k.eps,
        )
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)
        # As the stock processor: repeat the key/value heads rather than ask SDPA for GQA.
        groups = attn.num_heads // attn.num_kv_heads
        if groups > 1:
            key = key.repeat_interleave(groups, dim=2)
            value = value.repeat_interleave(groups, dim=2)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return attn.to_out[0](sigmoid_mul(gate, hidden_states.flatten(2, 3)))


def _table(modulation: torch.Tensor) -> torch.Tensor:
    """A ``[B, 1, D]`` modulation as the per-sample ``[B, D]`` table; ``[B, L, D]`` per token."""

    return modulation.squeeze(1)


def _residual(hidden_states: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``hidden_states + gate * y`` in the dtype the stock expression promotes to."""

    out_dtype = torch.promote_types(hidden_states.dtype, torch.promote_types(gate.dtype, y.dtype))
    return gated_residual(hidden_states, _table(gate), y, out_dtype=out_dtype)


class FusedKrea2TransformerBlock(Krea2TransformerBlock):
    """``Krea2TransformerBlock`` whose norms, modulation, residuals and SwiGLU run fused."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            rms_norm_modulate(
                hidden_states,
                _scale(self.norm1),
                _table(prescale),
                _table(preshift),
                eps=self.norm1.eps,
            ),
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = _residual(hidden_states, pregate, attn_out)
        modulated = rms_norm_modulate(
            hidden_states,
            _scale(self.norm2),
            _table(postscale),
            _table(postshift),
            eps=self.norm2.eps,
        )
        ff = self.ff
        ff_out = ff.down(silu_mul(ff.gate(modulated), ff.up(modulated)))
        return _residual(hidden_states, postgate, ff_out)


def _check_fusable(block: Krea2TransformerBlock) -> None:
    """Refuse a block whose norms, attention or feed-forward the fused kernels do not compute."""

    attn = block.attn
    for norm in (block.norm1, block.norm2, attn.norm_q, attn.norm_k):
        if type(norm) is not Krea2RMSNorm:
            raise ValueError("fused Krea 2 blocks need Krea2RMSNorm norms")
    if type(attn.processor) is not Krea2AttnProcessor:
        raise ValueError(
            f"fused Krea 2 blocks replace Krea2AttnProcessor, found {type(attn.processor).__name__}"
        )
    if type(block.ff) is not Krea2SwiGLU:
        raise ValueError("fused Krea 2 blocks need the Krea2SwiGLU feed-forward")


def _prepare(block: Krea2TransformerBlock) -> None:
    _check_fusable(block)
    swap_processor(block.attn, FusedKrea2AttnProcessor)


def install_fused_blocks(root: nn.Module) -> int:
    """Swap every Krea 2 block under ``root`` to the fused one; return how many are fused."""

    return swap_block_class(
        root, Krea2TransformerBlock, FusedKrea2TransformerBlock, prepare=_prepare
    )


__all__ = ["FusedKrea2AttnProcessor", "FusedKrea2TransformerBlock", "install_fused_blocks"]
