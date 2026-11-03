"""Stable Diffusion 3.5 (MMDiT) transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``JointTransformerBlock`` of a root
to :class:`FusedJointTransformerBlock` and, where its attentions carry q/k
RMSNorms (3.5's ``qk_norm="rms_norm"``), their processors to
:class:`FusedJointAttnProcessor`. Modules, parameters and state-dict names
stay; only the elementwise chains between the projections change
(``vrl.nn.kernels.dit_elementwise``):

- every AdaLN LayerNorm and its modulation in one pass (both modulations of
  3.5 Medium's dual-attention blocks, and the last block's continuous
  context norm), where the stock block normalizes in fp32 and scales and
  shifts through fp32 temporaries;
- the gated residual adds;
- the q/k RMSNorm of both streams in one pass, written straight into the
  joint ``[image, text]`` sequence.

The AdaLN projections, the GELU MLPs and the output projections keep the
stock modules. ``model.fused_kernels`` installs the blocks in the rollout and
the replay alike, so both sides keep running one computation. The swap is
not bitwise to the stock block: it rounds once where the stock chains round
two or three times.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import JointTransformerBlock, _chunked_feed_forward
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    RMSNorm,
    SD35AdaLayerNormZeroX,
)

from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    joint_rms_norm_rope,
    layer_norm_modulate,
)


def _residual(hidden_states: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``hidden_states + gate * y`` in the dtype the stock expression promotes to."""

    out_dtype = torch.promote_types(hidden_states.dtype, torch.promote_types(gate.dtype, y.dtype))
    return gated_residual(hidden_states, gate, y, out_dtype=out_dtype)


class FusedJointAttnProcessor(JointAttnProcessor2_0):
    """The stock joint attention with the q/k RMSNorms fused into the concatenation."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # The stock processor ignores the mask and any extra arguments too.
        del attention_mask, args, kwargs

        def heads(projection: Any, states: torch.Tensor) -> torch.Tensor:
            return projection(states).unflatten(-1, (attn.heads, -1))

        # Joint order is [image, text], as the stock processor concatenates.
        streams = [hidden_states]
        projections = [(attn.to_q, attn.to_k, attn.to_v, attn.norm_q, attn.norm_k)]
        if encoder_hidden_states is not None:
            streams.append(encoder_hidden_states)
            projections.append(
                (
                    attn.add_q_proj,
                    attn.add_k_proj,
                    attn.add_v_proj,
                    attn.norm_added_q,
                    attn.norm_added_k,
                )
            )
        query = joint_rms_norm_rope(
            [heads(p[0], s) for p, s in zip(projections, streams, strict=True)],
            [p[3].weight for p in projections],
            [None] * len(streams),
            eps=attn.norm_q.eps,
        )
        key = joint_rms_norm_rope(
            [heads(p[1], s) for p, s in zip(projections, streams, strict=True)],
            [p[4].weight for p in projections],
            [None] * len(streams),
            eps=attn.norm_k.eps,
        )
        value = torch.cat([heads(p[2], s) for p, s in zip(projections, streams, strict=True)], 1)
        joint = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        joint = joint.transpose(1, 2).flatten(2, 3).to(query.dtype)

        image = hidden_states.shape[1]
        hidden_states = attn.to_out[1](attn.to_out[0](joint[:, :image]))
        if encoder_hidden_states is None:
            return hidden_states
        encoder_hidden_states = joint[:, image:]
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class FusedJointTransformerBlock(JointTransformerBlock):
    """``JointTransformerBlock`` whose norms, modulation and residuals run fused."""

    def _feed_forward(self, ff: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._chunk_size is not None:
            return _chunked_feed_forward(ff, hidden_states, self._chunk_dim, self._chunk_size)
        return ff(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        emb = self.norm1.linear(self.norm1.silu(temb))
        norm_eps = self.norm1.norm.eps
        if self.use_dual_attention:
            (
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                shift_msa2, scale_msa2, gate_msa2,
            ) = emb.chunk(9, dim=1)  # fmt: skip
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)

        context_norm = self.norm1_context
        if self.context_pre_only:
            context_emb = context_norm.linear(
                context_norm.silu(temb).to(encoder_hidden_states.dtype)
            )
            c_scale_msa, c_shift_msa = context_emb.chunk(2, dim=1)
        else:
            context_emb = context_norm.linear(context_norm.silu(temb))
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
                context_emb.chunk(6, dim=1)
            )

        attn_output, context_attn_output = self.attn(
            hidden_states=layer_norm_modulate(hidden_states, scale_msa, shift_msa, eps=norm_eps),
            encoder_hidden_states=layer_norm_modulate(
                encoder_hidden_states, c_scale_msa, c_shift_msa, eps=context_norm.norm.eps
            ),
            **joint_attention_kwargs,
        )
        if self.use_dual_attention:
            norm_hidden_states2 = layer_norm_modulate(
                hidden_states, scale_msa2, shift_msa2, eps=norm_eps
            )
        hidden_states = _residual(hidden_states, gate_msa, attn_output)
        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
            hidden_states = _residual(hidden_states, gate_msa2, attn_output2)

        ff_output = self._feed_forward(
            self.ff, layer_norm_modulate(hidden_states, scale_mlp, shift_mlp, eps=self.norm2.eps)
        )
        hidden_states = _residual(hidden_states, gate_mlp, ff_output)

        if self.context_pre_only:
            return None, hidden_states
        encoder_hidden_states = _residual(encoder_hidden_states, c_gate_msa, context_attn_output)
        context_ff_output = self._feed_forward(
            self.ff_context,
            layer_norm_modulate(
                encoder_hidden_states, c_scale_mlp, c_shift_mlp, eps=self.norm2_context.eps
            ),
        )
        encoder_hidden_states = _residual(encoder_hidden_states, c_gate_mlp, context_ff_output)
        return encoder_hidden_states, hidden_states


def _check_layer_norm(norm: Any, where: str) -> None:
    if type(norm) is not nn.LayerNorm or norm.weight is not None:
        raise ValueError(f"fused SD3 blocks need an affine-free LayerNorm at {where}")


def _fuses_attention(attn: Attention) -> bool:
    """Whether ``attn`` has the weighted q/k RMSNorms :class:`FusedJointAttnProcessor` fuses."""

    norms = [attn.norm_q, attn.norm_k]
    if getattr(attn, "add_q_proj", None) is not None:
        norms += [attn.norm_added_q, attn.norm_added_k]
    return (
        type(attn.processor) is JointAttnProcessor2_0
        and all(isinstance(norm, RMSNorm) and norm.weight is not None for norm in norms)
        and len({norm.eps for norm in norms[::2]}) == 1
        and len({norm.eps for norm in norms[1::2]}) == 1
    )


def _prepare(block: JointTransformerBlock) -> None:
    norm1_type = SD35AdaLayerNormZeroX if block.use_dual_attention else AdaLayerNormZero
    if type(block.norm1) is not norm1_type or getattr(block.norm1, "emb", None) is not None:
        raise ValueError(f"fused SD3 blocks need an embedding-free {norm1_type.__name__}")
    context_type = AdaLayerNormContinuous if block.context_pre_only else AdaLayerNormZero
    if type(block.norm1_context) is not context_type or (
        getattr(block.norm1_context, "emb", None) is not None
    ):
        raise ValueError(f"fused SD3 blocks need an embedding-free {context_type.__name__}")
    _check_layer_norm(block.norm1.norm, "norm1.norm")
    _check_layer_norm(block.norm1_context.norm, "norm1_context.norm")
    _check_layer_norm(block.norm2, "norm2")
    if not block.context_pre_only:
        _check_layer_norm(block.norm2_context, "norm2_context")
    for attn in (block.attn, block.attn2):
        if attn is not None and _fuses_attention(attn):
            swap_processor(attn, FusedJointAttnProcessor)


def install_fused_blocks(root: nn.Module) -> int:
    """Swap every SD3 block under ``root`` to the fused one; return how many are fused."""

    return swap_block_class(
        root, JointTransformerBlock, FusedJointTransformerBlock, prepare=_prepare
    )


__all__ = ["FusedJointAttnProcessor", "FusedJointTransformerBlock", "install_fused_blocks"]
