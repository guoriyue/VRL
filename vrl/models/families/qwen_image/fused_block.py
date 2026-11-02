"""Qwen-Image transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``QwenImageTransformerBlock`` of a
root to :class:`FusedQwenImageTransformerBlock` and its attention processor to
:class:`FusedQwenDoubleStreamAttnProcessor`. The blocks keep their modules,
parameters and state-dict names; only the elementwise chains between the
projections change (``vrl.nn.kernels.dit_elementwise``): LayerNorm and
modulation in one pass that emits the bf16 tensor every projection reads, the
q/k RMSNorm and RoPE written straight into the joint attention sequence, and
the gated residual adds. ``model.fused_kernels`` installs them in the rollout
and the replay alike, so both sides keep running one computation.

The swap is not bitwise to the stock block: it rounds once where the stock
chains round two or three times, so it moves outputs by about one bf16 unit
toward the exact result. A block the fused forward does not cover (per-token
modulation, ``zero_cond_t``) runs the stock forward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformerBlock,
)

from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    joint_rms_norm_rope,
    layer_norm_modulate,
)


class FusedQwenDoubleStreamAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    """The stock joint attention with the q/k RMSNorm and RoPE fused into the concatenation."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None or attention_mask is not None or image_rotary_emb is None:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                attention_mask,
                image_rotary_emb,
            )
        if encoder_hidden_states_mask is not None:
            image_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
            attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)[
                :, None, None, :
            ]
        seq_txt = encoder_hidden_states.shape[1]
        head_dim = attn.inner_dim // attn.heads

        def heads(projection: Any, states: torch.Tensor) -> torch.Tensor:
            return projection(states).unflatten(-1, (-1, head_dim))

        img_freqs, txt_freqs = image_rotary_emb
        # Joint order is [text, image], as the stock processor concatenates.
        joint_query = joint_rms_norm_rope(
            [heads(attn.add_q_proj, encoder_hidden_states), heads(attn.to_q, hidden_states)],
            [attn.norm_added_q.weight, attn.norm_q.weight],
            [txt_freqs, img_freqs],
            eps=attn.norm_q.eps,
        )
        joint_key = joint_rms_norm_rope(
            [heads(attn.add_k_proj, encoder_hidden_states), heads(attn.to_k, hidden_states)],
            [attn.norm_added_k.weight, attn.norm_k.weight],
            [txt_freqs, img_freqs],
            eps=attn.norm_k.eps,
        )
        joint_value = torch.cat(
            [heads(attn.add_v_proj, encoder_hidden_states), heads(attn.to_v, hidden_states)],
            dim=1,
        )
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
        img_attn_output = attn.to_out[0](joint_hidden_states[:, seq_txt:, :].contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(joint_hidden_states[:, :seq_txt, :].contiguous())
        return img_attn_output, txt_attn_output


class FusedQwenImageTransformerBlock(QwenImageTransformerBlock):
    """``QwenImageTransformerBlock`` whose norms, modulation and residuals run fused."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        modulate_index: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modulate_index is not None or self.zero_cond_t:
            return super().forward(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
                modulate_index,
            )
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)
        img_shift1, img_scale1, img_gate1 = img_mod1.chunk(3, dim=-1)
        img_shift2, img_scale2, img_gate2 = img_mod2.chunk(3, dim=-1)
        txt_shift1, txt_scale1, txt_gate1 = txt_mod1.chunk(3, dim=-1)
        txt_shift2, txt_scale2, txt_gate2 = txt_mod2.chunk(3, dim=-1)

        img_attn_output, txt_attn_output = self.attn(
            hidden_states=layer_norm_modulate(
                hidden_states, img_scale1, img_shift1, eps=self.img_norm1.eps
            ),
            encoder_hidden_states=layer_norm_modulate(
                encoder_hidden_states, txt_scale1, txt_shift1, eps=self.txt_norm1.eps
            ),
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        hidden_states = gated_residual(hidden_states, img_gate1, img_attn_output)
        encoder_hidden_states = gated_residual(encoder_hidden_states, txt_gate1, txt_attn_output)

        img_mlp_output = self.img_mlp(
            layer_norm_modulate(hidden_states, img_scale2, img_shift2, eps=self.img_norm2.eps)
        )
        hidden_states = gated_residual(hidden_states, img_gate2, img_mlp_output)
        txt_mlp_output = self.txt_mlp(
            layer_norm_modulate(
                encoder_hidden_states, txt_scale2, txt_shift2, eps=self.txt_norm2.eps
            )
        )
        encoder_hidden_states = gated_residual(encoder_hidden_states, txt_gate2, txt_mlp_output)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


def _check_fusable(block: QwenImageTransformerBlock) -> None:
    """Refuse a block whose norms the fused kernels do not compute."""

    for name in ("img_norm1", "img_norm2", "txt_norm1", "txt_norm2"):
        norm = getattr(block, name)
        if not isinstance(norm, nn.LayerNorm) or norm.weight is not None:
            raise ValueError(f"fused Qwen-Image blocks need an affine-free LayerNorm at {name}")
    attn = block.attn
    norms = [attn.norm_q, attn.norm_k, attn.norm_added_q, attn.norm_added_k]
    if not all(
        isinstance(norm, RMSNorm) and norm.weight is not None and norm.bias is None
        for norm in norms
    ):
        raise ValueError("fused Qwen-Image blocks need weighted, bias-free RMSNorm q/k norms")
    if attn.norm_q.eps != attn.norm_added_q.eps or attn.norm_k.eps != attn.norm_added_k.eps:
        raise ValueError(
            "fused Qwen-Image blocks need one q and one k RMSNorm eps per stream pair"
        )
    if type(attn.processor) is not QwenDoubleStreamAttnProcessor2_0:
        raise ValueError(
            f"fused Qwen-Image blocks replace QwenDoubleStreamAttnProcessor2_0, "
            f"found {type(attn.processor).__name__}"
        )


def install_fused_blocks(root: nn.Module, *, keep_lora_input_cast: bool) -> int:
    """Swap every Qwen-Image block under ``root`` to the fused one; return how many are fused.

    Blocks already fused count and stay as they are.

    The fused kernels emit bf16 where the stock LayerNorm emitted fp32. A
    LoRA projection that runs PEFT's own forward (the batch-invariant adapter
    path does not cast; PEFT's handles dropout, DoRA and several adapters)
    under autocast would upcast its bf16 input to its fp32 adapter weights
    only for autocast to round it straight back to bf16 at the adapter GEMM:
    an exact round trip that reads and writes the whole activation twice.
    ``keep_lora_input_cast=False`` turns that upcast off in the fused blocks;
    it is exact only when every call runs under a half-precision autocast,
    and without one PEFT's adapter GEMM fails on the dtype mismatch rather
    than computing anything else.
    """

    from peft.tuners.tuners_utils import BaseTunerLayer

    def prepare(block: QwenImageTransformerBlock) -> None:
        _check_fusable(block)
        swap_processor(block.attn, FusedQwenDoubleStreamAttnProcessor)
        if not keep_lora_input_cast:
            for layer in block.modules():
                if isinstance(layer, BaseTunerLayer):
                    layer.cast_input_dtype_enabled = False

    return swap_block_class(
        root, QwenImageTransformerBlock, FusedQwenImageTransformerBlock, prepare=prepare
    )


__all__ = [
    "FusedQwenDoubleStreamAttnProcessor",
    "FusedQwenImageTransformerBlock",
    "install_fused_blocks",
]
