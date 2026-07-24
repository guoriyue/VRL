"""Qwen3-to-Cosmos text adapter modules used by Anima checkpoints."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AnimaLLMAdapter(nn.Module):
    """Qwen3-to-Cosmos text adapter shipped inside Anima transformer files."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32128, 1024)
        self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(64)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    source_dim=1024,
                    model_dim=1024,
                    num_heads=16,
                )
                for _ in range(6)
            ],
        )
        self.out_proj = nn.Linear(1024, 1024)
        self.norm = nn.RMSNorm(1024, eps=1e-6)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids).to(context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        model_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.self_attn = Attention(model_dim, model_dim, num_heads, model_dim // num_heads)
        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = Attention(model_dim, source_dim, num_heads, model_dim // num_heads)
        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_context: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        normed = self.norm_self_attn(x)
        # Self-attention keys are x itself, so they take the target-side
        # rotary table; the qwen-side table only applies to cross-attn keys
        # (the two tokenizers can disagree on sequence length).
        x = x + self.self_attn(
            normed,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings,
        )
        normed = self.norm_cross_attn(x)
        x = x + self.cross_attn(
            normed,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        return x + self.mlp(self.norm_mlp(x))


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_context: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        cos, sin = position_embeddings_context
        key_states = apply_rotary_pos_emb(key_states, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


__all__ = [
    "AnimaLLMAdapter",
    "Attention",
    "RotaryEmbedding",
    "TransformerBlock",
    "apply_rotary_pos_emb",
    "rotate_half",
]
