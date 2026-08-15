"""Ulysses sequence parallelism for MMDiT joint-attention transformers.

One engine's ranks split the image token sequence after patch/pos embedding
and run every transformer block on their local shard — token-local work
(norms, MLPs) is correct on shards, and attention exchanges shards for heads
around scaled-dot-product attention (the Ulysses exchange). Text tokens are
small and replicated; each rank attends with its head group and the text
stream is head-gathered back so the replicated stream stays identical on
every rank.

Integration is by module-graph hooks and attention processors only — the
stable diffusers extension points — never by copying the transformer's
forward internals:

- ``pos_embed -> blocks[0]``: a forward pre-hook shards the image tokens;
- every block's ``Attention``: ``UlyssesJointAttnProcessor``;
- ``blocks[-1] -> norm_out``: a forward hook gathers the shards back.

The collectives are composed from ``all_gather`` + ``narrow``: numerically
identical on gloo (CPU tests) and nccl, one code path everywhere. The
bandwidth-optimal ``all_to_all_single`` rewrite is a P6 profiling decision,
not a correctness one (docs/sprints/SPRINT_engine_worker_vocabulary.md).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _group_info(group: Any) -> tuple[int, int]:
    return dist.get_rank(group), dist.get_world_size(group)


def _gather_dim(tensor: torch.Tensor, *, dim: int, group: Any) -> torch.Tensor:
    """All-gather shards of ``tensor`` along ``dim`` (equal shard sizes)."""

    world = dist.get_world_size(group)
    chunks = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(chunks, tensor.contiguous(), group=group)
    return torch.cat(chunks, dim=dim)


def _local_chunk(tensor: torch.Tensor, *, dim: int, group: Any) -> torch.Tensor:
    """This rank's contiguous chunk of ``tensor`` along ``dim``."""

    rank, world = _group_info(group)
    size = tensor.shape[dim]
    if size % world:
        raise ValueError(
            f"dimension {dim} of size {size} is not divisible across {world} ranks",
        )
    return tensor.narrow(dim, (size // world) * rank, size // world)


def _shards_to_heads(tensor: torch.Tensor, *, group: Any) -> torch.Tensor:
    """[B, H, s_local, hd] -> [B, H/P, S, hd]: trade sequence shard for heads."""

    full_sequence = _gather_dim(tensor, dim=2, group=group)
    return _local_chunk(full_sequence, dim=1, group=group)


def _heads_to_shards(tensor: torch.Tensor, *, group: Any) -> torch.Tensor:
    """[B, H/P, S, hd] -> [B, H, s_local, hd]: the inverse Ulysses exchange."""

    full_heads = _gather_dim(tensor, dim=1, group=group)
    return _local_chunk(full_heads, dim=2, group=group)


class UlyssesJointAttnProcessor:
    """Sequence-parallel twin of diffusers' ``JointAttnProcessor2_0``.

    Receives the image stream as this rank's sequence shard and the text
    stream replicated in full. Attention runs over the full joint sequence
    with this rank's head group; the image output returns as the shard and
    the text output is head-gathered so every rank keeps the identical
    replicated text stream.
    """

    def __init__(self, group: Any) -> None:
        self.group = group

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if attention_mask is not None:
            raise ValueError(
                "Ulysses joint attention does not support attention masks",
            )
        residual_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        def heads_view(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query = heads_view(query)
        key = heads_view(key)
        value = heads_view(value)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # The Ulysses exchange: local sequence shard with all heads becomes the
        # full sequence with this rank's head group.
        query = _shards_to_heads(query, group=self.group)
        key = _shards_to_heads(key, group=self.group)
        value = _shards_to_heads(value, group=self.group)

        text_length = 0
        if encoder_hidden_states is not None:
            text_query = heads_view(attn.add_q_proj(encoder_hidden_states))
            text_key = heads_view(attn.add_k_proj(encoder_hidden_states))
            text_value = heads_view(attn.add_v_proj(encoder_hidden_states))
            if attn.norm_added_q is not None:
                text_query = attn.norm_added_q(text_query)
            if attn.norm_added_k is not None:
                text_key = attn.norm_added_k(text_key)
            # Text is replicated on every rank: taking the local head chunk
            # aligns it with the exchanged image heads without communication.
            text_length = text_query.shape[2]
            query = torch.cat([query, _local_chunk(text_query, dim=1, group=self.group)], dim=2)
            key = torch.cat([key, _local_chunk(text_key, dim=1, group=self.group)], dim=2)
            value = torch.cat([value, _local_chunk(text_value, dim=1, group=self.group)], dim=2)

        joint = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        image = joint[:, :, : joint.shape[2] - text_length]
        image = _heads_to_shards(image, group=self.group)
        image = image.transpose(1, 2).reshape(batch_size, residual_length, inner_dim)
        image = image.to(query.dtype)
        image = attn.to_out[0](image)
        image = attn.to_out[1](image)

        if encoder_hidden_states is None:
            return image

        text = joint[:, :, joint.shape[2] - text_length :]
        text = _gather_dim(text, dim=1, group=self.group)
        text = text.transpose(1, 2).reshape(batch_size, text_length, inner_dim)
        text = text.to(query.dtype)
        if not attn.context_pre_only:
            text = attn.to_add_out(text)
        return image, text


def install_sd3_sequence_parallel(transformer: Any, group: Any) -> list[Any]:
    """Make one SD3-shaped transformer run its blocks sequence-parallel.

    Shards the image tokens entering ``transformer_blocks[0]``, swaps every
    attention processor for the Ulysses twin, and gathers the shards leaving
    ``transformer_blocks[-1]``. Returns the hook handles (tests remove them;
    production installs for the model's lifetime).
    """

    if dist.get_world_size(group) < 2:
        raise ValueError("sequence parallelism requires a rank group of >= 2")

    transformer.set_attn_processor(UlyssesJointAttnProcessor(group))
    blocks = transformer.transformer_blocks

    def shard_entry(_module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        if args:
            raise RuntimeError(
                "sequence-parallel install expects keyword block calls (the "
                "diffusers inference path); positional calls indicate the "
                "gradient-checkpointing path, which is not sequence-parallel",
            )
        kwargs["hidden_states"] = _local_chunk(kwargs["hidden_states"], dim=1, group=group)
        return args, kwargs

    def gather_exit(_module: Any, _args: tuple, _kwargs: dict, output: tuple) -> tuple:
        encoder_hidden_states, hidden_states = output
        return encoder_hidden_states, _gather_dim(hidden_states, dim=1, group=group)

    return [
        blocks[0].register_forward_pre_hook(shard_entry, with_kwargs=True),
        blocks[-1].register_forward_hook(gather_exit, with_kwargs=True),
    ]


__all__ = [
    "UlyssesJointAttnProcessor",
    "install_sd3_sequence_parallel",
]
