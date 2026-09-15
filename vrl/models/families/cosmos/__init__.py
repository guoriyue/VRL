"""Shared pieces of the Cosmos model families."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any


class NoOpCosmosSafetyChecker:
    """Avoid loading guardrail weights in internal optimization diagnostics.

    Both predict2 and predict2.5 pipelines instantiate a CosmosSafetyChecker
    during from_pretrained; this stub satisfies the pipeline's surface without
    downloading guardrail weights (identical hand-copies previously lived in
    each family module).
    """

    def to(self, *_args: Any, **_kwargs: Any) -> NoOpCosmosSafetyChecker:
        return self

    def check_text_safety(self, _prompt: str) -> bool:
        return True

    def check_video_safety(self, video: Any) -> Any:
        return video


@contextlib.contextmanager
def no_safety_checker(pipeline_module: Any) -> Iterator[None]:
    """Swap the module's CosmosSafetyChecker for the stub during from_pretrained."""

    original = pipeline_module.CosmosSafetyChecker
    pipeline_module.CosmosSafetyChecker = NoOpCosmosSafetyChecker
    try:
        yield
    finally:
        pipeline_module.CosmosSafetyChecker = original


class CosmosReplayForward:
    """Cosmos replay uses the real ``timestep_idx`` instead of a rebuilt index 0.

    Unlike sd3/wan which pack timesteps as ``[1, B]`` and call
    ``forward_step(state, 0)``, Cosmos's ``forward_step`` indexes
    ``state.scheduler.sigmas[step_idx]`` so the eval path must pass
    through the actual ``timestep_idx`` to keep sigma scaling consistent with
    rollout. The base replay methods share this hook for stored and caller
    latents, preventing the SFT regularizer from drifting to ``sigma[0]``.
    """

    def _replay_forward_step_index(self, timestep_idx: int) -> int:
        return int(timestep_idx)


class CosmosContextParallelAttnProcessor:
    """Cosmos attention over equal, contiguous query-token shards.

    The caller supplies local RoPE positions and identical collective order on
    the explicit CP group. Optional text cross-attention uses replicated context
    and local heads; image context and query-dependent masks are unsupported.
    Parameter gradients must be summed over CP by the strategy;
    these exchanges account only for activation gradients. This processor does
    not install token splitting, precision policy, or gradient synchronization.
    """

    def __init__(self, group: Any, *, cross_attention: bool = False):
        self.group = group
        self.cross_attention = cross_attention

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        image_rotary_emb: Any = None,
    ) -> Any:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.embeddings import apply_rotary_emb

        from vrl.trainers.distributed import (
            context_parallel_heads_to_tokens,
            context_parallel_tokens_to_heads,
        )

        if not self.cross_attention and (
            encoder_hidden_states is not None or attention_mask is not None
        ):
            raise ValueError("Cosmos CP processor supports only unmasked self-attention")
        if self.cross_attention:
            if encoder_hidden_states is None or isinstance(encoder_hidden_states, tuple):
                raise ValueError("Cosmos CP cross-attention requires a replicated text tensor")
            if image_rotary_emb is not None:
                raise ValueError("Cosmos CP cross-attention does not apply RoPE")
            if attention_mask is not None and (
                attention_mask.ndim != 4 or attention_mask.shape[1:3] != (1, 1)
            ):
                raise ValueError("Cosmos CP cross-attention requires a broadcast key-only mask")
        context = encoder_hidden_states if self.cross_attention else hidden_states

        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = attn.to_k(context).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = attn.to_v(context).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query, key = attn.norm_q(query), attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(
                query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2
            )
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        key = key.repeat_interleave(query.size(3) // key.size(3), dim=3)
        value = value.repeat_interleave(query.size(3) // value.size(3), dim=3)
        query = context_parallel_tokens_to_heads(query, group=self.group)
        if self.cross_attention:
            import torch.distributed as dist

            world, rank = dist.get_world_size(self.group), dist.get_rank(self.group)
            key, value = (tensor.chunk(world, dim=1)[rank].contiguous() for tensor in (key, value))
        else:
            key, value = (
                context_parallel_tokens_to_heads(tensor, group=self.group)
                for tensor in (key, value)
            )
        output = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        output = context_parallel_heads_to_tokens(output, group=self.group)
        output = output.transpose(1, 2).flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))


__all__ = [
    "CosmosContextParallelAttnProcessor",
    "CosmosReplayForward",
    "NoOpCosmosSafetyChecker",
    "no_safety_checker",
]
