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


class CosmosContextParallelSelfAttnProcessor:
    """Cosmos self-attention over equal, contiguous token shards.

    The caller supplies local RoPE positions and identical collective order on
    the explicit CP group. Cross-attention and masked attention are deliberately
    unsupported. Parameter gradients must be summed over CP by the strategy;
    these exchanges account only for activation gradients. This processor does
    not install token splitting, precision policy, or gradient synchronization.
    """

    def __init__(self, group: Any):
        self.group = group

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

        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("Cosmos CP processor supports only unmasked self-attention")

        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query, key = attn.norm_q(query), attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(
                query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2
            )
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        key = key.repeat_interleave(query.size(3) // key.size(3), dim=3)
        value = value.repeat_interleave(query.size(3) // value.size(3), dim=3)
        query, key, value = (
            context_parallel_tokens_to_heads(tensor, group=self.group)
            for tensor in (query, key, value)
        )
        output = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        output = context_parallel_heads_to_tokens(output, group=self.group)
        output = output.transpose(1, 2).flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))


__all__ = [
    "CosmosContextParallelSelfAttnProcessor",
    "CosmosReplayForward",
    "NoOpCosmosSafetyChecker",
    "no_safety_checker",
]
