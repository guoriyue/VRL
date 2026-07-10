"""Shared paged-attention helpers and CFG token-loop base for AR runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.ar.decode_loop import (
    ARStepBatch,
    ARStepOutput,
    ARStepResult,
)
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionPrefillInput,
    ARAttentionStepInput,
)

__all__ = [
    "PagedCFGARState",
    "PagedCFGTokenRunner",
    "append_attention_token",
    "normalize_paged_last_hidden",
    "scatter_paged_states",
    "select_paged_states",
]


def append_attention_token(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            attention_mask,
            torch.ones(
                attention_mask.shape[0],
                1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ],
        dim=1,
    )


def normalize_paged_last_hidden(last_hidden: torch.Tensor) -> torch.Tensor:
    """Squeeze a paged-attention ``[B, 1, H]`` last-hidden to ``[B, H]``.

    Accepts an already-``[B, H]`` tensor unchanged; rejects any other rank.
    """

    if last_hidden.ndim == 3:
        if last_hidden.shape[1] != 1:
            raise ValueError("paged attention last_hidden must be [B, H] or [B, 1, H]")
        return last_hidden[:, 0, :]
    if last_hidden.ndim != 2:
        raise ValueError("paged attention last_hidden must be [B, H] or [B, 1, H]")
    return last_hidden


def select_paged_states(
    states: list[Any] | None,
    row_indices: list[int],
) -> list[Any]:
    if states is None:
        raise RuntimeError("paged attention state is not initialized")
    return [states[index] for index in row_indices]


def scatter_paged_states(
    states: list[Any] | None,
    row_indices: list[int],
    values: list[Any],
) -> None:
    if states is None:
        raise RuntimeError("paged attention state is not initialized")
    if len(row_indices) != len(values):
        raise ValueError("paged attention state updates must match row indices")
    for row_index, value in zip(row_indices, values, strict=True):
        states[row_index] = value


@dataclass(slots=True, kw_only=True)
class PagedCFGARState:
    """Shared mutable state for a cond/uncond paged-CFG AR token loop.

    ``kw_only`` so family subclasses (Emu3's structural-mask schedule) can add
    required fields after these defaulted ones.
    """

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    guidance_scale: float
    temperature: float
    total_token_num: int
    paged_cond_states: list[Any] | None = None
    paged_uncond_states: list[Any] | None = None
    prefill_forwards: int = 0
    decode_forwards: int = 0
    decode_tokens: int = 0


class PagedCFGTokenRunner:
    """Shared cond/uncond paged-attention token loop for CFG AR families.

    Emu3 and Janus-Pro run the exact same loop: prefill both CFG branches,
    per-step gather/scatter of paged KV states, attention-mask growth, and the
    step/finalize bookkeeping. A family supplies only:

    - ``family``: registry family key (backend metadata + error strings);
    - ``init_ar``: family-shaped entry that builds its state and lanes;
    - ``_sample_cfg_image_token(state, hidden, position)``: CFG combine +
      sampling + the RL log-prob scoring contract;
    - ``_embed_sampled_token(sampled)``: token -> embedding for the next step.
    """

    family: str = ""

    def __init__(
        self,
        model: Any,
        *,
        attention_backend: ARAttentionBackend,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend

    @torch.no_grad()
    def step_ar(
        self,
        state: PagedCFGARState,
        batch: ARStepBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ARStepOutput:
        del generator
        cache_updates, row_updates = self._sample_ar_step(state, batch)
        return ARStepOutput(
            result=ARStepResult(
                debug_counters={
                    "ar_kv_cache_enabled": True,
                    "ar_paged_attention_enabled": state.paged_cond_states is not None,
                    "ar_prefill_forwards": state.prefill_forwards,
                    "ar_decode_forwards": state.decode_forwards,
                    "ar_decode_tokens": state.decode_tokens,
                },
            ),
            updated_cache_lanes=cache_updates,
            updated_row_lanes=row_updates,
        )

    @torch.no_grad()
    def finalize_ar(self, state: PagedCFGARState) -> tuple[torch.Tensor, torch.Tensor]:
        return state.token_ids, state.logprobs

    # -- family hooks ----------------------------------------------------

    def _sample_cfg_image_token(
        self,
        state: PagedCFGARState,
        hidden: torch.Tensor,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CFG combine + sampling + RL log-prob scoring (family-owned)."""

        raise NotImplementedError

    def _embed_sampled_token(self, sampled: torch.Tensor) -> torch.Tensor:
        """Embed one sampled image token for the next decode step (family-owned)."""

        raise NotImplementedError

    # -- shared loop internals --------------------------------------------

    def _prefill_ar_prompt_paged(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        branch: str,
        image_token_num: int,
    ) -> Any:
        return self.attention_backend.prefill(
            ARAttentionPrefillInput(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                branch=branch,
                metadata={"family": self.family, "image_token_num": image_token_num},
            )
        )

    def _sample_ar_step(
        self,
        state: PagedCFGARState,
        batch: ARStepBatch,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_ar_step_batch(state, batch)
        return self._sample_ar_step_kv(state, batch)

    def _validate_ar_step_batch(
        self,
        state: PagedCFGARState,
        batch: ARStepBatch,
    ) -> None:
        row_indices = batch.row_indices
        if not row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 or row >= state.token_ids.shape[0] for row in row_indices):
            raise ValueError(f"invalid {self.family} row indices: {row_indices}")
        if len(set(batch.positions)) != 1:
            raise ValueError("ActiveSequence positions must match within one AR step")
        if batch.position >= state.total_token_num:
            raise ValueError(f"{type(state).__name__} has already finished sampling")

    def _sample_ar_step_kv(
        self,
        state: PagedCFGARState,
        batch: ARStepBatch,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        row_indices = batch.row_indices
        position = batch.position
        batch_size = len(row_indices)
        rows = torch.tensor(row_indices, dtype=torch.long, device=state.token_ids.device)
        cond_hidden = batch.row_lanes["cond_last_hidden"]
        uncond_hidden = batch.row_lanes["uncond_last_hidden"]
        hidden = torch.cat([cond_hidden, uncond_hidden], dim=0).unsqueeze(1)
        sampled, lp = self._sample_cfg_image_token(state, hidden, position)
        state.token_ids[rows, position] = sampled
        state.logprobs[rows, position] = lp

        cache_updates: dict[str, Any] = {}
        row_updates: dict[str, Any] = {}
        if position + 1 < state.total_token_num:
            cache_updates, row_updates = self._advance_after_sample(
                state,
                batch=batch,
                sampled=sampled,
            )

        state.decode_tokens += batch_size
        return cache_updates, row_updates

    def _advance_after_sample(
        self,
        state: PagedCFGARState,
        *,
        batch: ARStepBatch,
        sampled: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        batch_size = len(batch.row_indices)
        cond_states = select_paged_states(state.paged_cond_states, batch.row_indices)
        uncond_states = select_paged_states(
            state.paged_uncond_states,
            batch.row_indices,
        )
        token_embed = self._embed_sampled_token(sampled)
        inputs_embeds = torch.cat([token_embed, token_embed], dim=0)

        cond_next_attn = append_attention_token(batch.row_lanes["cond_attn"])
        uncond_next_attn = append_attention_token(batch.row_lanes["uncond_attn"])
        output = self.attention_backend.step(
            ARAttentionStepInput(
                input_embeds=inputs_embeds,
                attention_mask=torch.cat([cond_next_attn, uncond_next_attn], dim=0),
                sequence_states=tuple(cond_states + uncond_states),
                branch_names=tuple(["cond"] * batch_size + ["uncond"] * batch_size),
                position=batch.position,
                row_indices=tuple(batch.row_indices + batch.row_indices),
                metadata={"family": self.family},
            )
        )

        updated_states = list(output.sequence_states)
        if len(updated_states) != 2 * batch_size:
            raise ValueError(
                "paged attention step returned "
                f"{len(updated_states)} states for batch={batch_size}",
            )
        scatter_paged_states(
            state.paged_cond_states,
            batch.row_indices,
            updated_states[:batch_size],
        )
        scatter_paged_states(
            state.paged_uncond_states,
            batch.row_indices,
            updated_states[batch_size:],
        )
        hidden = normalize_paged_last_hidden(output.last_hidden)
        state.decode_forwards += 1
        return (
            {},
            {
                "cond_last_hidden": hidden[:batch_size],
                "uncond_last_hidden": hidden[batch_size:],
                "cond_attn": cond_next_attn,
                "uncond_attn": uncond_next_attn,
            },
        )
