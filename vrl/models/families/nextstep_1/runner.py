"""NextStep-1 AR model runner executed by the AR engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.steps.token import (
    TokenLoopInit,
    TokenStepBatch,
    TokenStepOutput,
)
from vrl.math.token.flow_matching import flow_sample_with_logprob
from vrl.models.steps.token.paged_attention_helpers import (
    append_attention_token,
    normalize_paged_last_hidden,
    scatter_paged_states,
    select_paged_states,
)
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionPrefillInput,
    ARAttentionStepInput,
)


@dataclass(slots=True)
class NextStep1ARState:
    """Mutable NextStep state owned by one scheduled AR token loop."""

    tokens: torch.Tensor
    saved_noise: torch.Tensor
    logprobs: torch.Tensor
    guidance_scale: float
    num_steps: int
    noise_level: float
    paged_cond_states: list[Any]
    generator: torch.Generator | None = None
    paged_uncond_states: list[Any] | None = None

    @property
    def image_token_num(self) -> int:
        """Return the generated sequence length owned by ``tokens``."""

        return int(self.tokens.shape[1])


class NextStep1ARModelRunner:
    """Family model runner that lets the AR engine schedule NextStep token steps."""

    def __init__(
        self,
        model: Any,
        *,
        attention_backend: ARAttentionBackend,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend

    @torch.no_grad()
    def init_token(
        self,
        prompt_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None,
        *,
        guidance_scale: float | None = None,
        num_steps: int | None = None,
        noise_level: float | None = None,
        image_token_num: int | None = None,
        generator: torch.Generator | None = None,
    ) -> TokenLoopInit:
        cfg = self.model.config
        guidance_scale = guidance_scale if guidance_scale is not None else cfg.guidance_scale
        num_steps = num_steps if num_steps is not None else cfg.num_steps
        noise_level = noise_level if noise_level is not None else cfg.noise_level
        image_token_num = image_token_num if image_token_num is not None else cfg.image_token_num

        batch_size = prompt_embeds.shape[0]
        token_dim = cfg.token_dim
        device = prompt_embeds.device

        tokens = torch.zeros(
            batch_size, image_token_num, token_dim, device=device, dtype=self.model.dtype
        )
        saved_noise = torch.zeros(
            batch_size, image_token_num, token_dim, device=device, dtype=self.model.dtype
        )
        logprobs = torch.zeros(batch_size, image_token_num, device=device, dtype=torch.float32)

        cond_prefill = self._prefill_paged(
            prompt_embeds,
            prompt_mask,
            branch="cond",
            image_token_num=int(image_token_num),
        )
        c_cond = cond_prefill.last_hidden
        row_lanes = {"c_cond": c_cond, "cond_attn": prompt_mask}
        paged_cond_states = list(cond_prefill.sequence_states)
        paged_uncond_states = None
        if uncond_embeds is not None and uncond_mask is not None:
            uncond_prefill = self._prefill_paged(
                uncond_embeds,
                uncond_mask,
                branch="uncond",
                image_token_num=int(image_token_num),
            )
            row_lanes["c_uncond"] = uncond_prefill.last_hidden
            row_lanes["uncond_attn"] = uncond_mask
            paged_uncond_states = list(uncond_prefill.sequence_states)

        return TokenLoopInit(
            state=NextStep1ARState(
                tokens=tokens,
                saved_noise=saved_noise,
                logprobs=logprobs,
                guidance_scale=float(guidance_scale),
                num_steps=int(num_steps),
                noise_level=float(noise_level),
                paged_cond_states=paged_cond_states,
                generator=generator,
                paged_uncond_states=paged_uncond_states,
            ),
            row_count=batch_size,
            step_count=int(image_token_num),
            row_lanes=row_lanes,
        )

    @torch.no_grad()
    def step_token(
        self,
        state: NextStep1ARState,
        batch: TokenStepBatch,
    ) -> TokenStepOutput:
        row_updates = self._sample_ar_step(state, batch)
        return TokenStepOutput(updated_row_lanes=row_updates)

    @torch.no_grad()
    def finalize_token(
        self,
        state: NextStep1ARState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return state.tokens, state.saved_noise, state.logprobs

    def _sample_ar_step(
        self,
        state: NextStep1ARState,
        batch: TokenStepBatch,
    ) -> dict[str, Any]:
        row_indices = batch.row_indices
        if any(row >= state.tokens.shape[0] for row in row_indices):
            raise ValueError(f"invalid NextStep row indices: {row_indices}")
        position = batch.position
        if position >= state.image_token_num:
            raise ValueError("NextStep1ARState has already finished sampling")

        batch_size = len(row_indices)
        token_dim = state.tokens.shape[-1]
        device = state.tokens.device
        rows = torch.tensor(row_indices, device=device, dtype=torch.long)
        initial_noise = torch.randn(
            batch_size,
            token_dim,
            device=device,
            dtype=self.model.dtype,
            generator=state.generator,
        )
        token, log_prob, replay_noise = flow_sample_with_logprob(
            self.model.image_head,
            cond=batch.row_lanes["c_cond"],
            num_steps=state.num_steps,
            noise_level=state.noise_level,
            cfg_uncond=batch.row_lanes.get("c_uncond"),
            guidance_scale=state.guidance_scale,
            generator=state.generator,
            initial_noise=initial_noise,
        )

        state.tokens[rows, position] = token
        state.saved_noise[rows, position] = replay_noise
        state.logprobs[rows, position] = log_prob.float()

        # The advanced hidden state only conditions the next image token.
        # Advancing after the final token mutates KV state that finalize_token
        # never reads and pays for one unnecessary transformer forward.
        if position + 1 < state.image_token_num:
            return self._advance_paged_attention(
                state,
                batch=batch,
                token=token,
            )
        return {}

    def _prefill_paged(
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
                metadata={"image_token_num": image_token_num},
            )
        )

    def _advance_paged_attention(
        self,
        state: NextStep1ARState,
        *,
        batch: TokenStepBatch,
        token: torch.Tensor,
    ) -> dict[str, Any]:
        batch_size = len(batch.row_indices)
        cond_states = select_paged_states(state.paged_cond_states, batch.row_indices)
        cond_embed = self.model._image_in_projector(token).unsqueeze(1)
        input_embeds = [cond_embed]
        sequence_states = list(cond_states)
        cond_next_attn = append_attention_token(batch.row_lanes["cond_attn"])
        row_updates: dict[str, Any] = {"cond_attn": cond_next_attn}
        uncond_next_attn: torch.Tensor | None = None

        has_uncond = state.paged_uncond_states is not None and "uncond_attn" in batch.row_lanes
        if has_uncond:
            uncond_states = select_paged_states(
                state.paged_uncond_states,
                batch.row_indices,
            )
            uncond_embed = self.model._image_in_projector(token).unsqueeze(1)
            input_embeds.append(uncond_embed)
            sequence_states.extend(uncond_states)
            uncond_next_attn = append_attention_token(batch.row_lanes["uncond_attn"])
            row_updates["uncond_attn"] = uncond_next_attn

        output = self.attention_backend.step(
            ARAttentionStepInput(
                input_embeds=torch.cat(input_embeds, dim=0),
                attention_mask=torch.cat(
                    [cond_next_attn] + ([] if uncond_next_attn is None else [uncond_next_attn]),
                    dim=0,
                ),
                sequence_states=tuple(sequence_states),
            )
        )
        updated_states = list(output.sequence_states)
        scatter_paged_states(
            state.paged_cond_states,
            batch.row_indices,
            updated_states[:batch_size],
        )
        hidden = normalize_paged_last_hidden(output.last_hidden)
        row_updates["c_cond"] = hidden[:batch_size]
        if has_uncond:
            scatter_paged_states(
                state.paged_uncond_states,
                batch.row_indices,
                updated_states[batch_size:],
            )
            row_updates["c_uncond"] = hidden[batch_size:]
        return row_updates


__all__ = [
    "NextStep1ARModelRunner",
    "NextStep1ARState",
]
