"""NextStep-1 AR model runner executed by the AR engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.ar.token_loop import ARStepBatch, ARStepOutput, ARTokenLoopInit
from vrl.generation.ar.token_loop.state import ARStepResult
from vrl.models.ar.nextstep_1.flow_step import flow_sample_with_logprob


@dataclass(slots=True)
class NextStep1ARState:
    """Mutable NextStep state owned by one scheduled AR token loop."""

    tokens: torch.Tensor
    saved_noise: torch.Tensor
    logprobs: torch.Tensor
    cfg_scale: float
    num_flow_steps: int
    noise_level: float
    image_token_num: int
    generator: torch.Generator | None = None
    prefill_forwards: int = 0
    decode_forwards: int = 0
    decode_tokens: int = 0


class NextStep1ARModelRunner:
    """Family model runner that lets the AR engine schedule NextStep token steps."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @torch.no_grad()
    def init_ar(
        self,
        prompt_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None,
        *,
        cfg_scale: float | None = None,
        num_flow_steps: int | None = None,
        noise_level: float | None = None,
        image_token_num: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ARTokenLoopInit:
        cfg = self.model.config
        cfg_scale = cfg_scale if cfg_scale is not None else cfg.cfg_scale
        num_flow_steps = (
            num_flow_steps if num_flow_steps is not None else cfg.num_flow_steps
        )
        noise_level = noise_level if noise_level is not None else cfg.noise_level
        image_token_num = (
            image_token_num if image_token_num is not None else cfg.image_token_num
        )

        batch_size = prompt_embeds.shape[0]
        token_dim = cfg.token_dim
        device = prompt_embeds.device

        tokens = torch.zeros(
            batch_size, image_token_num, token_dim, device=device, dtype=self.model.dtype
        )
        saved_noise = torch.zeros(
            batch_size, image_token_num, token_dim, device=device, dtype=self.model.dtype
        )
        logprobs = torch.zeros(
            batch_size, image_token_num, device=device, dtype=torch.float32
        )

        kv_cond = self.model._init_kv(prompt_embeds, prompt_mask)
        kv_uncond = (
            self.model._init_kv(uncond_embeds, uncond_mask)
            if uncond_embeds is not None else None
        )
        c_cond = self.model._last_hidden(kv_cond)
        c_uncond = self.model._last_hidden(kv_uncond) if kv_uncond is not None else None

        cache_lanes = {"kv_cond": kv_cond}
        row_lanes = {"c_cond": c_cond}
        cache_lane_owners = {"kv_cond": "nextstep.cond_kv"}
        row_lane_owners = {"c_cond": "nextstep.c_cond"}
        if kv_uncond is not None:
            cache_lanes["kv_uncond"] = kv_uncond
            cache_lane_owners["kv_uncond"] = "nextstep.uncond_kv"
        if c_uncond is not None:
            row_lanes["c_uncond"] = c_uncond
            row_lane_owners["c_uncond"] = "nextstep.c_uncond"

        return ARTokenLoopInit(
            state=NextStep1ARState(
                tokens=tokens,
                saved_noise=saved_noise,
                logprobs=logprobs,
                cfg_scale=float(cfg_scale),
                num_flow_steps=int(num_flow_steps),
                noise_level=float(noise_level),
                image_token_num=int(image_token_num),
                generator=generator,
                prefill_forwards=1 + int(kv_uncond is not None),
            ),
            cache_lanes=cache_lanes,
            row_lanes=row_lanes,
            cache_lane_owners=cache_lane_owners,
            row_lane_owners=row_lane_owners,
        )

    @torch.no_grad()
    def step_ar(
        self,
        state: NextStep1ARState,
        batch: ARStepBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ARStepOutput:
        step, cache_updates, row_updates = self._sample_ar_step(
            state,
            batch,
            generator=generator,
        )
        return ARStepOutput(
            result=ARStepResult(
                sequence_ids=batch.sequence_ids,
                positions=batch.positions,
                token=step.token,
                log_prob=step.log_prob.float(),
                replay_extras={"saved_noise": step.initial_noise},
                debug_counters={
                    "ar_kv_cache_enabled": True,
                    "ar_prefill_forwards": state.prefill_forwards,
                    "ar_decode_forwards": state.decode_forwards,
                    "ar_decode_tokens": state.decode_tokens,
                },
            ),
            updated_cache_lanes=cache_updates,
            updated_row_lanes=row_updates,
        )

    @torch.no_grad()
    def finalize_ar(
        self,
        state: NextStep1ARState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return state.tokens, state.saved_noise, state.logprobs

    def _sample_ar_step(
        self,
        state: NextStep1ARState,
        batch: ARStepBatch,
        generator: torch.Generator | None = None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        row_indices = batch.row_indices
        if not row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 or row >= state.tokens.shape[0] for row in row_indices):
            raise ValueError(f"invalid NextStep row indices: {row_indices}")
        if len(set(batch.positions)) != 1:
            raise ValueError("NextStep rows in one AR step must share a position")
        position = batch.position
        if position >= state.image_token_num:
            raise ValueError("NextStep1ARState has already finished sampling")

        step_generator = generator if generator is not None else state.generator
        batch_size = len(row_indices)
        token_dim = state.tokens.shape[-1]
        device = state.tokens.device
        rows = torch.tensor(row_indices, device=device, dtype=torch.long)
        initial_noise = torch.randn(
            batch_size,
            token_dim,
            device=device,
            dtype=self.model.dtype,
            generator=step_generator,
        )
        step = flow_sample_with_logprob(
            self.model.image_head,
            cond=batch.row_lanes["c_cond"],
            num_flow_steps=state.num_flow_steps,
            noise_level=state.noise_level,
            cfg_uncond=batch.row_lanes.get("c_uncond"),
            cfg_scale=state.cfg_scale,
            generator=step_generator,
            initial_noise=initial_noise,
        )

        state.tokens[rows, position] = step.token
        state.saved_noise[rows, position] = step.initial_noise
        state.logprobs[rows, position] = step.log_prob.float()

        proj = self.model._image_in_projector(step.token)
        kv_cond = batch.cache_lanes["kv_cond"]
        kv_cond, c_cond_next = self.model._step_llm(kv_cond, proj)
        state.decode_forwards += 1
        cache_updates: dict[str, Any] = {"kv_cond": kv_cond}
        row_updates: dict[str, Any] = {"c_cond": c_cond_next}

        if "kv_uncond" in batch.cache_lanes:
            proj_u = self.model._image_in_projector(step.token)
            kv_uncond = batch.cache_lanes["kv_uncond"]
            kv_uncond, c_uncond_next = self.model._step_llm(kv_uncond, proj_u)
            state.decode_forwards += 1
            cache_updates["kv_uncond"] = kv_uncond
            row_updates["c_uncond"] = c_uncond_next

        state.decode_tokens += batch_size
        return step, cache_updates, row_updates

__all__ = [
    "NextStep1ARModelRunner",
    "NextStep1ARState",
]
