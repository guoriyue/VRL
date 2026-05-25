"""NextStep-1 AR model runner executed by the AR engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.ar.decode_loop import (
    ARStepBatch,
    ARStepOutput,
    ARStepResult,
    ARTokenLoopInit,
)
from vrl.math.ar.flow_matching import flow_sample_with_logprob
from vrl.models.ar.paged_attention_helpers import (
    append_attention_token,
    scatter_paged_states,
    select_paged_states,
)
from vrl.nn.layers.attention.paged import (
    ARPagedAttentionBackend,
    ARPagedAttentionConfig,
    ARPagedAttentionPrefillInput,
    ARPagedAttentionStepInput,
)
from vrl.nn.modules.ar_decoder import (
    VllmDecoderPagedAttentionBackend,
)


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
    paged_cond_states: list[Any] | None = None
    paged_uncond_states: list[Any] | None = None
    prefill_forwards: int = 0
    decode_forwards: int = 0
    decode_tokens: int = 0


class NextStep1ARModelRunner:
    """Family model runner that lets the AR engine schedule NextStep token steps."""

    def __init__(
        self,
        model: Any,
        *,
        paged_attention_backend: ARPagedAttentionBackend | None = None,
    ) -> None:
        self.model = model
        self.paged_attention_backend = paged_attention_backend

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
        num_flow_steps = num_flow_steps if num_flow_steps is not None else cfg.num_flow_steps
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

        if self.paged_attention_backend is None:
            kv_cond = self.model._init_kv(prompt_embeds, prompt_mask)
            kv_uncond = (
                self.model._init_kv(uncond_embeds, uncond_mask)
                if uncond_embeds is not None
                else None
            )
            c_cond = self.model._last_hidden(kv_cond)
            c_uncond = self.model._last_hidden(kv_uncond) if kv_uncond is not None else None

            cache_lanes = {"kv_cond": kv_cond}
            row_lanes = {"c_cond": c_cond}
            cache_lane_owners = {"kv_cond": "nextstep.cond_kv"}
            row_lane_owners = {"c_cond": "nextstep.c_cond"}
            paged_cond_states = None
            paged_uncond_states = None
            if kv_uncond is not None:
                cache_lanes["kv_uncond"] = kv_uncond
                cache_lane_owners["kv_uncond"] = "nextstep.uncond_kv"
            if c_uncond is not None:
                row_lanes["c_uncond"] = c_uncond
                row_lane_owners["c_uncond"] = "nextstep.c_uncond"
        else:
            cond_prefill = self._prefill_paged(
                prompt_embeds,
                prompt_mask,
                branch="cond",
                image_token_num=int(image_token_num),
            )
            c_cond = cond_prefill.last_hidden
            cache_lanes = {}
            row_lanes = {"c_cond": c_cond, "cond_attn": prompt_mask}
            cache_lane_owners = {}
            row_lane_owners = {
                "c_cond": "nextstep.c_cond",
                "cond_attn": "nextstep.cond_attn",
            }
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
                row_lane_owners["c_uncond"] = "nextstep.c_uncond"
                row_lane_owners["uncond_attn"] = "nextstep.uncond_attn"
                paged_uncond_states = list(uncond_prefill.sequence_states)

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
                paged_cond_states=paged_cond_states,
                paged_uncond_states=paged_uncond_states,
                prefill_forwards=1 + int(uncond_embeds is not None),
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
        token, log_prob, replay_noise, cache_updates, row_updates = self._sample_ar_step(
            state,
            batch,
            generator=generator,
        )
        return ARStepOutput(
            result=ARStepResult(
                sequence_ids=batch.sequence_ids,
                positions=batch.positions,
                token=token,
                log_prob=log_prob.float(),
                replay_extras={"saved_noise": replay_noise},
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
        dict[str, Any],
    ]:
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
        token, log_prob, replay_noise = flow_sample_with_logprob(
            self.model.image_head,
            cond=batch.row_lanes["c_cond"],
            num_flow_steps=state.num_flow_steps,
            noise_level=state.noise_level,
            cfg_uncond=batch.row_lanes.get("c_uncond"),
            cfg_scale=state.cfg_scale,
            generator=step_generator,
            initial_noise=initial_noise,
        )

        state.tokens[rows, position] = token
        state.saved_noise[rows, position] = replay_noise
        state.logprobs[rows, position] = log_prob.float()

        if self.paged_attention_backend is not None:
            cache_updates, row_updates = self._advance_paged_attention(
                state,
                batch=batch,
                token=token,
            )
            state.decode_tokens += batch_size
            return token, log_prob, replay_noise, cache_updates, row_updates

        proj = self.model._image_in_projector(token)
        kv_cond = batch.cache_lanes["kv_cond"]
        kv_cond, c_cond_next = self.model._step_llm(kv_cond, proj)
        state.decode_forwards += 1
        cache_updates = {"kv_cond": kv_cond}
        row_updates = {"c_cond": c_cond_next}

        if "kv_uncond" in batch.cache_lanes:
            proj_u = self.model._image_in_projector(token)
            kv_uncond = batch.cache_lanes["kv_uncond"]
            kv_uncond, c_uncond_next = self.model._step_llm(kv_uncond, proj_u)
            state.decode_forwards += 1
            cache_updates["kv_uncond"] = kv_uncond
            row_updates["c_uncond"] = c_uncond_next

        state.decode_tokens += batch_size
        return token, log_prob, replay_noise, cache_updates, row_updates

    def _prefill_paged(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        branch: str,
        image_token_num: int,
    ) -> Any:
        return self._require_paged_attention_backend().prefill(
            ARPagedAttentionPrefillInput(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                branch=branch,
                metadata={"family": "nextstep_1", "image_token_num": image_token_num},
            )
        )

    def _advance_paged_attention(
        self,
        state: NextStep1ARState,
        *,
        batch: ARStepBatch,
        token: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        batch_size = len(batch.row_indices)
        cond_states = select_paged_states(state.paged_cond_states, batch.row_indices)
        cond_embed = self.model._image_in_projector(token).unsqueeze(1)
        input_embeds = [cond_embed]
        sequence_states = list(cond_states)
        branch_names = ["cond"] * batch_size
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
            branch_names.extend(["uncond"] * batch_size)
            uncond_next_attn = append_attention_token(batch.row_lanes["uncond_attn"])
            row_updates["uncond_attn"] = uncond_next_attn

        output = self._require_paged_attention_backend().step(
            ARPagedAttentionStepInput(
                input_embeds=torch.cat(input_embeds, dim=0),
                attention_mask=torch.cat(
                    [cond_next_attn] + ([] if uncond_next_attn is None else [uncond_next_attn]),
                    dim=0,
                ),
                sequence_states=tuple(sequence_states),
                branch_names=tuple(branch_names),
                position=batch.position,
                row_indices=tuple(batch.row_indices * (2 if has_uncond else 1)),
                metadata={"family": "nextstep_1"},
            )
        )
        updated_states = list(output.sequence_states)
        scatter_paged_states(
            state.paged_cond_states,
            batch.row_indices,
            updated_states[:batch_size],
        )
        hidden = self._normalize_paged_last_hidden(output.last_hidden)
        row_updates["c_cond"] = hidden[:batch_size]
        if has_uncond:
            scatter_paged_states(
                state.paged_uncond_states,
                batch.row_indices,
                updated_states[batch_size:],
            )
            row_updates["c_uncond"] = hidden[batch_size:]
            state.decode_forwards += 2
        else:
            state.decode_forwards += 1
        return {}, row_updates

    def _require_paged_attention_backend(self) -> ARPagedAttentionBackend:
        if self.paged_attention_backend is None:
            raise RuntimeError("NextStep paged-attention path requires a backend")
        return self.paged_attention_backend

    @staticmethod
    def _normalize_paged_last_hidden(last_hidden: torch.Tensor) -> torch.Tensor:
        if last_hidden.ndim == 3:
            if last_hidden.shape[1] != 1:
                raise ValueError("paged attention last_hidden must be [B, H] or [B, 1, H]")
            return last_hidden[:, 0, :]
        if last_hidden.ndim != 2:
            raise ValueError("paged attention last_hidden must be [B, H] or [B, 1, H]")
        return last_hidden


def build_nextstep_vllm_paged_attention_backend(
    model: Any,
    *,
    block_size: int = 16,
    cache_dtype: str = "auto",
) -> VllmDecoderPagedAttentionBackend:
    """Construct the explicit NextStep backend that borrows vLLM paged attention."""

    config = ARPagedAttentionConfig(
        family="nextstep_1",
        model_key=str(getattr(model.config, "model_path", "nextstep_1")),
        block_size=block_size,
        dtype=str(getattr(model, "dtype", "")) or None,
        device=str(getattr(model, "device", "")) or None,
        extra={
            "cache_dtype": cache_dtype,
            "backend_label": "nextstep_vllm_paged_attention",
        },
    )
    return VllmDecoderPagedAttentionBackend(
        trunk=model._lm_trunk(),
        config=config,
    )


__all__ = [
    "NextStep1ARModelRunner",
    "NextStep1ARState",
    "build_nextstep_vllm_paged_attention_backend",
]
