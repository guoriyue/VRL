"""Shared base for autoregressive model wrappers on the RL path.

AR families (janus_pro, nextstep_1, ...) each wrap a ``self.language_model``
trunk and sync trainable weights under the ``model.`` prefix. Two behaviors were
byte-identical across families — trainable-state load and adapter disable — so
they live here once and a new AR family inherits them instead of copying. This
is the AR analog of :class:`vrl.models.diffusion.base.DiffusionModelBase`, minus
the diffusion-only pieces (versioned slots, denoise ``forward_step``): AR wrappers
keep their own family-specific forward / replay / LoRA-attach math.

Unlike ``DiffusionModelBase`` this base is intentionally NOT an ABC — AR families
expose different rollout/replay surfaces, so there is no single abstract contract
to enforce here beyond ``nn.Module``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vrl.generation.ar.decode_loop import (
    ARStepBatch,
    ARStepOutput,
    ARStepResult,
)
from vrl.models.utils import disable_adapter_on, load_weights_into


class ARModelBase(nn.Module):
    """Shared model base for autoregressive families on the RL path."""

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load only the trainable AR parameters from a rollout sync state.

        The payload carries ``model.*`` keys for this wrapper's ``requires_grad``
        parameters. ``label`` is the concrete class name so a malformed-payload
        error names the actual model (base or replay subclass).
        """
        return load_weights_into(
            self,
            state_dict,
            prefix="model",
            label=type(self).__name__,
        )

    def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
        """Swap the language trunk's big GEMMs to fp8 in place (rollout only).

        Quantizes attention/MLP linears under ``self.language_model``; the
        vocabulary heads (lm_head / gen_head / llamagen's ``output``) and
        embeddings stay high precision — the per-token log-probs the RL loss
        consumes are computed from them. VQ decoders / vision towers live
        outside ``language_model`` and are never touched. The trainer's replay
        core keeps its configured base-precision parameters and is never quantized.
        """

        from vrl.nn.quantization import LM_EXCLUDE, swap_linears_to_fp8

        return swap_linears_to_fp8(
            self.language_model,
            recipe=recipe,
            exclude=LM_EXCLUDE,
        )

    def quantize_rollout_nvfp4(self) -> list[str]:
        """Swap the language trunk's eligible MLP GEMMs to NVFP4.

        Attention projections and vocabulary heads remain in the rollout base
        dtype. The head exclusion preserves the logits scored by the RL
        objective, while MLP-only targeting is the validated NVFP4 rollout
        profile shared with diffusion models.
        """

        from vrl.nn.quantization import LM_EXCLUDE, swap_linears_to_nvfp4

        return swap_linears_to_nvfp4(
            self.language_model,
            exclude=LM_EXCLUDE,
        )

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        """Disable the LoRA adapter for a reference forward, or no-op when absent."""
        return disable_adapter_on(self.language_model)


class ARReplayRolloutStubs:
    """Rollout-only surface stubs shared by AR replay models.

    The AR analog of ``vrl.models.diffusion.base.ReplayRolloutStubs``: replay
    models load only the modules needed to recompute log-probs, so the
    rollout-side decode surface is unreachable by construction and raises with
    the concrete class name instead of each family re-writing the stub.
    """

    def decode_image_tokens(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{type(self).__name__} cannot decode image tokens")


@dataclass(slots=True, kw_only=True)
class ARDiscreteTokenState:
    """Shared scheduler-visible state for discrete-token AR loops."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    total_token_num: int
    prefill_forwards: int = 0
    decode_forwards: int = 0
    decode_tokens: int = 0


class ARDiscreteTokenRunner:
    """Shared step/finalize bookkeeping for discrete-token family runners.

    Families still own prefill, sampling, and cache advancement. This base owns
    only the engine contract that was identical across paged-CFG, GLM-Image,
    and LlamaGen: validate the scheduled rows, run one family step, report the
    common counters, and return ``(token_ids, logprobs)`` at finalization.
    """

    family: str = ""
    validation_family: str = ""

    @torch.no_grad()
    def step_ar(
        self,
        state: ARDiscreteTokenState,
        batch: ARStepBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ARStepOutput:
        del generator
        self._validate_ar_step_batch(state, batch)
        cache_updates, row_updates = self._sample_ar_step(state, batch)
        return ARStepOutput(
            result=ARStepResult(
                debug_counters={
                    "ar_kv_cache_enabled": True,
                    "ar_paged_attention_enabled": self._paged_attention_enabled(state),
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
        state: ARDiscreteTokenState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return state.token_ids, state.logprobs

    def _validate_ar_step_batch(
        self,
        state: ARDiscreteTokenState,
        batch: ARStepBatch,
    ) -> None:
        row_indices = batch.row_indices
        if not row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 or row >= state.token_ids.shape[0] for row in row_indices):
            label = self.validation_family or self.family
            raise ValueError(f"invalid {label} row indices: {row_indices}")
        if len(set(batch.positions)) != 1:
            raise ValueError("ActiveSequence positions must match within one AR step")
        if batch.position >= state.total_token_num:
            raise ValueError(f"{type(state).__name__} has already finished sampling")
        self._validate_family_step(state, batch)

    def _validate_family_step(
        self,
        state: ARDiscreteTokenState,
        batch: ARStepBatch,
    ) -> None:
        del state, batch

    def _paged_attention_enabled(self, state: ARDiscreteTokenState) -> bool:
        del state
        return False

    def _sample_ar_step(
        self,
        state: ARDiscreteTokenState,
        batch: ARStepBatch,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError


__all__ = [
    "ARDiscreteTokenRunner",
    "ARDiscreteTokenState",
    "ARModelBase",
    "ARReplayRolloutStubs",
]
