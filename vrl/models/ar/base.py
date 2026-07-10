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
from typing import Any

import torch.nn as nn

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
        core keeps its bf16/fp32 master and is never quantized.
        """

        from vrl.nn.quantization import swap_linears_to_fp8
        from vrl.nn.quantization.fp8 import LM_EXCLUDE

        return swap_linears_to_fp8(
            self.language_model, recipe=recipe, exclude=LM_EXCLUDE,
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


__all__ = ["ARModelBase", "ARReplayRolloutStubs"]
