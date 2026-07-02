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

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        """Disable the LoRA adapter for a reference forward, or no-op when absent."""
        return disable_adapter_on(self.language_model)


__all__ = ["ARModelBase"]
