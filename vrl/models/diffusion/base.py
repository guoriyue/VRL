"""Shared diffusion model base for RL runtimes.

The public trainer-facing replay interface is ``vrl.models.interfaces.ReplayModel``.
This base class only factors shared diffusion model behavior: generation
primitives, replay-state projection helpers, and trainable transformer weight
loading for diffusion families.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from vrl.engine.diffusion.layout import VideoGenerationRequest
from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult


class DiffusionModelBase(nn.Module, ABC):
    """Shared model base for diffusion families on the RL path."""

    family: str = "diffusion"
    # Some upstream diffusion-RL recipes intentionally keep LoRA replay outside
    # autocast. The trainer reads this flag when choosing the replay context.
    disable_train_autocast: bool = False

    async def load(self) -> None:
        """Load heavy modules. Default no-op for adapters constructed eagerly."""
        return None

    def describe(self) -> dict[str, Any]:
        return {"family": self.family}

    @abstractmethod
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt and optional negative prompt into embedding tensors."""

    @abstractmethod
    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Build a private per-family sampling state for the denoise loop."""

    @abstractmethod
    def forward_step(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Run one transformer forward without stepping the scheduler."""

    def forward(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Run one trainable denoise transformer step."""

        return self.forward_step(state, step_idx)

    @abstractmethod
    def decode_latents(self, latents: Any) -> Any:
        """Decode latents to a frame tensor."""

    def export_batch_context(self, state: Any) -> dict[str, Any]:
        """Project private sampling state into shared trajectory context."""
        raise NotImplementedError

    def export_replay_tensors(self, state: Any) -> dict[str, Any]:
        """Project private sampling state into per-sample trajectory tensors."""
        raise NotImplementedError

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        """Rebuild private sampling state for trainer replay."""
        raise NotImplementedError

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        """Rebuild diffusion sampling state and run one replay forward."""
        del request
        from vrl.engine.trajectory import TrajectoryResolver

        state = self.restore_eval_state(
            TrajectoryResolver.from_batch(batch).replay_tensor_dict("denoise"),
            batch.context,
            batch.observations[:, timestep_idx],
            timestep_idx,
        )
        values = self.forward(state, 0)
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values=dict(values),
                ),
            },
        )

    def _require_transformer(self) -> Any:
        """Return the registered trainable transformer."""

        transformer = getattr(self, "transformer", None)
        if transformer is None:
            raise RuntimeError(
                f"{type(self).__name__} has no registered trainable transformer",
            )
        return transformer

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        """Disable LoRA/adapters, or return a no-op context when absent."""

        transformer = self._require_transformer()
        disable = getattr(transformer, "disable_adapter", None)
        if not callable(disable):
            return contextlib.nullcontext()
        return disable()

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load trainable transformer weights from module-prefixed keys."""

        transformer = self._require_transformer()
        state = dict(state_dict)
        if not state:
            raise ValueError("load_trainable_state received an empty state dict")
        prefix = "transformer."
        bad_keys = [key for key in state if not key.startswith(prefix)]
        if bad_keys:
            raise ValueError(
                "load_trainable_state only accepts trainable keys prefixed with "
                f"{prefix!r}; got {bad_keys}",
            )
        state = {
            key[len(prefix):]: value
            for key, value in state.items()
        }
        if not state:
            raise ValueError("load_trainable_state requires transformer.* keys")
        named_parameters = getattr(transformer, "named_parameters", None)
        if not callable(named_parameters):
            raise TypeError(
                f"{type(transformer).__name__} must expose named_parameters()",
            )
        trainable_keys = {
            name
            for name, parameter in named_parameters()
            if bool(getattr(parameter, "requires_grad", False))
        }
        if not trainable_keys:
            raise ValueError(f"{type(transformer).__name__} has no trainable parameters")
        extra = sorted(set(state) - trainable_keys)
        missing = sorted(trainable_keys - set(state))
        if extra or missing:
            raise ValueError(
                "load_trainable_state must receive exactly trainable "
                f"transformer keys; missing={missing[:5]}, extra={extra[:5]}",
            )
        return transformer.load_state_dict(state, strict=False)

    @classmethod
    def from_spec(cls, spec: Any) -> DiffusionModelBase:  # pragma: no cover (abstract)
        """Load the backend from a runtime spec."""
        raise NotImplementedError

    def apply_lora(self, spec: Any) -> None:  # pragma: no cover (default no-op)
        raise NotImplementedError

    def enable_full_finetune(self) -> None:  # pragma: no cover (default no-op)
        raise NotImplementedError

    def torch_compile_transformer(self, mode: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def set_num_steps(self, n: int) -> None:  # pragma: no cover
        raise NotImplementedError

    @property
    def trainable_modules(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    @property
    def scheduler(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    @property
    def backend_handle(self) -> Any:  # pragma: no cover
        raise NotImplementedError


__all__ = ["DiffusionModelBase"]
