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

import torch
import torch.nn as nn

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult
from vrl.models.utils import disable_adapter_on, load_weights_into
from vrl.trajectory.device import move_value_to_device


class DiffusionModelBase(nn.Module, ABC):
    """Shared model base for diffusion families on the RL path."""

    family: str = "diffusion"
    # Some upstream diffusion-RL recipes intentionally keep LoRA replay outside
    # autocast. The trainer reads this flag when choosing the replay context.
    disable_train_autocast: bool = False

    async def load(self) -> None:
        """Load heavy modules. Default no-op for adapters constructed eagerly."""
        return None

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
        replay_tensors, batch_context, latents = self._replay_inputs_for_step(
            batch,
            timestep_idx,
        )
        state = self.restore_eval_state(
            replay_tensors,
            batch_context,
            latents,
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

    def _replay_inputs_for_step(
        self,
        batch: Any,
        timestep_idx: int,
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        """Resolve only the current denoise step's replay tensors on model device."""

        from vrl.trajectory import TrajectoryResolver

        try:
            device = self.device
        except Exception:
            device = None
        replay_tensors = TrajectoryResolver.from_batch(batch).replay_tensor_dict(
            "denoise",
            axis="timestep",
            axis_index=timestep_idx,
            device=device,
        )
        latents = move_value_to_device(batch.observations[:, timestep_idx], device)
        return replay_tensors, dict(batch.context), latents

    def _require_transformer(self) -> Any:
        """Return the registered trainable transformer."""

        transformer = getattr(self, "transformer", None)
        if transformer is None:
            raise RuntimeError(
                f"{type(self).__name__} has no registered trainable transformer",
            )
        return transformer

    def _transformer_dtype(self) -> torch.dtype:
        """Return the dtype of the current trainable transformer."""

        transformer = self._require_transformer()
        dtype = getattr(transformer, "dtype", None)
        if dtype is not None:
            return dtype
        try:
            return next(transformer.parameters()).dtype
        except StopIteration as exc:
            raise RuntimeError(
                f"{type(self).__name__} transformer has no parameters to infer dtype",
            ) from exc

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        """Disable LoRA/adapters, or return a no-op context when absent."""

        return disable_adapter_on(self._require_transformer())

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load trainable transformer weights from ``transformer.*`` sync keys."""

        transformer = self._require_transformer()
        return load_weights_into(
            transformer,
            state_dict,
            prefix="transformer",
            label=type(transformer).__name__,
        )

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
