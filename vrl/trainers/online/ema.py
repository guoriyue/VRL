"""Exponential Moving Average for model parameters.

Ported from flow_grpo/ema.py.  Maintains a shadow copy of trainable
parameters and supports eval-time weight swap via copy_ema_to/copy_temp_to.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class EMAModuleWrapper:
    """EMA wrapper for any set of torch parameters.

    Typical usage::

        ema = EMAModuleWrapper(model.parameters(), decay=0.9, update_step_interval=8)
        # after each optimizer step:
        ema.step(model.parameters(), global_step)
        # for evaluation:
        ema.copy_ema_to(model.parameters(), store_temp=True)
        evaluate(model)
        ema.copy_temp_to(model.parameters())
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ) -> None:
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        self.temp_stored_parameters: list[torch.Tensor] | None = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device
        self.num_updates = 0

    @property
    def has_updates(self) -> bool:
        return self.num_updates > 0

    def get_current_decay(self, optimization_step: int) -> float:
        """Warmup decay: ramps from ~0.1 to ``self.decay``."""
        return min(
            (1 + optimization_step) / (10 + optimization_step),
            self.decay,
        )

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step: int) -> None:
        """Update EMA parameters."""
        parameters = list(parameters)
        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            # Same-device pairs update via one batched _foreach lerp (the
            # per-parameter loop launched ~1.7k kernels/step on LoRA models);
            # ema.lerp_(param, w) computes ema + w*(param - ema), the exact
            # formula of the previous loop. Cross-device pairs keep the
            # explicit staging copy.
            fused_ema: list[torch.Tensor] = []
            fused_param: list[torch.Tensor] = []
            for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
                if not param.requires_grad:
                    continue
                if ema_param.device == param.device:
                    fused_ema.append(ema_param)
                    fused_param.append(param.detach())
                else:
                    param_copy = param.detach().to(ema_param.device)
                    param_copy.sub_(ema_param)
                    param_copy.mul_(one_minus_decay)
                    ema_param.add_(param_copy)
                    del param_copy
            if fused_ema:
                # DTensor shadows (FSDP2) go through the same fused call: the
                # shadow is a clone of its param (identical sharding), and torch
                # dispatches _foreach_lerp_ on DTensor natively (probed on the
                # pinned torch; a torch downgrade that loses coverage would
                # fail loudly here, not silently).
                torch._foreach_lerp_(fused_ema, fused_param, one_minus_decay)
            self.num_updates += 1

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """Move EMA parameters to a device/dtype."""
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(
        self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True
    ) -> None:
        """Replace model parameters with EMA values; optionally save originals."""
        parameters = list(parameters)
        if store_temp:
            # copy=True is required: plain .cpu() is a no-op when params already
            # live on CPU, so detach() would share storage and the in-place copy_
            # below would clobber the saved originals — copy_temp_to would then
            # restore EMA values instead of the pre-swap weights. copy=True keeps
            # the CPU-offload intent while guaranteeing an independent buffer.
            # (DTensor params stage as CPU-local DTensors — same code path.)
            self.temp_stored_parameters = [p.detach().to("cpu", copy=True) for p in parameters]

        for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
            param.data.copy_(ema_param.to(param.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore original parameters from temporary storage."""
        assert self.temp_stored_parameters is not None
        for temp_param, param in zip(self.temp_stored_parameters, parameters, strict=True):
            param.data.copy_(temp_param.data)
        self.temp_stored_parameters = None

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint-facing state; always plain full tensors.

        DTensor shadows (FSDP2) gather to full CPU tensors so the rank0-written
        checkpoint carries every rank's data — the gather is a COLLECTIVE, so
        this must run on every rank (save_training_checkpoint calls
        trainer.state_dict() on all ranks before its primary-only file write).
        """

        from torch.distributed.tensor import DTensor

        return {
            "decay": self.decay,
            "ema_parameters": [
                p.full_tensor().detach().cpu() if isinstance(p, DTensor) else p
                for p in self.ema_parameters
            ],
            "num_updates": self.num_updates,
        }

    def checkpoint_state_dict(self, *, is_primary: bool) -> dict[str, Any]:
        """Join FSDP gathers everywhere while retaining shadows only on rank0."""

        from torch.distributed.tensor import DTensor

        gathered: list[torch.Tensor] = []
        for parameter in self.ema_parameters:
            full = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            if is_primary:
                gathered.append(full.detach().cpu().clone())
        if not is_primary:
            return {}
        return {
            "decay": self.decay,
            "ema_parameters": gathered,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        from torch.distributed.tensor import DTensor, distribute_tensor

        self.decay = state_dict.get("decay", self.decay)
        incoming = state_dict.get("ema_parameters", self.ema_parameters)
        resharded: list[torch.Tensor] = []
        # Checkpoints store full tensors; when the live shadows are DTensor
        # (FSDP2), re-shard each one onto the live shadow's layout (SPMD:
        # every resuming rank read the full checkpoint and takes its shard).
        for current, loaded in zip(self.ema_parameters, incoming, strict=True):
            if isinstance(current, DTensor) and not isinstance(loaded, DTensor):
                loaded = distribute_tensor(
                    loaded.to(device=current.device, dtype=current.dtype),
                    current.device_mesh,
                    current.placements,
                )
            resharded.append(loaded)
        self.ema_parameters = resharded
        self.num_updates = int(state_dict.get("num_updates", self.num_updates))
        self.to(self.device)
