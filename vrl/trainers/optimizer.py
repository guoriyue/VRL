"""How the training optimizer is built and, when needed, wrapped.

Two things live here because they are one decision:

* :func:`build_optimizer` — the AdamW / bitsandbytes-8bit factory.
* :class:`FP32MasterWeightOptimizer` — the mixed-precision wrapper the trainer puts
  around that optimizer whenever a trainable parameter is fp16/bf16.

WHY the wrapper exists: an AdamW step updates the tensors it is given, in their own
dtype. With bf16 weights (``precision.training.dtype: bf16``, our default) a full-param
update of ~lr=1e-5 on a ~1e-2 weight is smaller than bf16's ULP, so every step rounds
away and the weights never move at all. Keeping FP32 master copies lets those sub-ULP
residuals accumulate while the model itself stays low-precision — so the forward keeps
its fast bf16 kernels, half-size activations, and bf16 rollout weight sync.

LoRA runs never reach the wrapper: PEFT keeps adapter weights in fp32, so the trainer's
gate sees no low-precision trainable parameter. It is the full-param path
(``model.use_lora: false``) that depends on it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import torch
from torch import nn

from vrl.trainers.core.types import TrainerConfig

OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]


def build_optimizer(parameters: Any, config: TrainerConfig) -> torch.optim.Optimizer:
    """Create the AdamW optimizer a training step runs on."""

    optim = config.optim
    parameters = list(parameters)
    # fused=True collapses the per-parameter optimizer step into a handful of
    # kernels (the loop variant launched ~1.7k/step on LoRA models). It is
    # only valid for CUDA float params; anything else falls back to default.
    use_fused = bool(parameters) and all(
        isinstance(p, torch.Tensor) and p.is_cuda and p.is_floating_point() for p in parameters
    )
    if getattr(optim, "optim_8bit", False):
        # int8 Adam state -> full-parameter 2B+ DiT fits on one 32GB card. Quantizes the
        # optimizer state only (not the forward), so rollout/replay logprobs are unchanged
        # — safe on the RL policy path. Requires bitsandbytes (CUDA, incl. Blackwell sm_120).
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            parameters,
            lr=optim.lr,
            betas=(optim.adam_beta1, optim.adam_beta2),
            weight_decay=optim.weight_decay,
            eps=optim.eps,
        )
    return torch.optim.AdamW(
        parameters,
        lr=optim.lr,
        betas=(optim.adam_beta1, optim.adam_beta2),
        weight_decay=optim.weight_decay,
        eps=optim.eps,
        fused=use_fused or None,
    )

class FP32MasterWeightOptimizer(torch.optim.Optimizer):
    """Run an optimizer on FP32 masters while the model keeps low-precision weights.

    Forward and backward continue to use the source model parameters. Before
    unscaling or clipping, callers must invoke :meth:`prepare_gradients`; this
    copies the accumulated source gradients into the FP32 parameters exposed by
    ``param_groups``. A successful :meth:`step` copies the updated masters back
    to the source parameters without discarding sub-ULP residuals in the masters.

    Existing FP32 source parameters are reused instead of cloned, so mixed-dtype
    models pay for a master only where one is necessary. Dynamic parameter-group
    insertion is deliberately unsupported because a new source/master binding
    cannot be inferred from an optimizer parameter alone.
    """

    def __init__(
        self,
        source_parameters: Iterable[nn.Parameter],
        optimizer_factory: OptimizerFactory,
    ) -> None:
        sources = tuple(source_parameters)
        self._validate_sources(sources)

        masters = tuple(self._make_master(parameter) for parameter in sources)
        optimizer = optimizer_factory(masters)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer_factory must return a torch.optim.Optimizer, "
                f"got {type(optimizer).__name__}",
            )
        self._validate_optimizer_parameters(optimizer, masters)

        # Initialize Optimizer's hook machinery, then share the wrapped
        # optimizer's live groups/state so schedulers and GradScaler see exactly
        # the parameters that its step consumes.
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self._optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.defaults = optimizer.defaults
        self.state = optimizer.state
        self._source_parameters = sources
        self._master_parameters = masters
        self._gradients_prepared = False
        self._initialized = True

    @staticmethod
    def _validate_sources(parameters: tuple[nn.Parameter, ...]) -> None:
        if not parameters:
            raise ValueError("FP32 master optimizer requires at least one source parameter")
        seen: set[int] = set()
        for parameter in parameters:
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(
                    "FP32 master optimizer requires torch.nn.Parameter sources, "
                    f"got {type(parameter).__name__}",
                )
            if parameter.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                raise TypeError(
                    "FP32 master optimizer supports float16, bfloat16, and float32 "
                    f"parameters, got {parameter.dtype}",
                )
            if id(parameter) in seen:
                raise ValueError("a source parameter may appear only once")
            seen.add(id(parameter))

    @staticmethod
    def _make_master(source: nn.Parameter) -> nn.Parameter:
        if source.dtype == torch.float32:
            return source
        return nn.Parameter(
            source.detach().to(dtype=torch.float32),
            requires_grad=source.requires_grad,
        )

    @staticmethod
    def _validate_optimizer_parameters(
        optimizer: torch.optim.Optimizer,
        expected: tuple[nn.Parameter, ...],
    ) -> None:
        actual = tuple(
            parameter for group in optimizer.param_groups for parameter in group["params"]
        )
        if len(actual) != len(expected) or any(
            got is not want for got, want in zip(actual, expected, strict=True)
        ):
            raise ValueError(
                "optimizer_factory must preserve every FP32 master parameter in order",
            )

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield the FP32 parameters used for unscaling and gradient clipping."""

        return iter(self._master_parameters)

    def prepare_gradients(self) -> None:
        """Copy accumulated source gradients to their FP32 optimizer parameters."""

        if self._gradients_prepared:
            raise RuntimeError("FP32 master gradients have already been prepared")
        # Validate every binding before mutating any gradient so an unsupported
        # sparse/cross-device parameter cannot leave a half-prepared optimizer.
        for source, master in zip(
            self._source_parameters,
            self._master_parameters,
            strict=True,
        ):
            if source is master:
                continue
            gradient = source.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise TypeError("FP32 master optimizer does not support sparse gradients")
            if gradient.device != master.device:
                raise RuntimeError(
                    "source and FP32 master parameters must be on the same device "
                    "before preparing gradients",
                )

        for source, master in zip(
            self._source_parameters,
            self._master_parameters,
            strict=True,
        ):
            if source is master:
                continue
            gradient = source.grad
            if gradient is None:
                master.grad = None
                continue
            if master.grad is None:
                master.grad = torch.empty_like(master)
            master.grad.copy_(gradient.detach())
            # From this boundary onward unscale/clip/step consume only master
            # grads. Releasing the low-precision copy saves ~3.2 GB for SANA
            # 1.6B and lets the allocator reuse those pages while later master
            # gradients are materialized.
            source.grad = None
        self._gradients_prepared = True

    @torch.no_grad()
    def _copy_masters_to_sources(self) -> None:
        for source, master in zip(
            self._source_parameters,
            self._master_parameters,
            strict=True,
        ):
            if source is not master:
                source.copy_(master)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Update FP32 masters and publish their rounded values to the model."""

        if closure is not None:
            raise RuntimeError("FP32 master optimizer does not support closures")
        if not self._gradients_prepared:
            raise RuntimeError("prepare_gradients() must be called before step()")
        result = self._optimizer.step()
        self._copy_masters_to_sources()
        self._gradients_prepared = False
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear both model-side and master-side gradient storage."""

        self._optimizer.zero_grad(set_to_none=set_to_none)
        for source, master in zip(
            self._source_parameters,
            self._master_parameters,
            strict=True,
        ):
            if source is master:
                continue
            if set_to_none:
                source.grad = None
            elif source.grad is not None:
                source.grad.zero_()
        self._gradients_prepared = False

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer moments plus master values hidden by source rounding."""

        state = self._optimizer.state_dict()
        if "fp32_master_weights" in state:
            raise RuntimeError("wrapped optimizer uses reserved fp32_master_weights state key")
        state["fp32_master_weights"] = {
            "version": 1,
            "optimizer_type": self._optimizer_type(),
            "parameters": [
                None if source is master else master.detach()
                for source, master in zip(
                    self._source_parameters,
                    self._master_parameters,
                    strict=True,
                )
            ],
        }
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore optimizer moments and exact FP32 master residuals."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("optimizer state_dict must be a mapping")
        master_state = state_dict.get("fp32_master_weights")
        saved_masters = self._validate_master_state(master_state)

        optimizer_state = dict(state_dict)
        del optimizer_state["fp32_master_weights"]
        self._optimizer.load_state_dict(optimizer_state)
        self.param_groups = self._optimizer.param_groups
        self.defaults = self._optimizer.defaults
        self.state = self._optimizer.state

        with torch.no_grad():
            for saved, source, master in zip(
                saved_masters,
                self._source_parameters,
                self._master_parameters,
                strict=True,
            ):
                if source is master:
                    continue
                assert isinstance(saved, torch.Tensor)
                master.copy_(saved.to(device=master.device))
        self._copy_masters_to_sources()
        self._gradients_prepared = False

    def _validate_master_state(self, state: Any) -> list[torch.Tensor | None]:
        if not isinstance(state, Mapping):
            raise ValueError("optimizer checkpoint is missing fp32_master_weights")
        if set(state) != {"version", "optimizer_type", "parameters"} or state["version"] != 1:
            raise ValueError("unsupported fp32_master_weights checkpoint schema")
        if state["optimizer_type"] != self._optimizer_type():
            raise ValueError(
                "FP32 master checkpoint optimizer type does not match the current "
                f"optimizer: {state['optimizer_type']!r} != {self._optimizer_type()!r}",
            )
        saved = state["parameters"]
        if not isinstance(saved, list) or len(saved) != len(self._master_parameters):
            raise ValueError("FP32 master checkpoint parameter count does not match the model")
        for index, (value, source, master) in enumerate(
            zip(
                saved,
                self._source_parameters,
                self._master_parameters,
                strict=True,
            ),
        ):
            if source is master:
                if value is not None:
                    raise ValueError(
                        f"FP32 source parameter {index} must not have a duplicate master",
                    )
                continue
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"FP32 master parameter {index} is missing from checkpoint")
            if value.dtype != torch.float32 or value.shape != master.shape:
                raise ValueError(
                    f"FP32 master parameter {index} has incompatible dtype or shape",
                )
        return saved

    def _optimizer_type(self) -> str:
        optimizer_type = type(self._optimizer)
        return f"{optimizer_type.__module__}:{optimizer_type.__qualname__}"

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Reject bindings that bypass source/master construction after init."""

        if getattr(self, "_initialized", False):
            raise RuntimeError("FP32 master optimizer does not support add_param_group()")
        super().add_param_group(param_group)


__all__ = ["FP32MasterWeightOptimizer", "OptimizerFactory", "build_optimizer"]
