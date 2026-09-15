"""Project resolved role precision onto PyTorch execution boundaries."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from vrl.config.precision import Float32Precision, RolePrecision

if TYPE_CHECKING:
    import torch


@contextlib.contextmanager
def fixed_row_linear_compute(
    model: Any, *, rows: int = 64, fp32_modules: Iterable[Any] = ()
) -> Iterator[None]:
    """Use equal GEMM row counts on full and token-sharded model executions.

    Padding is private to each Linear and is removed before returning; no
    attention tokens are added. Explicit FP32 modules (e.g. LoRA A/B linears)
    must already have FP32 parameters. Parameters and state keys are unchanged.
    Apply identically to rollout/replay and keep active through backward and
    checkpoint recomputation. Install after model copying/loading; copying an
    actively wrapped model is unsupported. Installation is not concurrent-safe.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("fixed Linear row count must be a positive integer")
    linears = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    fp32 = set(fp32_modules)
    if not fp32.issubset(set(linears)):
        raise ValueError("FP32 Linear modules must belong to the model")
    for module in linears:
        if getattr(module.forward, "_vrl_fixed_rows", False):
            raise ValueError("fixed-row Linear compute is already installed")
        if module in fp32 and any(p.dtype != torch.float32 for p in module.parameters()):
            raise ValueError("FP32 Linear compute requires FP32 parameters")

    def wrap(original, force_fp32):
        def forward(input):
            precision = (
                torch.autocast(input.device.type, enabled=False)
                if force_fp32
                else contextlib.nullcontext()
            )
            with precision:
                value = input.float() if force_fp32 else input
                if value.numel() == 0:
                    return original(value)
                flat = value.reshape(-1, value.shape[-1])
                outputs = []
                for part in flat.split(rows, dim=0):
                    count = part.shape[0]
                    if count < rows:
                        part = F.pad(part, (0, 0, 0, rows - count))
                    outputs.append(original(part)[:count])
                output = torch.cat(outputs, dim=0)
                return output.reshape(*value.shape[:-1], output.shape[-1])

        forward._vrl_fixed_rows = True
        return forward

    missing = object()
    originals = []
    try:
        for module in linears:
            originals.append((module, module.__dict__.get("forward", missing)))
            module.forward = wrap(module.forward, module in fp32)
        yield
    finally:
        for module, original in originals:
            if original is missing:
                del module.forward
            else:
                module.forward = original


def model_precision(model: Any) -> RolePrecision:
    """Return the role precision stamped on a runtime or replay model."""

    return model.precision


def model_autocast(
    model: Any,
    device: torch.device | str,
) -> AbstractContextManager[Any]:
    """Apply a model's resolved role dtype and outer-autocast policy."""

    import torch

    precision = model_precision(model)
    dtype = precision.dtype
    if not precision.outer_autocast or dtype == "fp32":
        return contextlib.nullcontext()
    if dtype not in ("fp16", "bf16"):
        raise ValueError(f"unsupported autocast dtype: {dtype!r}")
    device_type = torch.device(device).type
    if device_type == "cpu" and dtype == "fp16":
        return contextlib.nullcontext()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device_type, dtype=torch_dtype)


def apply_float32_precision(mode: Float32Precision) -> None:
    """Apply one concrete FP32 matmul mode to the current PyTorch process.

    PyTorch 2.9 introduced the string-valued ``fp32_precision`` API. The
    fallback keeps supported older releases on the legacy bool API without
    mixing both mechanisms in one process.
    """

    import torch

    if mode not in ("ieee", "tf32"):
        raise ValueError(f"unsupported float32 precision mode: {mode!r}")
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if hasattr(matmul, "fp32_precision") and hasattr(cudnn, "fp32_precision"):
        matmul.fp32_precision = mode
        cudnn.fp32_precision = mode
        return
    enabled = mode == "tf32"
    matmul.allow_tf32 = enabled
    cudnn.allow_tf32 = enabled


def float32_precision_state() -> dict[str, str]:
    """Return the effective PyTorch FP32 backend modes for diagnostics."""

    import torch

    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if hasattr(matmul, "fp32_precision") and hasattr(cudnn, "fp32_precision"):
        return {
            "matmul": str(matmul.fp32_precision),
            "cudnn": str(cudnn.fp32_precision),
        }
    return {
        "matmul": "tf32" if bool(matmul.allow_tf32) else "ieee",
        "cudnn": "tf32" if bool(cudnn.allow_tf32) else "ieee",
    }


__all__ = [
    "ModulePrecisionTrace",
    "apply_float32_precision",
    "fixed_row_linear_compute",
    "float32_precision_state",
    "model_autocast",
    "model_precision",
]


class ModulePrecisionTrace:
    """Opt-in metadata-only observation at explicitly selected module boundaries.

    Hooks affect execution/compilation overhead, so this is an acceptance tool,
    not a default training instrument. Callers label phases (including backward)
    explicitly; grad mode alone cannot identify non-reentrant recomputation.
    Tensor dtype at a module boundary is not proof of a kernel's compute dtype.
    """

    def __init__(self, modules: dict[str, Any]) -> None:
        if not modules:
            raise ValueError("precision trace requires explicitly selected modules")
        self.modules = dict(modules)
        self.events: list[dict[str, Any]] = []
        self.phase = "unspecified"
        self._handles: list[Any] = []

    def snapshot(self, phase: str) -> dict[str, Any]:
        """Record all selected parameters/buffers at a load/wrap boundary."""

        self.phase = phase
        event = {
            "kind": "snapshot",
            "phase": phase,
            "modules": {
                name: self._module_state(module, recurse=True)
                for name, module in self.modules.items()
            },
        }
        self.events.append(event)
        return event

    @staticmethod
    def _tensor_records(value: Any, path: str = "") -> list[dict[str, Any]]:
        import torch

        if isinstance(value, torch.Tensor):
            return [
                {
                    "path": path,
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "shape": list(value.shape),
                    "requires_grad": value.requires_grad,
                }
            ]
        if isinstance(value, dict):
            return [
                record
                for key, item in value.items()
                for record in ModulePrecisionTrace._tensor_records(item, f"{path}.{key}")
            ]
        if isinstance(value, (tuple, list)):
            return [
                record
                for index, item in enumerate(value)
                for record in ModulePrecisionTrace._tensor_records(item, f"{path}[{index}]")
            ]
        return []

    @classmethod
    def _module_state(cls, module: Any, *, recurse: bool) -> dict[str, Any]:
        return {
            "parameters": cls._tensor_records(dict(module.named_parameters(recurse=recurse))),
            "buffers": cls._tensor_records(dict(module.named_buffers(recurse=recurse))),
        }

    def _record_forward_enter(self, name: str, module: Any, args: Any, kwargs: Any) -> None:
        import torch

        self.events.append(
            {
                "kind": "forward_enter",
                "module": name,
                "phase": self.phase,
                "grad_enabled": torch.is_grad_enabled(),
                "inputs": self._tensor_records({"args": args, "kwargs": kwargs}),
                "state": self._module_state(module, recurse=False),
                "autocast": {
                    device: {
                        "enabled": torch.is_autocast_enabled(device),
                        "dtype": str(torch.get_autocast_dtype(device)),
                    }
                    for device in ("cpu", "cuda")
                },
            }
        )

    def _record_forward_exit(
        self, name: str, _module: Any, _args: Any, _kwargs: Any, output: Any
    ) -> None:
        self.events.append(
            {
                "kind": "forward_exit",
                "module": name,
                "phase": self.phase,
                "outputs": self._tensor_records(output),
            }
        )

    def __enter__(self) -> ModulePrecisionTrace:
        from functools import partial

        if self._handles:
            raise RuntimeError("precision trace is already attached")
        try:
            for name, module in self.modules.items():
                self._handles.append(
                    module.register_forward_pre_hook(
                        partial(self._record_forward_enter, name), with_kwargs=True
                    )
                )
                self._handles.append(
                    module.register_forward_hook(
                        partial(self._record_forward_exit, name), with_kwargs=True
                    )
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, _kind: Any, _error: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
