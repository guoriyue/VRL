"""Shared device and dtype resolution for the checkpoint-eval scripts.

The five eval entrypoints (`cosmos_predict25_kling_eval`, `sana_aesthetic_checkpoint_eval`,
`sana_checkpoint_compare`, `video_reward_suite`, `wan_robotics_checkpoint_eval`) each
carried a verbatim copy of the same ``--device`` resolver. This is the single owner.

``torch`` is imported inside the functions so importing this module (and, transitively,
``sana_aesthetic_checkpoint_eval``, which deliberately keeps torch out of its module-load
path) does not eagerly pull torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import RootConfig


def resolve_eval_device(value: str) -> torch.device:
    """Resolve a ``--device`` CLI string to a ``torch.device``.

    ``"auto"`` selects ``cuda:0`` when CUDA is available, otherwise ``cpu``. An explicit
    CUDA device that is unavailable is a hard error rather than a silent CPU fallback.
    """
    import torch

    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def resolve_eval_dtype(
    dtype_arg: str,
    root: RootConfig,
    *,
    precision: PrecisionPolicy,
    device: torch.device,
    requires_trainer: str,
) -> torch.dtype:
    """Resolve a ``--dtype`` CLI string against the run's resolved precision.

    ``"auto"`` follows the config's training dtype, except on CPU where only
    float32 is dependable. ``requires_trainer`` names the caller's error domain
    for the missing-trainer case ("Anima generation", "Cosmos checkpoint
    evaluation"); everything else is identical across the eval entrypoints.
    """
    import torch

    from vrl.models.dtypes import resolve_torch_dtype

    if dtype_arg != "auto":
        return resolve_torch_dtype(dtype_arg)
    if root.trainer is None:
        raise ValueError(f"{requires_trainer} requires an online trainer config")
    if getattr(device, "type", str(device)) == "cpu":
        return torch.float32
    return resolve_torch_dtype(precision.training.dtype)
