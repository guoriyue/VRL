"""Distributed training process identity for the single-process / FSDP trainer.

This module answers "which training process am I, and on what device" without
touching Ray, creating a process group, or wrapping the model. Ray placement and
actor lifecycle stay in ``vrl/ray/``. The strategy seam (backward / clip / state
export) lives in ``vrl/trainers/strategy.py`` and consumes the context produced
here.

The readiness sprint only resolves and validates the context. Real FSDP2 process
group setup and model wrapping land in ``SPRINT_multi_gpu_training.md``; until then
``assert_strategy_executable`` fail-fasts the ``fsdp`` execution path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from vrl.utils.config import cfg_get

# torchrun / env-launcher contract. Source of truth for the keys the fsdp context
# parses; the missing-env error lists exactly these.
_TORCHRUN_ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


@dataclass(frozen=True, slots=True)
class DistributedTrainingContext:
    """Identity of the current training process — pure description.

    Creates no process group and wraps no model; it only records who this process
    is (rank / local_rank / world_size / primary) and which device it owns, so the
    trainer and rank0-only output paths can branch without reading env directly.
    """

    strategy: str
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_primary: bool
    device: torch.device


def _require_env_int(env: Mapping[str, str], key: str) -> int:
    raw = env.get(key)
    if not raw:
        missing = [k for k in _TORCHRUN_ENV_KEYS if not env.get(k)]
        raise ValueError(
            "distributed.training.strategy=fsdp requires torchrun env vars "
            f"{list(_TORCHRUN_ENV_KEYS)}; missing {missing}. Launch with "
            "`torchrun --nproc-per-node=<N>` or set them explicitly before the run "
            "(this fails here, not later at CUDA-device or Ray-launch time)."
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"distributed.training: {key}={raw!r} is not an integer") from exc


def resolve_training_context(
    cfg: Any,
    *,
    device: torch.device,
    env: Mapping[str, str] | None = None,
) -> DistributedTrainingContext:
    """Resolve the training process identity from config + torchrun env.

    ``single_process`` always returns rank0 / world1 / primary and keeps the
    resource-resolved ``device``; it ignores env entirely. ``fsdp`` parses and
    validates ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` (fail-fast on missing or
    inconsistent values) and derives a per-process ``cuda:<local_rank>`` device. It
    does NOT create a process group — execution stays gated by
    ``assert_strategy_executable`` until FSDP2 lands.
    """

    env = os.environ if env is None else env
    distributed = cfg_get(cfg, "distributed", {})
    training = cfg_get(distributed, "training", {})
    strategy = str(cfg_get(training, "strategy", "single_process"))

    if strategy == "single_process":
        return DistributedTrainingContext(
            strategy=strategy,
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_primary=True,
            device=device,
        )

    if strategy == "fsdp":
        rank = _require_env_int(env, "RANK")
        local_rank = _require_env_int(env, "LOCAL_RANK")
        world_size = _require_env_int(env, "WORLD_SIZE")
        num_nodes = int(cfg_get(training, "num_nodes", 1))
        gpus_per_node = int(cfg_get(training, "gpus_per_node", 1))
        expected = num_nodes * gpus_per_node
        if world_size != expected:
            raise ValueError(
                f"distributed.training: WORLD_SIZE={world_size} must equal "
                f"num_nodes*gpus_per_node={expected} "
                f"(num_nodes={num_nodes}, gpus_per_node={gpus_per_node})."
            )
        if not 0 <= local_rank < gpus_per_node:
            raise ValueError(
                f"distributed.training: LOCAL_RANK={local_rank} is out of range for "
                f"gpus_per_node={gpus_per_node} (expected 0..{gpus_per_node - 1})."
            )
        return DistributedTrainingContext(
            strategy=strategy,
            distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_primary=(rank == 0),
            device=torch.device(f"cuda:{local_rank}"),
        )

    # Schema (TrainingSection.strategy Literal) rejects other values before we get
    # here; this guards direct callers that bypass schema validation.
    raise ValueError(
        f"unknown distributed.training.strategy={strategy!r}; "
        "expected 'single_process' or 'fsdp'"
    )


def assert_strategy_executable(context: DistributedTrainingContext) -> None:
    """Fail-fast for strategies whose execution path is not implemented yet.

    This readiness sprint installs schema + context + strategy seam only. The real
    FSDP2 wrap-and-run lands in ``SPRINT_multi_gpu_training.md``; configuring
    ``strategy=fsdp`` should preflight cleanly (schema / resources / context) and
    then stop here with an actionable message rather than half-running.
    """

    if context.strategy == "fsdp":
        raise NotImplementedError(
            "distributed.training.strategy=fsdp is configured, but FSDP2 execution "
            "is not implemented yet. Complete SPRINT_multi_gpu_training.md after this "
            "readiness sprint."
        )
