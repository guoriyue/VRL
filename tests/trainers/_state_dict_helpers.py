"""Test-only oracle: gather a sharded FSDP2 module's full CPU state dict.

Moved out of ``vrl.trainers.fsdp`` because production weight-sync/checkpoint no
longer materialize a full state: rollout sync routes through
``gather_trainable_state_dict`` (requires-grad keys) and checkpoint export
through ``gather_checkpoint_state_dict`` (checkpoint-owned keys). The
full-materialization gather survives only here, as the known-correct reference
that those production paths are pinned against (legacy full-checkpoint compat,
frozen-param invariance under LoRA load, and load round-trips).
"""

from __future__ import annotations

from typing import Any

from torch import nn


def gather_full_state_dict(module: nn.Module) -> dict[str, Any]:
    """Gather a sharded module's params into a full CPU state dict ON EVERY RANK.

    FSDP2 holds each parameter as a DTensor shard; checkpoint and rollout sync
    both need ordinary full tensors in the unwrapped, policy-facing key space.
    ``get_model_state_dict(full_state_dict=True)`` all-gathers + materializes each
    param to a full tensor, and (verified on a single CPU rank) returns plain
    ``torch.Tensor`` leaves with no ``module.`` / ``_orig_mod.`` / FSDP shard-key
    leakage.

    ``cpu_offload`` MUST stay False: with ``cpu_offload=True`` DCP returns the full
    state ONLY on rank0 and an EMPTY dict on every other rank (a rank0-only
    checkpoint optimization). That is wrong for the symmetric colocated model, where
    EACH rank pushes its gathered weights to its own colocated rollout — non-rank0
    would get nothing and ``select_trainable_state`` then reports every LoRA param
    "missing" (reproduced on a real 2x1 NCCL run; the world_size=1 CPU test never
    hit it because the gather is a no-op there). We move the full tensors to CPU
    ourselves below, so each rank still ends with plain CPU tensors. The first
    version is rank-replicated full state, not sharded checkpoints
    (``SPRINT_multi_gpu_training.md`` §8).
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )
    from torch.distributed.tensor import DTensor

    state = get_model_state_dict(
        module,
        options=StateDictOptions(full_state_dict=True, cpu_offload=False),
    )
    # full_state_dict=True already all-gathers each param to a full tensor;
    # defensively materialize any DTensor leaf (no-op gather in a real single-mesh
    # run) so the contract — plain full CPU tensors — holds regardless of how many
    # meshes the process has built.
    return {
        key: (value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu()
        for key, value in state.items()
    }
