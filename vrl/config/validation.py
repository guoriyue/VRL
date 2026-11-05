"""Launch gates for a training config (validation tier 3).

``require_training_config`` is the one entry every training launch runs:
``parse_config`` (tier 1 shapes + tier 2 cross-section rules, see
``vrl/config/rules.py``), the precision policy, then the launch gates. A gate
is a check that tier 2 cannot afford: it needs the resolved precision policy
or a runtime module (the compile matrix reads the build-role resolver and the
checkpointing resolver). Eval and perf tools call ``parse_config`` alone, so a
gate never taxes them. Checks that read the filesystem (reward backends)
belong to ``python -m vrl.scripts.rewards.preflight``, which runs the real
reward over the real rows before training does. A check with no runtime
dependency is a tier 2 rule instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import RootConfig, parse_config


def require_training_config(cfg: DictConfig) -> tuple[RootConfig, PrecisionPolicy]:
    """Validate config once and return the typed config plus its precision policy.

    Returns the pair rather than a wrapper struct: the precision policy is a pure
    derivation of ``root``, and it is returned only so the caller does not resolve
    it a second time (asserted by tests/config/test_builders.py).
    """

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    conflicts = compile_conflicts(root)
    if conflicts:
        joined = "\n  - ".join(conflict.message for conflict in conflicts)
        raise ValueError(f"model.torch_compile.enable=true cannot combine with:\n  - {joined}")
    return root, precision


# ---- torch.compile compatibility matrix ---------------------------------------


@dataclass(frozen=True, slots=True)
class CompileConflict:
    """One feature that cannot coexist with the requested torch.compile."""

    # The config feature that collides, stable for programmatic filtering.
    feature: str
    message: str


def compile_conflicts(root: RootConfig) -> tuple[CompileConflict, ...]:
    """Every feature this config turns on that cannot coexist with torch.compile.

    ONE home for the compile compatibility matrix. Each of these was discovered
    separately and used to be enforced somewhere different (grad-checkpointing
    in the trainer, say), so adding the next one meant first finding the
    others. They are all decided by config keys, so config load is where they
    belong: a
    combination that can never run should fail before a GPU is touched.

    Returns one entry per conflict (empty when compatible).
    """

    compile_block = root.model.torch_compile if root.model is not None else None
    if compile_block is None or not bool(compile_block.enable):
        return ()

    # Each conflict below binds one build role, so the matrix honors
    # ``model.torch_compile.scope``: trainer constraints cannot veto a
    # rollout-only compile. The (block, role) decision is owned by the typed
    # build contract.
    from vrl.models.interfaces.runtime import torch_compile_for_role

    compiles_replay = torch_compile_for_role(compile_block, "replay") is not None

    conflicts: list[CompileConflict] = []

    # The family's runtime either compiles or it does not, whichever role the
    # scope selects, so this reads the raw enable bit.
    from vrl.models.families.names import normalize_model_family
    from vrl.models.families.registry import get_model_family_entry

    family = normalize_model_family(str(root.model.family))
    if not get_model_family_entry(family).runtime_capabilities.supports_torch_compile:
        conflicts.append(
            CompileConflict(
                "family",
                f"model.family={family}: this family's runtime does not support torch.compile.",
            )
        )

    # Full and selective activation checkpointing compose with torch.compile.
    # Full: replay forward+backward measured on torch 2.11 for SD3.5 Medium
    # (with and without single-rank FSDP2), Cosmos-Predict2.5 2B and Wan2.1
    # 1.3B. Selective: SD3.5 Medium with and without FSDP2 on GPU, and tiny
    # SD3/Wan transformers under compile on CPU match
    # eager gradients once its context_fn is a module-level function
    # (tests/trainers/test_activation_checkpointing.py, and root compile under
    # FSDP2 in tests/trainers/test_fsdp_compile_distributed.py). The
    # CPU-offload variant was never measured with compile.
    checkpointing = "off" if root.actor is None else root.actor.gradient_checkpointing_mode
    if compiles_replay and checkpointing in ("full_cpu", "selective_attention"):
        conflicts.append(
            CompileConflict(
                "gradient_checkpointing",
                f"actor.gradient_checkpointing={checkpointing!r}: unmeasured with "
                "torch.compile. Use full or selective checkpointing with compile, eager + "
                f"{checkpointing}, or model.torch_compile.scope=rollout to keep the trainer "
                "eager while the rollout policy compiles.",
            )
        )

    # The fused block kernels are autograd functions over Triton launches that
    # inductor would trace into graphs of its own; the two speedups are
    # alternatives, and the fused kernels exist so rollout and replay share one
    # computation, which compile's separate inference and training graphs break.
    if root.model.fused_kernels:
        conflicts.append(
            CompileConflict(
                "fused_kernels",
                "model.fused_kernels: unsupported with torch.compile. Use one of the two.",
            )
        )

    # FSDP2 replay is not a conflict. On two Gloo ranks a tiny SD3 transformer
    # compiled with full or selective checkpointing, then sharded with
    # reshard-after-forward, trains like the same compiled module unsharded
    # (tests/trainers/test_fsdp_compile_distributed.py): FSDP2's all-gather
    # and reduce-scatter hooks run outside the compiled graphs.

    return tuple(conflicts)


__all__ = [
    "CompileConflict",
    "compile_conflicts",
    "require_training_config",
]
