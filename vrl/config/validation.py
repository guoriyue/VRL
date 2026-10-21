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
    validate_guarded_rollout_drift(root, precision)
    return root, precision


# ---- torch.compile compatibility matrix ---------------------------------------


@dataclass(frozen=True, slots=True)
class CompileConflict:
    """One feature that cannot coexist with the requested torch.compile."""

    # The config feature that collides, stable for programmatic filtering
    # (the runtime checkpointing pass re-checks only its own entry).
    feature: str
    message: str


def compile_conflicts(root: RootConfig) -> tuple[CompileConflict, ...]:
    """Every feature this config turns on that cannot coexist with torch.compile.

    ONE home for the compile compatibility matrix. Each of these was discovered
    separately and used to be enforced somewhere different — grad-checkpointing
    in the trainer, sequence parallelism nowhere at all — so adding the next
    one meant first finding the others. They are
    all decided by config keys, so config load is where they belong: a
    combination that can never run should fail before a GPU is touched.

    Returns one entry per conflict (empty when compatible).
    """

    compile_block = root.model.torch_compile if root.model is not None else None
    if compile_block is None or not bool(compile_block.enable):
        return ()

    # Each conflict below binds one build role, so the matrix honors
    # ``model.torch_compile.scope``: trainer constraints cannot veto a
    # rollout-only compile, and rollout constraints cannot veto a replay-only
    # one. The (block, role) decision is owned by the typed build contract.
    from vrl.models.interfaces.runtime import torch_compile_for_role

    compiles_replay = torch_compile_for_role(compile_block, "replay") is not None
    compiles_rollout = torch_compile_for_role(compile_block, "rollout") is not None

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

    # Full activation checkpointing composes with torch.compile: measured with
    # replay forward+backward on torch 2.11 for SD3.5 Medium (regional and root
    # compile, with and without single-rank FSDP2), Cosmos-Predict2.5 2B
    # (regional and root) and Wan2.1 1.3B (regional). Selective checkpointing
    # does not: dynamo cannot trace its context_fn ("checkpoint not implemented
    # for NestedUserFunctionVariable context_fn", Wan2.1 1.3B regional), and
    # the CPU-offload variant was never measured with compile.
    checkpointing = "off" if root.actor is None else root.actor.gradient_checkpointing_mode
    if compiles_replay and checkpointing in ("selective", "full_cpu"):
        conflicts.append(
            CompileConflict(
                "gradient_checkpointing",
                f"actor.gradient_checkpointing={checkpointing!r}: torch.compile cannot "
                "trace selective checkpointing's context_fn (InternalTorchDynamoError on "
                "Wan2.1), and full_cpu is unmeasured with compile. Use full "
                "checkpointing with compile, eager + this mode, or "
                "model.torch_compile.scope=rollout to keep the trainer eager while the "
                "rollout policy compiles.",
            )
        )

    # FSDP2 replay is not a conflict. On two Gloo ranks a tiny SD3 transformer
    # compiled regionally or at the root, with full checkpointing, then
    # sharded with reshard-after-forward, trains like the same compiled module
    # unsharded (tests/trainers/test_fsdp_compile_distributed.py): FSDP2's
    # all-gather and reduce-scatter hooks run outside the compiled graphs.
    distributed = root.distributed

    # Sequence parallelism is installed by the rollout WORKER, after the family
    # builder has already compiled the policy core: the installer swaps every
    # attention processor and registers forward hooks on the first/last block,
    # mutating a module inductor has already traced. sd3_5 declares BOTH
    # supports_torch_compile and a sequence_parallel_installer, so this is
    # reachable from config -- and it had no gate at all before.
    resources = None if distributed is None else distributed.resources
    gpus_per_engine = 1 if resources is None else int(resources.rollout.gpus_per_engine)
    if compiles_rollout and gpus_per_engine > 1:
        conflicts.append(
            CompileConflict(
                "sequence_parallel",
                f"distributed.resources.rollout.gpus_per_engine={gpus_per_engine}: "
                "sequence parallelism installs attention processors and forward hooks "
                "on the policy core AFTER the model is built and compiled, mutating "
                "the module torch.compile already traced.",
            )
        )

    return tuple(conflicts)


# ---- rollout drift ------------------------------------------------------------


def validate_guarded_rollout_drift(root: RootConfig, precision: PrecisionPolicy) -> None:
    """Refuse a rollout approximation that no drift correction will cover.

    Quantization needs no check here: it changes the rollout precision label, so
    ``stages_match`` goes False and the trainer already installs TIS/RS
    correction, under which the replay-parity gate still fails on catastrophic
    drift.

    A request-scoped approximation is the uncovered case. TeaCache reuses a
    cached ``noise_pred`` on skipped denoise steps, so the collection-time
    log-prob stops matching the trainer's exact replay forward -- while BOTH
    roles keep the same precision label, leaving every automatic correction off.
    Silently, the run would train on uncorrected off-policy gradients and still
    report convergence.
    """

    from vrl.nn.optimization import unguarded_drift_sources

    sampling = root.sampling
    sources = unguarded_drift_sources(
        sampling.model_dump(mode="python", exclude_none=True) if sampling is not None else None,
        precision,
    )
    if not sources:
        return
    # The same escape hatch the precision-split path honors: an explicit expert
    # block means the user has chosen the correction policy deliberately.
    explicit = set() if root.trainer is None else root.trainer.model_fields_set
    if "precision_correction" in explicit:
        return
    raise ValueError(
        f"sampling enables {', '.join(sources)}, which makes the rollout log-probs "
        "diverge from the trainer's exact replay forward, but rollout and training "
        "precision are identical so no importance-sampling correction is armed. "
        "Set an explicit trainer.precision_correction for this run, or disable "
        "the optimization.",
    )


__all__ = [
    "CompileConflict",
    "compile_conflicts",
    "require_training_config",
    "validate_guarded_rollout_drift",
]
