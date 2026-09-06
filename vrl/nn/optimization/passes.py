"""Rollout optimization passes and the one place that applies them in order.

A pass is a build-time transformation of the rollout policy: it reads the
resolved ``ModelBuild`` and mutates or replaces the model's declared
``policy_cores``. Families declare WHAT to optimize (``policy_cores``,
``quantization_exclude``); passes own HOW; :func:`apply_rollout_optimizations`
owns the ORDER. None of the three knows the other's internals, so a new scheme
needs no family change and a new family needs no pass change.

Ordering is a correctness constraint, not a style choice, and it lives here
rather than in each family builder:

* LoRA attaches BEFORE any pass — PEFT wraps plain ``nn.Linear`` and cannot see
  through a quantized replacement. LoRA is training configuration, not an
  optimization, so it stays with the builder and this module takes an
  already-adapted model.
* Quantization runs BEFORE compile so inductor traces the final module tree.
* Device moves and Accelerate offload hooks run AFTER every pass, for the same
  reason: hooks must see the final tree.

Not every rollout optimization is a pass. TeaCache is resolved per sampling
request in the denoise binding and skips forwards inside the loop without
touching the module tree, so it has no build-time seam to install; it declares
its replay drift through :data:`REQUEST_SCOPED_DRIFT_SOURCES` instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from vrl.utils.logging import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class PassResult:
    """What one pass did to the rollout policy."""

    name: str
    applied: bool
    detail: str


class OptimizationPass(Protocol):
    """A build-time transformation of the rollout policy.

    A structural contract: concrete passes satisfy it by shape and must not
    inherit it. ``apply`` is called only when ``enabled`` returned True.
    """

    name: str
    # True when the pass REPLACES a policy core with a wrapper (compile) rather
    # than mutating its tree in place (quantization). A replaced root hides the
    # real module from ``requires_grad_`` / ``.to()``, which is why the device
    # move seam sits immediately before the first replacing pass.
    replaces_modules: bool

    def enabled(self, build: Any) -> bool:
        """Whether this build requests the pass."""
        ...

    def apply(self, model: Any, build: Any) -> PassResult:
        """Transform ``model``'s declared policy cores in place."""
        ...


def validate_every_core_quantized(model: Any, scheme: str) -> None:
    """Every declared policy core must carry ``scheme``, not just the first.

    Checks the MODULES, not the pass's own account of what it did. A pass that
    reports which roots it touched is certifying itself: the Wan half-quantized
    bug (expert 2 silently left at the base dtype while expert 1 ran quantized,
    splitting one trajectory across two precisions) is exactly the case a
    self-report cannot catch, because the buggy pass believes it covered
    everything.
    """

    from vrl.nn.quantization import QuantizedLinear

    missed = []
    for name, core in model.policy_cores.items():
        core = getattr(core, "_orig_mod", core)  # unwrap torch.compile
        modules = getattr(core, "modules", None)
        if not callable(modules) or not any(
            isinstance(module, QuantizedLinear) and module.quantization_scheme == scheme
            for module in modules()
        ):
            missed.append(name)
    if missed:
        raise RuntimeError(
            f"rollout quantization {scheme!r} left policy core(s) "
            f"{', '.join(missed)} unquantized; every core the policy samples "
            "through must carry the scheme or one trajectory spans two "
            "precisions.",
        )


@dataclass(frozen=True, slots=True)
class QuantizationPass:
    """Swap large policy GEMMs to the configured low-precision scheme."""

    name: str = "quantization"
    # Swaps Linears inside the existing tree; the root object is unchanged.
    replaces_modules: bool = False

    def enabled(self, build: Any) -> bool:
        return getattr(getattr(build, "precision", None), "quantization", None) is not None

    def apply(self, model: Any, build: Any) -> PassResult:
        from vrl.models.loader import apply_rollout_quantization

        count = apply_rollout_quantization(model, build)
        quantization = build.precision.quantization
        if count:
            validate_every_core_quantized(model, quantization.format)
        return PassResult(
            name=self.name,
            applied=bool(count),
            detail=f"{quantization.format}: {count} linears",
        )


@dataclass(frozen=True, slots=True)
class CompilePass:
    """torch.compile every policy core.

    Compilation is a graph-level rewrite of the SAME math, so unlike
    quantization and TeaCache it is not a drift source: it does not need the
    drift guard armed.
    """

    name: str = "compile"
    # Wraps each root in an OptimizedModule, hiding the real module.
    replaces_modules: bool = True

    def enabled(self, build: Any) -> bool:
        return bool((getattr(build, "torch_compile", None) or {}).get("enable"))

    def apply(self, model: Any, build: Any) -> PassResult:
        mode = build.torch_compile["mode"]
        model.torch_compile_transformer(mode)
        # Verify the EFFECT on the real modules. A family that overrides
        # torch_compile_transformer and walks the wrong collection would
        # otherwise leave one expert of a multi-expert policy uncompiled --
        # the half-covering failure this layer exists to prevent. Asking the
        # model what changed would be self-certification; a compiled root is
        # observable, so observe it.
        missed = [
            name for name, core in model.policy_cores.items() if not hasattr(core, "_orig_mod")
        ]
        if missed:
            raise RuntimeError(
                f"{type(model).__name__}.torch_compile_transformer left "
                f"{', '.join(missed)} uncompiled; every rollout policy core must "
                "be compiled or the trajectory spans two execution paths.",
            )
        return PassResult(
            name=self.name,
            applied=True,
            detail=f"mode={mode}",
        )


@dataclass(frozen=True, slots=True)
class OffloadPass:
    """Install Accelerate CPU-offload hooks over the final module tree.

    Runs LAST for the same reason compile runs after quantization: the hooks
    must wrap whatever the earlier passes left behind. Wan's sequential path
    keeps a 16.4B transformer on CPU through LoRA, quantization and compile, so
    hooks installed any earlier would leave those wrappers outside the hook
    graph.

    Two-sided commit: the family must both accept the requested mode and report
    a healthy install, because a partially-installed hook graph must never be
    published as a reusable rollout model.
    """

    name: str = "offload"
    # Registers hooks on the existing tree; the root object is unchanged.
    replaces_modules: bool = False

    def enabled(self, build: Any) -> bool:
        """Always true — ``apply`` decides, because it can see the model.

        Deliberately NOT gated on ``mode != "none"``. A family's installer also
        validates that the mode has not CHANGED since the model was loaded (Wan
        raises "offload mode changed during rollout construction"), so skipping
        it for ``none`` would drop that check. Whether the family implements
        offload at all is a property of the MODEL, which ``enabled`` is not
        given, so that branch lives in ``apply``.
        """

        del build
        return True

    def apply(self, model: Any, build: Any) -> PassResult:
        mode = getattr(getattr(build, "rollout", None), "pipeline_offload_mode", "none")
        requested = mode != "none"
        install = getattr(model, "apply_generation_offload", None)
        if not callable(install):
            if requested:
                raise RuntimeError(
                    f"{type(model).__name__} does not implement the requested pipeline "
                    f"CPU-offload mode {mode!r}",
                )
            return PassResult(
                name=self.name,
                applied=False,
                detail="unsupported by family (none requested)",
            )
        install(build)
        committed = bool(getattr(model, "uses_pipeline_cpu_offload", False))
        if requested and not committed:
            raise RuntimeError(
                f"{type(model).__name__} did not commit the requested pipeline "
                f"CPU-offload mode {mode!r}",
            )
        if committed and not bool(getattr(model, "pipeline_cpu_offload_healthy", False)):
            raise RuntimeError(
                f"{type(model).__name__} installed unusable pipeline CPU-offload hooks",
            )
        return PassResult(
            name=self.name,
            applied=committed,
            detail=f"mode={mode}",
        )


@dataclass(frozen=True, slots=True)
class VaeDecodeMemoryPass:
    """Trade VAE decode latency for peak memory (tiling / slicing).

    The one pass that touches a DISJOINT subtree: the VAE is not a policy core,
    so this has no ordering relation to any other pass and could run anywhere in
    the sequence. It is last only because there is no reason for it to be first.
    """

    name: str = "vae_decode_memory"
    replaces_modules: bool = False

    def enabled(self, build: Any) -> bool:
        memory = getattr(build, "generation_memory", None)
        return getattr(memory, "vae_decode", None) is not None

    def apply(self, model: Any, build: Any) -> PassResult:
        from vrl.models.steps.denoise.common.vae_decode_memory import (
            apply_generation_memory_policy,
        )

        apply_generation_memory_policy(
            model,
            memory=build.generation_memory,
            owner=f"{getattr(build, 'family', '?')} model",
        )
        return PassResult(
            name=self.name,
            applied=True,
            detail=str(build.generation_memory.vae_decode),
        )


# Order is the contract; see the module docstring. Offload is last because its
# hooks must see the final tree; VAE memory is unordered (disjoint subtree) and
# sits last only because nothing requires it earlier.
ROLLOUT_PASSES: tuple[OptimizationPass, ...] = (
    QuantizationPass(),
    CompilePass(),
    OffloadPass(),
    VaeDecodeMemoryPass(),
)

# Optimizations that skip or approximate a rollout forward without a build-time
# module seam, so they are configured per request rather than as a pass. Keyed by
# the ``sampling`` key that switches them on; the value is the drift each one
# introduces, which is what ``unguarded_drift_sources`` reports.
REQUEST_SCOPED_DRIFT_SOURCES: dict[str, str] = {
    "teacache": "reuses a cached noise_pred on skipped denoise steps",
}


def apply_rollout_optimizations(
    model: Any,
    build: Any,
    *,
    before_compile: Callable[[], None] | None = None,
) -> None:
    """Run every enabled rollout pass in dependency order.

    Call AFTER LoRA attachment and BEFORE offload-hook installation (see the
    module docstring for why each boundary is where it is).

    ``before_compile`` runs once between the module-mutating passes and the
    compile pass. That seam is where the policy moves to its device: the move
    must see the quantized tree (so a large policy never holds both a source
    master and its cache on GPU) but must not act on a compiled wrapper.
    """

    # Cross-feature conflicts are refused at config load
    # (vrl.config.validation.compile_conflicts): every conflict is decided by
    # config keys this build object does not carry, so a per-pass conflicts()
    # seam here could never fire — it existed and never did.
    # Each pass verifies its own EFFECT on the real modules (see
    # ``validate_every_core_quantized`` and CompilePass.apply). There is no
    # coverage check here, because a pass reporting which roots it touched would
    # be certifying itself -- and a pass buggy enough to miss a root is exactly
    # the one whose self-report is wrong.
    seam_done = False
    for optimization in ROLLOUT_PASSES:
        # Fire the seam before the first REPLACING pass, whether or not it is
        # enabled, so the device move happens exactly once either way.
        if optimization.replaces_modules and not seam_done:
            seam_done = True
            if before_compile is not None:
                before_compile()
        if not optimization.enabled(build):
            continue
        result = optimization.apply(model, build)
        logger.info("rollout pass %s applied (%s)", result.name, result.detail)
    if not seam_done and before_compile is not None:
        before_compile()  # no replacing pass registered at all


def unguarded_drift_sources(sampling: Any, precision: Any) -> tuple[str, ...]:
    """Request-scoped drift sources this run's precision policy will not cover.

    The trainer already arms the precision drift guard and TIS correction
    automatically whenever rollout and training precision differ
    (``PrecisionPolicy.stages_match`` is False, and the guard's default
    ``mode="auto"`` resolves to ``"fail"``). Quantization therefore needs no
    extra check here — it moves rollout precision, so it flips exactly that
    switch.

    A request-scoped optimization does NOT: TeaCache approximates the rollout
    ``noise_pred`` while leaving both roles' precision labels identical, so
    ``stages_match`` stays True and every automatic correction stays off. That
    is the uncovered case, and it is the one this returns.
    """

    if precision is not None and not getattr(precision, "stages_match", True):
        return ()  # guard + TIS already armed by the precision split
    if not isinstance(sampling, Mapping):
        return ()
    return tuple(
        f"{key} ({reason})"
        for key, reason in REQUEST_SCOPED_DRIFT_SOURCES.items()
        if _sampling_enables(sampling.get(key))
    )


def _sampling_enables(value: Any) -> bool:
    """Whether a ``sampling.<key>`` value switches its optimization on.

    Mirrors the accepted spellings of ``TeaCacheConfig.from_sampling``: absent /
    False / ``{enabled: false}`` are off, True and any other mapping are on.
    """

    if value is None or value is False:
        return False
    if isinstance(value, Mapping):
        return bool(value.get("enabled", True))
    return bool(value)


__all__ = [
    "REQUEST_SCOPED_DRIFT_SOURCES",
    "ROLLOUT_PASSES",
    "CompilePass",
    "OptimizationPass",
    "PassResult",
    "QuantizationPass",
    "apply_rollout_optimizations",
    "unguarded_drift_sources",
]
