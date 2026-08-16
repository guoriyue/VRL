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
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from vrl.utils.logging import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class PassResult:
    """What one pass did to the rollout policy."""

    name: str
    applied: bool
    detail: str
    # RL semantics, not log text: a pass that makes the rollout forward differ
    # from the trainer's exact replay forward must be paired with a drift
    # correction. ``OptimizationReport.drift_sources`` is the consumer.
    introduces_replay_drift: bool
    # Policy-core keys this pass changed. The consumer is the coverage check in
    # ``apply_rollout_optimizations``: a multi-root family whose pass reached
    # only one root would otherwise split one trajectory across two precisions.
    touched: tuple[str, ...] = ()


@runtime_checkable
class OptimizationPass(Protocol):
    """A build-time transformation of the rollout policy.

    A structural contract: concrete passes satisfy it by shape and must not
    inherit it. ``apply`` is called only when ``enabled`` returned True.
    """

    name: str
    introduces_replay_drift: bool
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


@dataclass(frozen=True, slots=True)
class QuantizationPass:
    """Swap large policy GEMMs to the configured low-precision scheme."""

    name: str = "quantization"
    # Low-precision GEMMs make the rollout forward differ from the trainer's
    # base-dtype replay forward.
    introduces_replay_drift: bool = True
    # Swaps Linears inside the existing tree; the root object is unchanged.
    replaces_modules: bool = False

    def enabled(self, build: Any) -> bool:
        return getattr(getattr(build, "precision", None), "quantization", None) is not None

    def apply(self, model: Any, build: Any) -> PassResult:
        from vrl.models.loader import apply_rollout_quantization

        count = apply_rollout_quantization(model, build)
        quantization = build.precision.quantization
        return PassResult(
            name=self.name,
            applied=bool(count),
            detail=f"{quantization.format}: {count} linears",
            introduces_replay_drift=True,
            touched=tuple(model.policy_cores),
        )


@dataclass(frozen=True, slots=True)
class CompilePass:
    """torch.compile every policy core.

    Compilation is a graph-level rewrite of the SAME math, so unlike
    quantization and TeaCache it is not a drift source: it does not need the
    drift guard armed.
    """

    name: str = "compile"
    introduces_replay_drift: bool = False
    # Wraps each root in an OptimizedModule, hiding the real module.
    replaces_modules: bool = True

    def enabled(self, build: Any) -> bool:
        return bool((getattr(build, "torch_compile", None) or {}).get("enable"))

    def apply(self, model: Any, build: Any) -> PassResult:
        mode = build.torch_compile["mode"]
        touched = tuple(model.policy_cores)
        model.torch_compile_transformer(mode)
        return PassResult(
            name=self.name,
            applied=True,
            detail=f"mode={mode}",
            introduces_replay_drift=False,
            touched=touched,
        )


# Order is the contract (quantization before compile); see the module docstring.
ROLLOUT_PASSES: tuple[OptimizationPass, ...] = (QuantizationPass(), CompilePass())

# Optimizations that skip or approximate a rollout forward without a build-time
# module seam, so they are configured per request rather than as a pass. Keyed by
# the ``sampling`` key that switches them on; the value is the drift each one
# introduces, which is what ``unguarded_drift_sources`` reports.
REQUEST_SCOPED_DRIFT_SOURCES: dict[str, str] = {
    "teacache": "reuses a cached noise_pred on skipped denoise steps",
}


@dataclass(slots=True)
class OptimizationReport:
    """Every pass's outcome for one rollout build."""

    results: tuple[PassResult, ...] = field(default_factory=tuple)

    @property
    def applied(self) -> tuple[PassResult, ...]:
        return tuple(result for result in self.results if result.applied)

    @property
    def drift_sources(self) -> tuple[str, ...]:
        """Applied passes that make rollout diverge from the exact replay forward."""

        return tuple(result.name for result in self.applied if result.introduces_replay_drift)


def apply_rollout_optimizations(
    model: Any,
    build: Any,
    *,
    before_compile: Callable[[], None] | None = None,
) -> OptimizationReport:
    """Run every enabled rollout pass in dependency order.

    Call AFTER LoRA attachment and BEFORE offload-hook installation (see the
    module docstring for why each boundary is where it is).

    ``before_compile`` runs once between the module-mutating passes and the
    compile pass. That seam is where the policy moves to its device: the move
    must see the quantized tree (so a large policy never holds both a source
    master and its cache on GPU) but must not act on a compiled wrapper.
    """

    enabled = tuple(o for o in ROLLOUT_PASSES if o.enabled(build))
    # Only read the declaration when something will actually use it: a build with
    # no optimization requested must not force every model to own policy cores.
    declared = tuple(model.policy_cores) if enabled else ()
    if enabled and not declared:
        raise RuntimeError(
            f"{type(model).__name__} declares no policy_cores, but this build "
            f"requests {', '.join(o.name for o in enabled)}. A rollout policy must "
            "declare the module root(s) it samples through.",
        )
    results: list[PassResult] = []
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
        # A pass that reached only some roots would leave one expert of a
        # multi-expert policy unoptimized -- the failure mode ``policy_cores``
        # exists to make structurally impossible. Catch it here rather than
        # trusting every pass to walk the declaration correctly.
        missed = [name for name in declared if name not in result.touched]
        if result.applied and missed:
            raise RuntimeError(
                f"rollout pass {result.name!r} reached "
                f"{', '.join(result.touched) or 'no'} policy core(s) but the model "
                f"declares {', '.join(declared)}; {', '.join(missed)} would keep "
                "running unoptimized inside the same trajectory.",
            )
        results.append(result)
        logger.info("rollout pass %s applied (%s)", result.name, result.detail)
    if not seam_done and before_compile is not None:
        before_compile()  # no replacing pass registered at all
    return OptimizationReport(tuple(results))


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
    "OptimizationReport",
    "PassResult",
    "QuantizationPass",
    "apply_rollout_optimizations",
    "unguarded_drift_sources",
]
