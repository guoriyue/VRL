"""The rollout optimization layer: declaration in, passes out, order preserved.

Families declare WHAT to optimize (``policy_cores`` / ``quantization_exclude``),
passes own HOW, and ``apply_rollout_optimizations`` owns the ORDER. These tests
pin the seams between those three so a new scheme needs no family change and a
new family needs no pass change.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from torch import nn

from vrl.nn.optimization import (
    ROLLOUT_PASSES,
    CompilePass,
    OptimizationPass,
    PassResult,
    QuantizationPass,
    apply_rollout_optimizations,
    unguarded_drift_sources,
)
from vrl.nn.quantization import QuantizedLinear


class _Policy:
    """A rollout policy declaring ``roots`` MLP-shaped, quantizable cores."""

    quantization_exclude: tuple[str, ...] = ()

    def __init__(self, *roots: str) -> None:
        self._cores: dict[str, nn.Module] = {}
        for name in roots:
            root = nn.Module()
            root.ff = nn.Module()
            root.ff.net = nn.Linear(1024, 1024, bias=False)
            self._cores[name] = root
        self.compiled: list[str] = []

    @property
    def policy_cores(self) -> dict[str, nn.Module]:
        return self._cores

    def torch_compile_transformer(self, mode: str) -> None:
        self.compiled = [f"{name}:{mode}" for name in self._cores]


def _build(
    *,
    quantization: str | None = None,
    compile_mode: str | None = None,
) -> SimpleNamespace:
    from vrl.config.precision import QuantizationPolicy, RolePrecision

    policy = None if quantization is None else QuantizationPolicy(format=quantization)
    return SimpleNamespace(
        device="cpu",
        family="test",
        torch_compile=None if compile_mode is None else {"enable": True, "mode": compile_mode},
        precision=RolePrecision("bf16", "tf32", policy),
        rollout=SimpleNamespace(base_weight_sync=True),
    )


def _quantized(root: nn.Module) -> int:
    return sum(isinstance(module, QuantizedLinear) for module in root.modules())


# --- the declaration is the only thing a family supplies -----------------------


def test_passes_reach_every_declared_root() -> None:
    """A multi-root policy is optimized whole, with no per-family pass branch."""

    model = _Policy("transformer", "transformer_2")

    apply_rollout_optimizations(model, _build(quantization="fp8"))

    assert _quantized(model.policy_cores["transformer"])
    assert _quantized(model.policy_cores["transformer_2"]), (
        "second root left unoptimized -- one trajectory would span two precisions"
    )


def test_declared_exclude_is_honored() -> None:
    """The family's exclude list reaches the scheme without the scheme knowing it."""

    model = _Policy("transformer")
    model.quantization_exclude = ("ff",)

    with pytest.raises(RuntimeError, match="matched 0 linears"):
        apply_rollout_optimizations(model, _build(quantization="fp8"))


def test_no_optimization_requested_touches_nothing() -> None:
    """A plain bf16 rollout runs no pass and needs no declaration."""

    model = _Policy("transformer")

    report = apply_rollout_optimizations(model, _build())

    assert report.applied == ()
    assert not _quantized(model.policy_cores["transformer"])
    assert model.compiled == []


def test_policy_without_cores_fails_only_when_a_pass_is_requested() -> None:
    coreless = SimpleNamespace(policy_cores={}, quantization_exclude=())

    apply_rollout_optimizations(coreless, _build())  # no pass -> no requirement

    with pytest.raises(RuntimeError, match="declares no policy_cores"):
        apply_rollout_optimizations(coreless, _build(quantization="fp8"))


# --- ordering is the assembly point's contract ---------------------------------


def test_quantization_runs_before_compile() -> None:
    """Inductor must trace the final module tree, so compile is last."""

    order: list[str] = []

    class _OrderPolicy(_Policy):
        @property
        def policy_cores(self) -> dict[str, nn.Module]:
            return self._cores

        def torch_compile_transformer(self, mode: str) -> None:
            order.append("compile")
            super().torch_compile_transformer(mode)

    model = _OrderPolicy("transformer")
    import vrl.nn.quantization as quantization

    real_swap = quantization.Fp8Linear.swap_linears
    original = quantization.Fp8Linear.swap_linears

    def _tracked(cls, root, **kwargs):
        order.append("quantize")
        return real_swap.__func__(cls, root, **kwargs)

    quantization.Fp8Linear.swap_linears = classmethod(_tracked)
    try:
        apply_rollout_optimizations(
            model,
            _build(quantization="fp8", compile_mode="default"),
        )
    finally:
        quantization.Fp8Linear.swap_linears = original

    assert order == ["quantize", "compile"]


def test_device_move_seam_runs_between_quantize_and_compile() -> None:
    """The move must see the quantized tree but not a compiled wrapper."""

    order: list[str] = []

    class _SeamPolicy(_Policy):
        def torch_compile_transformer(self, mode: str) -> None:
            order.append("compile")
            super().torch_compile_transformer(mode)

    model = _SeamPolicy("transformer")

    apply_rollout_optimizations(
        model,
        _build(quantization="fp8", compile_mode="default"),
        before_compile=lambda: order.append("move"),
    )

    assert order == ["move", "compile"]
    assert _quantized(model.policy_cores["transformer"]), "quantization ran before the move"


def test_device_move_seam_runs_even_with_compile_disabled() -> None:
    """The move is not conditional on the compile pass being enabled."""

    order: list[str] = []
    model = _Policy("transformer")

    apply_rollout_optimizations(
        model,
        _build(quantization="fp8"),
        before_compile=lambda: order.append("move"),
    )

    assert order == ["move"]


def test_partial_coverage_fails_loud() -> None:
    """A pass that silently reaches only some roots must not pass unnoticed."""

    class _HalfPass:
        name = "half"
        introduces_replay_drift = False
        replaces_modules = False

        def enabled(self, build: Any) -> bool:
            return True

        def conflicts(self, build: Any) -> tuple[str, ...]:
            return ()

        def apply(self, model: Any, build: Any) -> PassResult:
            first = next(iter(model.policy_cores))
            return PassResult(
                name=self.name,
                applied=True,
                detail="reached one root",
                introduces_replay_drift=False,
                touched=(first,),
            )

    model = _Policy("transformer", "transformer_2")
    import vrl.nn.optimization.passes as passes

    original = passes.ROLLOUT_PASSES
    passes.ROLLOUT_PASSES = (_HalfPass(),)
    try:
        with pytest.raises(RuntimeError, match="transformer_2 would keep running"):
            apply_rollout_optimizations(model, _build())
    finally:
        passes.ROLLOUT_PASSES = original


# --- registry shape ------------------------------------------------------------


def test_registered_passes_satisfy_the_protocol_structurally() -> None:
    """Concrete passes conform by shape and must NOT inherit the Protocol.

    A consumer-facing Protocol is a structural contract, not an implementation
    base; inheriting it to advertise conformance would let a class ship the
    Protocol's ``...`` stubs as real methods.
    """

    assert ROLLOUT_PASSES, "no passes registered"
    for optimization in ROLLOUT_PASSES:
        assert isinstance(optimization, OptimizationPass)
        assert OptimizationPass not in type(optimization).__mro__


def test_compile_is_not_a_drift_source_but_quantization_is() -> None:
    """Compile rewrites the graph of the SAME math; quantization changes it."""

    assert QuantizationPass().introduces_replay_drift is True
    assert CompilePass().introduces_replay_drift is False


# --- drift accounting ----------------------------------------------------------


def test_teacache_is_reported_as_unguarded_when_precision_matches() -> None:
    """TeaCache drifts while both roles share precision, so nothing auto-arms.

    ``stages_match`` True is exactly the case the trainer's automatic guard and
    TIS correction do NOT cover, which is why this reports it.
    """

    matched = SimpleNamespace(stages_match=True)

    assert unguarded_drift_sources({"teacache": True}, matched)
    assert unguarded_drift_sources({"teacache": {"threshold": 0.1}}, matched)


def test_teacache_off_or_precision_split_reports_nothing() -> None:
    matched = SimpleNamespace(stages_match=True)
    split = SimpleNamespace(stages_match=False)

    assert unguarded_drift_sources({}, matched) == ()
    assert unguarded_drift_sources({"teacache": False}, matched) == ()
    assert unguarded_drift_sources({"teacache": {"enabled": False}}, matched) == ()
    # A precision split already arms the drift guard + TIS automatically.
    assert unguarded_drift_sources({"teacache": True}, split) == ()


# --- declared conflicts --------------------------------------------------------
#
# The second relation between passes. Ordering is the ROLLOUT_PASSES tuple;
# this is "these two must never both be on, in either order". It is declared so
# that a new incompatibility has one home -- compile alone is currently refused
# against grad-checkpointing, FSDP2, blockwise fp8 and sequence parallelism by
# four separate hand-written guards in four different files.


def test_a_declared_conflict_fails_before_any_mutation() -> None:
    """An incompatible combination must leave the model untouched."""

    class _ConflictingPass:
        name = "conflicting"
        introduces_replay_drift = False
        replaces_modules = False

        def enabled(self, build: Any) -> bool:
            return True

        def conflicts(self, build: Any) -> tuple[str, ...]:
            return ("some other feature (because reasons)",)

        def apply(self, model: Any, build: Any) -> PassResult:
            raise AssertionError("apply must not run when a conflict is declared")

    model = _Policy("transformer")
    import vrl.nn.optimization.passes as passes

    original = passes.ROLLOUT_PASSES
    passes.ROLLOUT_PASSES = (_ConflictingPass(),)
    try:
        with pytest.raises(ValueError, match="some other feature"):
            apply_rollout_optimizations(model, _build())
    finally:
        passes.ROLLOUT_PASSES = original
    assert not _quantized(model.policy_cores["transformer"]), "model was mutated"


def test_a_disabled_pass_conflict_is_not_consulted() -> None:
    """Only ENABLED passes can conflict; an off feature constrains nothing."""

    class _OffButConflicting:
        name = "off"
        introduces_replay_drift = False
        replaces_modules = False

        def enabled(self, build: Any) -> bool:
            return False

        def conflicts(self, build: Any) -> tuple[str, ...]:
            raise AssertionError("conflicts must not be consulted for a disabled pass")

        def apply(self, model: Any, build: Any) -> PassResult:
            raise AssertionError("apply must not run")

    import vrl.nn.optimization.passes as passes

    original = passes.ROLLOUT_PASSES
    passes.ROLLOUT_PASSES = (_OffButConflicting(),)
    try:
        apply_rollout_optimizations(_Policy("transformer"), _build())
    finally:
        passes.ROLLOUT_PASSES = original


def test_shipped_passes_declare_no_conflict_from_model_build_alone() -> None:
    """Compile's real conflicts are config-level, not ModelBuild-level.

    Pinned deliberately: an earlier draft probed `build.rank_group`, which does
    not exist on ModelBuild -- a guard that can never fire. The sequence-parallel
    conflict lives in config validation, where engine topology is visible.
    """

    for optimization in ROLLOUT_PASSES:
        assert optimization.conflicts(_build(quantization="fp8", compile_mode="default")) == ()
