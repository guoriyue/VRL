"""Folding LoRA into the rollout base weights: exact, idempotent, all-or-nothing.

The mechanism's whole reason to exist is that PEFT's unmerge is lossy in bf16,
so these tests pin the two properties that follow from that: a merge must be
numerically equivalent to the adapter-active forward, and re-merging across
many weight syncs must not drift the base weights.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vrl.nn.optimization.lora_merge import (
    capture_pristine_base,
    lora_layers,
    merge_lora_into_base,
    require_every_core_merged,
)

peft = pytest.importorskip("peft")


class _Policy:
    """Rollout policy whose declared cores carry PEFT LoRA layers."""

    def __init__(self, *, cores: int = 1, dim: int = 32, rank: int = 4) -> None:
        from peft import LoraConfig, get_peft_model

        self._cores: dict[str, nn.Module] = {}
        for index in range(cores):
            base = nn.Sequential(nn.Linear(dim, dim, bias=False))
            self._cores[f"core{index}"] = get_peft_model(
                base,
                LoraConfig(
                    r=rank,
                    lora_alpha=2 * rank,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules=["0"],
                ),
            )

    @property
    def policy_cores(self) -> dict[str, nn.Module]:
        return self._cores

    def randomize_adapter(self) -> None:
        """Stand in for one weight sync delivering fresh adapter state."""

        with torch.no_grad():
            for core in self._cores.values():
                for _, layer in lora_layers(core):
                    layer.lora_A.default.weight.normal_(0.0, 0.05)
                    layer.lora_B.default.weight.normal_(0.0, 0.05)


def _forward(policy: _Policy, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return policy.policy_cores["core0"](x)


def test_merged_forward_matches_adapter_active_forward() -> None:
    """Folding is a rewrite of the same math, so fp32 must agree to round-off."""

    policy = _Policy()
    policy.randomize_adapter()
    x = torch.randn(8, 32)

    before = _forward(policy, x)
    merge_lora_into_base(policy, capture_pristine_base(policy))
    after = _forward(policy, x)

    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)


def test_repeated_merges_do_not_drift_the_base_weight() -> None:
    """The property PEFT's own unmerge does NOT have.

    Weight sync re-merges once per optimizer step, so the question is whether
    error compounds across cycles. Restoring from the pristine copy means it
    cannot: after many cycles with random adapters, folding a ZERO adapter must
    reproduce the captured base bit-for-bit, because the fold is always
    ``pristine + delta`` and never ``current - old_delta + new_delta``.
    """

    policy = _Policy()
    pristine = capture_pristine_base(policy)
    layer = next(iter(lora_layers(policy.policy_cores["core0"])))[1]

    for _ in range(200):
        policy.randomize_adapter()
        merge_lora_into_base(policy, pristine)

    with torch.no_grad():
        layer.lora_B.default.weight.zero_()
    merge_lora_into_base(policy, pristine)

    assert torch.equal(
        layer.get_base_layer().weight.detach().cpu(),
        pristine["core0.base_model.model.0"],
    )


def test_merge_is_exact_against_an_independently_computed_delta() -> None:
    """The folded weight is exactly ``pristine + (alpha / r) * B @ A``.

    Computed here from the adapter tensors rather than read back from PEFT, so
    a scaling or transpose regression in the fold cannot agree with itself.
    """

    policy = _Policy(dim=16, rank=4)
    policy.randomize_adapter()
    pristine = capture_pristine_base(policy)
    layer = next(iter(lora_layers(policy.policy_cores["core0"])))[1]

    a = layer.lora_A.default.weight.detach()
    b = layer.lora_B.default.weight.detach()
    expected = pristine["core0.base_model.model.0"] + layer.scaling["default"] * (b @ a)

    merge_lora_into_base(policy, pristine)

    torch.testing.assert_close(
        layer.get_base_layer().weight.detach().cpu(),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_merge_covers_every_declared_core() -> None:
    """A multi-expert family must not end up half-folded."""

    policy = _Policy(cores=2)
    policy.randomize_adapter()
    merged = merge_lora_into_base(policy, capture_pristine_base(policy))

    assert len(merged) == 2
    require_every_core_merged(policy)


def test_half_merged_policy_is_rejected() -> None:
    """The Wan two-expert bug, in merge form: one core folded, one not."""

    policy = _Policy(cores=2)
    policy.randomize_adapter()
    pristine = capture_pristine_base(policy)
    merge_lora_into_base(policy, pristine)
    next(iter(lora_layers(policy.policy_cores["core1"])))[1].unmerge()

    with pytest.raises(RuntimeError, match="unfolded"):
        require_every_core_merged(policy)


def test_core_without_lora_layers_is_rejected() -> None:
    """``use_lora`` with a core PEFT never wrapped is a wiring bug, not a no-op."""

    policy = _Policy()
    policy.policy_cores["bare"] = nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="no LoRA layers found"):
        require_every_core_merged(policy)


def test_capture_refuses_an_already_merged_layer() -> None:
    """A pristine snapshot taken after a merge would bake the adapter in."""

    policy = _Policy()
    policy.randomize_adapter()
    merge_lora_into_base(policy, capture_pristine_base(policy))

    with pytest.raises(RuntimeError, match="already merged"):
        capture_pristine_base(policy)


def test_merge_without_a_pristine_snapshot_is_rejected() -> None:
    """Folding onto an unknown base is the drift bug this module exists to stop."""

    policy = _Policy()
    with pytest.raises(RuntimeError, match="no pristine base weight"):
        merge_lora_into_base(policy, {})


def test_bf16_pristine_restore_beats_peft_unmerge_round_trips() -> None:
    """Why the pristine copy exists, stated as a contrast.

    bf16 carries ~3 decimal digits, so PEFT's ``unmerge`` (which subtracts the
    old delta back out) loses bits every cycle and compounds them. Rollout
    re-merges once per optimizer step, so this is a per-step leak. Restoring
    from the captured base has no such term.
    """

    steps = 200

    drifting = _Policy()
    drifting.policy_cores["core0"].to(torch.bfloat16)
    base0 = capture_pristine_base(drifting)["core0.base_model.model.0"]
    layer = next(iter(lora_layers(drifting.policy_cores["core0"])))[1]
    for _ in range(steps):
        drifting.randomize_adapter()
        layer.merge(safe_merge=True)
        layer.unmerge()
    peft_error = (layer.get_base_layer().weight.detach().cpu().float() - base0.float()).abs().max()

    kept = _Policy()
    kept.policy_cores["core0"].to(torch.bfloat16)
    pristine = capture_pristine_base(kept)
    kept_layer = next(iter(lora_layers(kept.policy_cores["core0"])))[1]
    for _ in range(steps):
        kept.randomize_adapter()
        merge_lora_into_base(kept, pristine)
    with torch.no_grad():
        kept_layer.lora_B.default.weight.zero_()
    merge_lora_into_base(kept, pristine)
    kept_error = (
        (
            kept_layer.get_base_layer().weight.detach().cpu().float()
            - pristine["core0.base_model.model.0"].float()
        )
        .abs()
        .max()
    )

    assert kept_error == 0.0
    assert peft_error > 0.0


_DEFAULT_LORA = {"rank": 32, "alpha": 64, "merge_for_rollout": True}


def _build(*, lora: dict | None = _DEFAULT_LORA) -> object:
    """Minimal ModelBuild-shaped stand-in for the pass's enable decision.

    ``ModelBuild.lora`` is None when use_lora is off, so one field answers both
    halves of the question -- pass ``lora=None`` for that case.
    """

    from types import SimpleNamespace

    return SimpleNamespace(lora=lora)


def test_pass_runs_before_quantization() -> None:
    """The fold must land in the plain Linear PEFT wrapped, not after the swap."""

    from vrl.nn.optimization import ROLLOUT_PASSES

    names = [pass_.name for pass_ in ROLLOUT_PASSES]
    assert names.index("lora_merge") < names.index("quantization")
    assert names.index("lora_merge") < names.index("compile")


def test_pass_declares_replay_drift() -> None:
    """(W+BA)x accumulates differently from Wx+B(Ax); the guard must know."""

    from vrl.nn.optimization.passes import LoraMergePass

    assert LoraMergePass().introduces_replay_drift is True
    assert LoraMergePass().replaces_modules is False


def test_pass_reads_exactly_one_config_key() -> None:
    """``model.lora.merge_for_rollout``, off unless asked for.

    ``ModelBuild.lora`` is already None when use_lora is off, so the pass needs
    no second condition -- and the replay path never runs the pass layer at all.
    """

    from vrl.nn.optimization.passes import LoraMergePass

    pass_ = LoraMergePass()
    assert pass_.enabled(_build()) is True
    assert pass_.enabled(_build(lora={"rank": 32, "alpha": 64})) is False
    assert pass_.enabled(_build(lora={"merge_for_rollout": False})) is False
    assert pass_.enabled(_build(lora=None)) is False


def test_pass_folds_and_publishes_the_pristine_base() -> None:
    """Weight sync re-folds from this snapshot, so the pass must hand it over."""

    from vrl.nn.optimization.passes import LoraMergePass

    policy = _Policy(cores=2)
    policy.randomize_adapter()

    result = LoraMergePass().apply(policy, _build())

    assert result.applied and result.introduces_replay_drift
    assert "2 LoRA layer(s)" in result.detail
    require_every_core_merged(policy)
    assert set(policy.lora_pristine_base) == {
        "core0.base_model.model.0",
        "core1.base_model.model.0",
    }


def test_refold_after_weight_sync_tracks_the_new_adapter() -> None:
    """The property the worker's re-fold exists to guarantee.

    Weight sync installs a fresh adapter onto an ALREADY folded policy. Until
    it is re-folded the base still carries the previous delta, so sampling
    would serve stale weights. After the re-fold the forward must equal an
    unfolded policy holding the same adapter -- checked against an independent
    model so a stale base cannot agree with itself.
    """

    folded = _Policy()
    reference = _Policy()
    with torch.no_grad():  # identical starting base weights
        ref_layer = next(iter(lora_layers(reference.policy_cores["core0"])))[1]
        fold_layer = next(iter(lora_layers(folded.policy_cores["core0"])))[1]
        ref_layer.get_base_layer().weight.copy_(fold_layer.get_base_layer().weight)

    pristine = capture_pristine_base(folded)
    x = torch.randn(8, 32)

    for _ in range(5):
        folded.randomize_adapter()
        merge_lora_into_base(folded, pristine)
        # The trainer's next push lands on the already-folded policy.
        with torch.no_grad():
            for layer, source in ((ref_layer, fold_layer),):
                layer.lora_A.default.weight.copy_(source.lora_A.default.weight)
                layer.lora_B.default.weight.copy_(source.lora_B.default.weight)
        torch.testing.assert_close(
            _forward(folded, x),
            _forward(reference, x),
            rtol=1e-5,
            atol=1e-5,
        )


def test_stale_base_without_refold_is_observable() -> None:
    """Counter-test: without the re-fold the folded policy serves old weights.

    Pins that the assertion above is actually load-bearing -- if re-folding were
    a no-op this test would fail and the one above would pass vacuously.
    """

    policy = _Policy()
    policy.randomize_adapter()
    pristine = capture_pristine_base(policy)
    merge_lora_into_base(policy, pristine)
    x = torch.randn(8, 32)
    before = _forward(policy, x)

    policy.randomize_adapter()  # new adapter arrives, no re-fold
    assert torch.equal(_forward(policy, x), before)

    merge_lora_into_base(policy, pristine)
    assert not torch.equal(_forward(policy, x), before)


def test_folding_removes_the_adapter_from_the_autograd_graph() -> None:
    """Why OnlineTrainer refuses a folded policy.

    PEFT routes a merged layer straight to its frozen base layer, so the adapter
    leaves the graph. This is the failure the trainer-side guard exists to catch
    before it shows up as a training run that changes nothing.
    """

    policy = _Policy()
    policy.randomize_adapter()
    layer = next(iter(lora_layers(policy.policy_cores["core0"])))[1]
    x = torch.randn(4, 32)

    policy.policy_cores["core0"](x).pow(2).mean().backward()
    assert layer.lora_A.default.weight.grad is not None

    policy.policy_cores["core0"].zero_grad(set_to_none=True)
    merge_lora_into_base(policy, capture_pristine_base(policy))

    with pytest.raises(RuntimeError, match="does not require grad"):
        policy.policy_cores["core0"](x).pow(2).mean().backward()
