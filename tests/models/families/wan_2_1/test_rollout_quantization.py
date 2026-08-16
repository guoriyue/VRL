"""Wan rollout quantization must cover BOTH experts, not just ``transformer``.

Wan is the only dual-transformer family: a dual-stage checkpoint carries
``transformer`` and ``transformer_2`` and switches between them at
``boundary_ratio``. Quantization used to inherit a single-transformer base
implementation that hardcoded ``self.transformer``, so ``transformer_2``
silently stayed at the rollout base dtype.

That is worse than half the speedup: the two experts then run at different
precisions inside one sampling trajectory, so rollout-vs-replay drift differs
either side of the boundary and the drift guard's premise no longer holds.

Coverage now comes from Wan's ``policy_cores`` declaration rather than a
per-scheme override, so these assertions also pin the declaration: any future
pass automatically reaches both experts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel
from vrl.models.loader import apply_rollout_quantization
from vrl.nn.quantization import QuantizedLinear


def _rollout_build(scheme: str) -> SimpleNamespace:
    """A minimal rollout build requesting ``scheme`` with no compile / no sync."""

    return SimpleNamespace(
        device="cpu",
        torch_compile=None,
        family="wan_2_1",
        precision=SimpleNamespace(
            quantization=SimpleNamespace(
                format=scheme,
                recipe="rowwise" if scheme == "fp8" else None,
            ),
        ),
        rollout=SimpleNamespace(base_weight_sync=True),
    )


class _TinyExpert(nn.Module):
    """One MLP-shaped linear, wide enough to clear the min_features gate.

    Two targeting rules have to be satisfied for this fixture to exercise
    anything: ``swap_linears`` skips linears under 1024 in/out features, and
    the NVFP4 profile is ``mlp_only``, so the path must carry an MLP segment
    (``vrl/nn/quantization/targeting.py``). A 2x2 ``to_q`` clears neither and
    would make the test pass vacuously.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ffn = nn.Module()
        self.ffn.net = nn.Linear(1024, 1024, bias=False)


def _dual_stage_model() -> WanT2VDiffusersModel:
    """A Wan model with both experts present (boundary_ratio makes it dual-stage)."""

    return WanT2VDiffusersModel(
        pipeline=SimpleNamespace(
            transformer=_TinyExpert(),
            transformer_2=_TinyExpert(),
            config={"boundary_ratio": 0.5},
            device="cpu",
            scheduler=None,
        ),
        device="cpu",
        trainable_transformers=["transformer"],
    )


def _quantized_paths(module: nn.Module) -> set[str]:
    return {name for name, child in module.named_modules() if isinstance(child, QuantizedLinear)}


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4"])
def test_wan_rollout_quantization_covers_both_experts(scheme: str, monkeypatch) -> None:
    model = _dual_stage_model()
    assert model.transformer_2 is not None, "fixture is not dual-stage"
    # NVFP4's Blackwell gate is asserted in tests/nn/quantization/test_fp4.py; here
    # it would only stop the walk this test is about from ever running on CPU.
    monkeypatch.setattr("vrl.nn.quantization.nvfp4_available", lambda _device=None: True)

    count = apply_rollout_quantization(model, _rollout_build(scheme))

    assert count, "no linears swapped -- min_features/exclude gated everything"
    assert _quantized_paths(model.transformer), "expert 1 (transformer) not quantized"
    assert _quantized_paths(model.transformer_2), (
        "expert 2 (transformer_2) not quantized -- it would silently run at the "
        "rollout base dtype while expert 1 runs quantized"
    )


def test_wan_policy_cores_declares_both_experts() -> None:
    """The declaration -- not any per-scheme override -- is what covers both.

    Pinning it directly means a future pass (compile, a new scheme) inherits
    dual-expert coverage without its own Wan-specific branch.
    """

    model = _dual_stage_model()

    assert set(model.policy_cores) == {"transformer", "transformer_2"}
    # trainable_modules is deliberately NARROWER here (this fixture trains only
    # expert 1). Reusing it for optimization would reintroduce the half-quantized
    # bug, so the two must not be conflated.
    assert set(model.trainable_modules) == {"transformer"}


def test_wan_single_expert_model_still_quantizes() -> None:
    """A t2v-1.3B checkpoint has no transformer_2; the walk must not require one."""

    model = WanT2VDiffusersModel(
        pipeline=SimpleNamespace(
            transformer=_TinyExpert(),
            config={},
            device="cpu",
            scheduler=None,
        ),
        device="cpu",
    )
    assert model.transformer_2 is None
    assert set(model.policy_cores) == {"transformer"}

    assert apply_rollout_quantization(model, _rollout_build("fp8"))
    assert _quantized_paths(model.transformer)
