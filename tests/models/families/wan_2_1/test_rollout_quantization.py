"""Wan rollout quantization must cover BOTH experts, not just ``transformer``.

Wan is the only dual-transformer family: a dual-stage checkpoint carries
``transformer`` and ``transformer_2`` and switches between them at
``boundary_ratio``. ``torch_compile_transformer`` is overridden to walk both,
but quantization used to inherit the single-transformer base implementation,
so ``transformer_2`` silently stayed at the rollout base dtype.

That is worse than half the speedup: the two experts then run at different
precisions inside one sampling trajectory, so rollout-vs-replay drift differs
either side of the boundary and the drift guard's premise no longer holds. The
loader's fallback guard cannot catch it either -- it only asserts the swap count
is non-zero, which a half-quantized model satisfies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel
from vrl.nn.quantization import QuantizedLinear


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
def test_wan_rollout_quantization_covers_both_experts(scheme: str) -> None:
    model = _dual_stage_model()
    assert model.transformer_2 is not None, "fixture is not dual-stage"

    swapped = model.quantize_rollout_fp8() if scheme == "fp8" else model.quantize_rollout_nvfp4()

    assert swapped, "no linears swapped -- min_features/exclude gated everything"
    assert _quantized_paths(model.transformer), "expert 1 (transformer) not quantized"
    assert _quantized_paths(model.transformer_2), (
        "expert 2 (transformer_2) not quantized -- it would silently run at the "
        "rollout base dtype while expert 1 runs quantized"
    )
    # The reported paths must name both experts, so the loader's count-based
    # guard and the run log agree with what actually changed.
    assert any(path.startswith("transformer_2") for path in swapped), (
        f"swap report omits transformer_2: {swapped}"
    )


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

    swapped = model.quantize_rollout_fp8()

    assert swapped
    assert _quantized_paths(model.transformer)
