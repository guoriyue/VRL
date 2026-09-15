"""The fused GELU-projection swap: same parameters, same names, reference math.

The rollout pass applies it from ``model.fused_gelu_projection``; these tests pin
what the swap must preserve for weight sync (state_dict names, parameter
identity), which modules it may touch (tanh-approximate with a biased
projection, nothing else), and that every path a CPU host can reach -- and the
gradient-enabled path on any host -- is the diffusers reference kernel.
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from torch import nn

from vrl.nn.optimization import ROLLOUT_PASSES, apply_rollout_optimizations
from vrl.nn.optimization.fused_gelu_projection import (
    FusedGELUProjection,
    fuse_gelu_projections,
)


class _Block(nn.Module):
    """One tanh feed-forward, one exact-GELU feed-forward, one bias-free tanh one."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.ff = FeedForward(dim, activation_fn="gelu-approximate")
        self.ff_exact = FeedForward(dim, activation_fn="gelu")
        self.ff_no_bias = FeedForward(dim, activation_fn="gelu-approximate", bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x) + self.ff_exact(x) + self.ff_no_bias(x)


def _policy(blocks: int = 2) -> nn.Module:
    root = nn.Module()
    root.transformer_blocks = nn.ModuleList(_Block() for _ in range(blocks))
    return root


def test_swap_takes_tanh_biased_projections_only_and_keeps_names() -> None:
    root = _policy()
    before = dict(root.named_parameters())

    count = fuse_gelu_projections(root)

    assert count == 2, "one tanh+bias feed-forward per block; exact and bias-free stay"
    block = root.transformer_blocks[0]
    assert isinstance(block.ff.net[0], FusedGELUProjection)
    assert type(block.ff_exact.net[0]) is GELU
    assert type(block.ff_no_bias.net[0]) is GELU
    after = dict(root.named_parameters())
    assert after.keys() == before.keys()
    assert all(after[name] is before[name] for name in before), "weight sync targets moved"
    assert list(root.state_dict()) == list(before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_forward_and_backward_are_the_reference_kernel(dtype: torch.dtype) -> None:
    """Off CUDA (and under grad anywhere) the module IS diffusers' GELU."""

    torch.manual_seed(0)
    reference = _policy().to(dtype)
    fused = _policy().to(dtype)
    fused.load_state_dict(reference.state_dict())
    assert fuse_gelu_projections(fused) == 2

    x = torch.randn(2, 8, 16).to(dtype)
    with torch.no_grad():
        for ref_block, fused_block in zip(
            reference.transformer_blocks, fused.transformer_blocks, strict=True
        ):
            assert torch.equal(fused_block(x), ref_block(x))

    ref_x = x.clone().requires_grad_(True)
    fused_x = x.clone().requires_grad_(True)
    ref_out = sum(block(ref_x) for block in reference.transformer_blocks)
    fused_out = sum(block(fused_x) for block in fused.transformer_blocks)
    assert torch.equal(fused_out, ref_out)
    ref_out.float().sum().backward()
    fused_out.float().sum().backward()
    assert torch.equal(fused_x.grad, ref_x.grad)
    proj = "transformer_blocks.0.ff.net.0.proj.weight"
    assert torch.equal(
        dict(fused.named_parameters())[proj].grad,
        dict(reference.named_parameters())[proj].grad,
    )


def test_wrapped_or_mismatched_projection_keeps_the_reference_path() -> None:
    """A LoRA/quantized wrapper or a dtype split must not reach the epilogue."""

    class _Wrapped(nn.Linear):
        pass

    module = GELU(16, 32, approximate="tanh")
    fused = FusedGELUProjection.from_module(module)
    x = torch.randn(4, 16)

    fused.proj = _Wrapped(16, 32)
    assert not fused._epilogue_applies(x)
    fused.proj = nn.Linear(16, 32).to(torch.bfloat16)
    assert not fused._epilogue_applies(x)  # fp32 input through a bf16 projection
    fused.proj = nn.Linear(16, 32, bias=False)
    assert not fused._epilogue_applies(x)


def test_rollout_pass_reaches_every_core_only_when_enabled() -> None:
    from types import SimpleNamespace

    from vrl.config.precision import RolePrecision

    names = [type(optimization).__name__ for optimization in ROLLOUT_PASSES]
    assert names.index("FusedGeluProjectionPass") < names.index("QuantizationPass"), (
        "must swap before the GEMM swap so a quantized projection lands inside the module"
    )

    def build(flag: bool) -> SimpleNamespace:
        return SimpleNamespace(
            device="cpu",
            family="test",
            torch_compile=None,
            fused_gelu_projection=flag,
            precision=RolePrecision("bf16", "tf32", None),
            rollout=SimpleNamespace(base_weight_sync=True),
        )

    def fused_projections(model: SimpleNamespace) -> int:
        return sum(
            isinstance(module, FusedGELUProjection)
            for core in model.policy_cores.values()
            for module in core.modules()
        )

    model = SimpleNamespace(
        policy_cores={"transformer": _policy(), "transformer_2": _policy()},
        quantization_exclude=(),
    )
    apply_rollout_optimizations(model, build(False))
    assert fused_projections(model) == 0

    apply_rollout_optimizations(model, build(True))
    assert fused_projections(model) == 4, "both experts must project through one kernel"


def test_real_sd3_transformer_swaps_every_feed_forward_and_cosmos_none() -> None:
    from tests.models.steps.denoise.fixtures import (
        TINY_SD3_JOINT_DIM,
        TINY_SD3_LATENT_SHAPE,
        TINY_SD3_POOLED_DIM,
        build_tiny_cosmos_transformer,
        build_tiny_sd3_transformer,
    )

    assert fuse_gelu_projections(build_tiny_cosmos_transformer()) == 0, "exact GELU, no bias"

    reference = build_tiny_sd3_transformer()
    fused = build_tiny_sd3_transformer()
    # One layer: the last joint block is context_pre_only, so only the image
    # stream's ff exists; a full-depth SD3.5 swaps ff and ff_context per block.
    assert fuse_gelu_projections(fused) == 1

    torch.manual_seed(1)
    kwargs = dict(
        hidden_states=torch.randn(*TINY_SD3_LATENT_SHAPE),
        encoder_hidden_states=torch.randn(2, 3, TINY_SD3_JOINT_DIM),
        pooled_projections=torch.randn(2, TINY_SD3_POOLED_DIM),
        timestep=torch.full((2,), 500.0),
        return_dict=False,
    )
    with torch.no_grad():
        assert torch.equal(fused(**kwargs)[0], reference(**kwargs)[0])


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_epilogue_matches_the_fp32_truth_within_one_rounding(dtype: torch.dtype) -> None:
    """The epilogue rounds once (GEMM -> GELU -> dtype); the reference rounds twice.

    So the two half-precision paths may sit ~2 ulps apart from each other, and
    the assertion is against the fp32 truth of the same inputs: the fused output
    within one rounding of it, the reference within two.
    """

    torch.manual_seed(0)
    reference = GELU(64, 256, approximate="tanh").to("cuda", dtype)
    fused = FusedGELUProjection.from_module(reference)
    x = torch.randn(3, 32, 64, device="cuda", dtype=dtype)
    proj = reference.proj
    truth = torch.nn.functional.gelu(
        torch.nn.functional.linear(x.float(), proj.weight.float(), proj.bias.float()),
        approximate="tanh",
    )
    # fp32 has no second rounding to speak of; its slack is the epilogue's own
    # tanh evaluation (~1e-5) plus accumulation order.
    eps = torch.finfo(dtype).eps if dtype != torch.float32 else 0.0
    atol = 1e-4

    with torch.no_grad():
        assert fused._epilogue_applies(x)
        fused_out = fused(x)
        ref_out = reference(x)
    assert fused_out.dtype == dtype
    assert ((fused_out.float() - truth).abs() <= eps * truth.abs() + atol).all()
    assert ((ref_out.float() - truth).abs() <= 2 * eps * truth.abs() + 1e-3).all()
    with torch.enable_grad():
        assert not fused._epilogue_applies(x)
