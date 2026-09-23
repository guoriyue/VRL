"""The fused RMSNorm swap: same parameters, same names, same math to one ulp.

Both roles apply it from ``model.fused_rms_norm``; these tests pin what the swap
must preserve for weight sync (state_dict names, parameter identity), for the
trainer (gradients), and for the replay ratio (numerics against the hand-written
diffusers path it replaces).
"""

from __future__ import annotations

import torch
from diffusers.models.normalization import RMSNorm
from torch import nn

from vrl.nn.optimization import ROLLOUT_PASSES, apply_rollout_optimizations
from vrl.nn.optimization.fused_rms_norm import FusedRMSNorm, fuse_rms_norms


class _Attention(nn.Module):
    """A Q/K-normalized attention stub shaped like a diffusers DiT block."""

    def __init__(self, head_dim: int = 64) -> None:
        super().__init__()
        self.to_q = nn.Linear(head_dim, head_dim, bias=False)
        self.norm_q = RMSNorm(head_dim, eps=1e-6, elementwise_affine=True)
        self.norm_k = RMSNorm(head_dim, eps=1e-6, elementwise_affine=True)
        self.norm_v = nn.RMSNorm(head_dim, eps=1e-6)  # torch-native: already fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_q(self.to_q(x)) + self.norm_k(x) + self.norm_v(x)


def _policy(blocks: int = 2) -> nn.Module:
    root = nn.Module()
    root.transformer_blocks = nn.ModuleList(_Attention() for _ in range(blocks))
    return root


def test_swap_keeps_parameter_identity_and_state_dict_names() -> None:
    root = _policy()
    before = {name: param for name, param in root.named_parameters()}
    root.transformer_blocks[0].norm_q.weight.data.uniform_(0.5, 1.5)

    count = fuse_rms_norms(root)

    assert count == 4, "two diffusers norms per block, torch-native norm_v untouched"
    assert isinstance(root.transformer_blocks[0].norm_q, FusedRMSNorm)
    assert isinstance(root.transformer_blocks[0].norm_v, nn.RMSNorm)
    after = {name: param for name, param in root.named_parameters()}
    assert after.keys() == before.keys()
    assert all(after[name] is before[name] for name in before), "weight sync targets moved"
    assert list(root.state_dict()) == list(before)


def test_fp32_forward_and_backward_match_the_hand_written_norm() -> None:
    torch.manual_seed(0)
    reference = _policy()
    fused = _policy()
    fused.load_state_dict(reference.state_dict())
    for block in fused.transformer_blocks:
        block.norm_q.weight.data.uniform_(0.5, 1.5)
    reference.load_state_dict(fused.state_dict())
    assert fuse_rms_norms(fused) == 4

    x = torch.randn(2, 8, 16, 64)
    ref_x = x.clone().requires_grad_(True)
    fused_x = x.clone().requires_grad_(True)
    ref_out = sum(block(ref_x) for block in reference.transformer_blocks)
    fused_out = sum(block(fused_x) for block in fused.transformer_blocks)
    torch.testing.assert_close(fused_out, ref_out, rtol=1e-5, atol=1e-5)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    fused_out.backward(grad)
    torch.testing.assert_close(fused_x.grad, ref_x.grad, rtol=1e-5, atol=1e-5)
    for ref_block, fused_block in zip(
        reference.transformer_blocks, fused.transformer_blocks, strict=True
    ):
        torch.testing.assert_close(
            fused_block.norm_q.weight.grad,
            ref_block.norm_q.weight.grad,
            rtol=1e-5,
            atol=1e-5,
        )


def test_bf16_forward_is_within_one_rounding_of_the_hand_written_norm() -> None:
    """One fp32->bf16 rounding instead of two: at most one ulp apart, same dtype."""

    torch.manual_seed(0)
    reference = RMSNorm(64, eps=1e-6, elementwise_affine=True).to(torch.bfloat16)
    reference.weight.data.uniform_(0.5, 1.5)
    fused = FusedRMSNorm.from_module(reference)
    x = torch.randn(4, 8, 32, 64).to(torch.bfloat16)

    ref_out = reference(x)
    fused_out = fused(x)

    assert fused_out.dtype == torch.bfloat16
    ulp = torch.finfo(torch.bfloat16).eps * ref_out.abs().float()
    assert ((fused_out.float() - ref_out.float()).abs() <= ulp + 1e-6).all()


def test_mixed_dtype_keeps_the_weight_dtype_output() -> None:
    """An fp32 activation through a bf16 weight returns bf16, as diffusers does."""

    torch.manual_seed(0)
    reference = RMSNorm(64, eps=1e-6, elementwise_affine=True).to(torch.bfloat16)
    fused = FusedRMSNorm.from_module(reference)
    x = torch.randn(2, 4, 64)

    ref_out = reference(x)
    fused_out = fused(x)

    assert fused_out.dtype == torch.bfloat16
    torch.testing.assert_close(fused_out, ref_out)


def test_no_affine_norm_keeps_the_input_dtype() -> None:
    reference = RMSNorm(64, eps=1e-6, elementwise_affine=False)
    fused = FusedRMSNorm.from_module(reference)
    assert fused.weight is None
    x = torch.randn(2, 4, 64)

    torch.testing.assert_close(fused(x), reference(x), rtol=1e-5, atol=1e-5)


def test_torch_native_only_policy_swaps_nothing() -> None:
    root = nn.Module()
    root.norm = nn.RMSNorm(64)

    assert fuse_rms_norms(root) == 0
    assert isinstance(root.norm, nn.RMSNorm)


def test_rollout_pass_reaches_every_core_only_when_enabled() -> None:
    from types import SimpleNamespace

    from vrl.config.precision import RolePrecision

    assert type(ROLLOUT_PASSES[0]).__name__ == "FusedRmsNormPass", "runs before the GEMM swap"

    def build(flag: bool) -> SimpleNamespace:
        return SimpleNamespace(
            device="cpu",
            family="test",
            torch_compile=None,
            fused_rms_norm=flag,
            precision=RolePrecision("bf16", "tf32", None),
            rollout=SimpleNamespace(base_weight_sync=True),
        )

    def fused_norms(model: SimpleNamespace) -> int:
        return sum(
            isinstance(module, FusedRMSNorm)
            for core in model.policy_cores.values()
            for module in core.modules()
        )

    model = SimpleNamespace(
        policy_cores={"transformer": _policy(), "transformer_2": _policy()},
        quantization_exclude=(),
    )
    apply_rollout_optimizations(model, build(False))
    assert fused_norms(model) == 0

    apply_rollout_optimizations(model, build(True))
    assert fused_norms(model) == 8, "both experts must normalize through one kernel"


def test_real_cosmos_transformer_forward_is_unchanged() -> None:
    """The production eager site: every Q/K norm of a real Cosmos DiT swaps, same output."""

    from tests.models.steps.denoise.fixtures import (
        TINY_COSMOS_TEXT_DIM,
        build_tiny_transformer,
    )

    reference = build_tiny_transformer("cosmos")
    fused = build_tiny_transformer("cosmos")
    # time_embed.norm plus norm_q/norm_k of the self- and cross-attention.
    assert fuse_rms_norms(fused) == 5

    torch.manual_seed(1)
    kwargs = dict(
        hidden_states=torch.randn(2, 5, 1, 4, 4),
        timestep=torch.full((2,), 0.75),
        encoder_hidden_states=torch.randn(2, 3, TINY_COSMOS_TEXT_DIM),
        padding_mask=torch.zeros(1, 1, 4, 4),
        return_dict=False,
    )
    with torch.no_grad():
        torch.testing.assert_close(
            fused(**kwargs)[0], reference(**kwargs)[0], rtol=1e-6, atol=1e-6
        )
