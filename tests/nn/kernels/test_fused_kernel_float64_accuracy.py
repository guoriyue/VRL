"""The fused DiT kernels compute their reference math on CUDA, deterministically.

Each Triton kernel is compared with its float64 value: it must be within one
bf16 rounding of it (the single rounding it does) and no further from it than
the stock eager chain it replaces, with gradients to match.
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    joint_rms_norm_rope,
    layer_norm_modulate,
)

pytestmark = pytest.mark.gpu

_BF16_UNIT = 2.0**-8


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


def test_layer_norm_modulate_is_one_rounding_of_its_float64_value() -> None:
    torch.manual_seed(0)
    x = (torch.randn(2, 300, 192, device="cuda") * 3 + 1).bfloat16().requires_grad_()
    modulation = torch.randn(2, 3 * 192, device="cuda").bfloat16()
    shift, scale, _ = modulation.chunk(3, dim=-1)  # strided [B, D] views, as a block chunks
    scale.requires_grad_()
    dy = torch.randn(2, 300, 192, device="cuda").bfloat16()

    x64 = x.detach().double().requires_grad_()
    scale64 = scale.detach().double().requires_grad_()
    exact = torch.nn.functional.layer_norm(x64, (192,), eps=1e-6) * (1 + scale64[:, None])
    exact = exact + shift.double()[:, None]
    exact.backward(dy.double())
    with torch.autocast("cuda", torch.bfloat16):
        norm = torch.nn.LayerNorm(192, elementwise_affine=False, eps=1e-6)
        eager = (norm(x.detach()) * (1 + scale.detach()[:, None]) + shift[:, None]).bfloat16()
    fused = layer_norm_modulate(x, scale, shift, eps=1e-6)
    fused.backward(dy)

    assert fused.dtype == torch.bfloat16
    assert _relative(fused, exact) < _BF16_UNIT
    assert _relative(fused, exact) <= _relative(eager, exact)
    assert _relative(x.grad, x64.grad) < _BF16_UNIT
    assert _relative(scale.grad, scale64.grad) < _BF16_UNIT
    assert torch.equal(fused, layer_norm_modulate(x, scale, shift, eps=1e-6))


def test_gated_residual_rounds_once_and_passes_the_residual_gradient_through() -> None:
    torch.manual_seed(1)
    residual = torch.randn(2, 300, 192, device="cuda").bfloat16().requires_grad_()
    gate = torch.randn(2, 192, device="cuda").bfloat16().requires_grad_()
    y = torch.randn(2, 300, 192, device="cuda").bfloat16().requires_grad_()
    dout = torch.randn(2, 300, 192, device="cuda").bfloat16()

    out = gated_residual(residual, gate, y)
    out.backward(dout)
    exact = residual.double() + gate.double()[:, None] * y.double()

    assert _relative(out, exact.detach()) < _BF16_UNIT
    assert torch.equal(residual.grad, dout)
    assert _relative(y.grad, dout.double() * gate.double()[:, None]) < _BF16_UNIT
    assert _relative(gate.grad, (dout.double() * y.double()).sum(1)) < _BF16_UNIT


def test_joint_rms_norm_rope_matches_the_stock_norm_rotation_and_concatenation() -> None:
    torch.manual_seed(2)
    heads, dim = 3, 128
    text = torch.randn(2, 5, heads, dim, device="cuda").bfloat16()
    image = (torch.randn(2, 40, heads, dim, device="cuda") * 4).bfloat16()
    image[..., 10] *= 30  # one dominant channel, as Qwen-Image block 0's q has
    inputs = [text.requires_grad_(), image.requires_grad_()]
    weights = [
        (torch.rand(dim, device="cuda") + 0.5).bfloat16().requires_grad_() for _ in range(2)
    ]
    freqs = [
        torch.polar(
            torch.ones(n, dim // 2, device="cuda"), torch.rand(n, dim // 2, device="cuda") * 50
        )
        for n in (5, 40)
    ]
    dout = torch.randn(2, 45, heads, dim, device="cuda").bfloat16()

    def exact_value(xs, ws):
        parts = []
        for x, w, f in zip(xs, ws, freqs, strict=True):
            n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
            rotated = torch.view_as_complex(n.unflatten(-1, (-1, 2)).contiguous())
            parts.append(
                torch.view_as_real(rotated * f.to(torch.complex128)[None, :, None]).flatten(-2)
            )
        return torch.cat(parts, dim=1)

    xs64 = [x.detach().double().requires_grad_() for x in inputs]
    ws64 = [w.detach().double().requires_grad_() for w in weights]
    exact = exact_value(xs64, ws64)
    exact.backward(dout.double())
    eager_parts = []
    for x, w, f in zip(inputs, weights, freqs, strict=True):
        norm = RMSNorm(dim, eps=1e-6).to("cuda", torch.bfloat16)
        norm.weight.data.copy_(w.detach())
        eager_parts.append(apply_rotary_emb_qwen(norm(x.detach()), f, use_real=False))
    eager = torch.cat(eager_parts, dim=1)
    fused = joint_rms_norm_rope(inputs, weights, freqs, eps=1e-6)
    fused.backward(dout)

    assert fused.shape == (2, 45, heads, dim) and fused.dtype == torch.bfloat16
    assert _relative(fused, exact.detach()) < _BF16_UNIT
    assert _relative(fused, exact.detach()) <= _relative(eager, exact.detach())
    for x, x64 in zip(inputs, xs64, strict=True):
        assert _relative(x.grad, x64.grad) < _BF16_UNIT
    for w, w64 in zip(weights, ws64, strict=True):
        assert _relative(w.grad, w64.grad) < _BF16_UNIT
    assert torch.equal(fused, joint_rms_norm_rope(inputs, weights, freqs, eps=1e-6))
