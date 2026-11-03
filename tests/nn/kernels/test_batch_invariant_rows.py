"""A sample's rows through the invariant linears and attention do not depend on its batch.

Each case runs samples alone and then batched with others, and requires the
batched rows to be bitwise the lone ones; the shapes are the GEMM kinds
cuBLAS varies on (one conditioning vector per sample, a text stream of tens
of tokens per sample, and a LoRA's thin rank-r side) and the short sequence
flash attention splits by batch. The linears must also be as close to
float64 as cuBLAS's.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from vrl.nn.kernels.batch_invariant import (
    PerSampleLinear,
    install_batch_invariant_linears,
    lora_delta,
    per_sample_attention,
    per_sample_linear,
)

pytestmark = pytest.mark.gpu


def _max_error(actual: torch.Tensor, exact: torch.Tensor) -> float:
    return (actual.double() - exact).abs().max().item()


@pytest.mark.parametrize("tracked", [False, True], ids=["rollout-no-grad", "autograd"])
def test_linear_samples_are_the_same_at_every_batch_size(tracked: bool) -> None:
    torch.manual_seed(0)
    weight = (torch.randn(6144, 3072, device="cuda") / 3072**0.5).bfloat16()
    bias = torch.randn(6144, device="cuda").bfloat16()
    # A text stream: tens of tokens per sample, where cuBLAS varies with the batch.
    tokens = torch.randn(8, 40, 3072, device="cuda").bfloat16().requires_grad_(tracked)

    alone = torch.cat([per_sample_linear(tokens[i : i + 1], weight, bias) for i in range(8)])
    for batch in (2, 3, 4, 8):
        assert torch.equal(per_sample_linear(tokens[:batch], weight, bias), alone[:batch])
    exact = tokens.double() @ weight.double().t() + bias.double()
    assert _max_error(alone, exact) <= 2 * _max_error(F.linear(tokens, weight, bias), exact)


def test_lora_rows_are_the_same_at_every_batch_size() -> None:
    torch.manual_seed(1)
    tokens = torch.randn(4, 2048, 3072, device="cuda").bfloat16()
    base = torch.randn(4, 2048, 3072, device="cuda").bfloat16()
    lora_a = torch.randn(32, 3072, device="cuda") / 3072**0.5
    lora_b = torch.randn(3072, 32, device="cuda") / 32**0.5

    with torch.autocast("cuda", torch.bfloat16):
        alone = torch.cat(
            [
                lora_delta(base[i : i + 1], tokens[i : i + 1], lora_a, lora_b, scaling=2.0)
                for i in range(4)
            ]
        )
        for batch in (2, 4):
            batched = lora_delta(base[:batch], tokens[:batch], lora_a, lora_b, scaling=2.0)
            assert torch.equal(batched, alone[:batch])
        eager = base + F.linear(F.linear(tokens, lora_a), lora_b) * 2.0
    exact = base.double() + 2.0 * (
        tokens.double() @ lora_a.bfloat16().double().t() @ lora_b.bfloat16().double().t()
    )
    assert _max_error(alone, exact) <= _max_error(eager, exact)


def test_attention_samples_are_the_same_at_every_batch_size() -> None:
    torch.manual_seed(3)
    # A 256-pixel FLUX smoke shape: 24 heads over 64 text and 256 image tokens,
    # where flash attention splits the keys by a count that follows the batch.
    query, key, value = (torch.randn(4, 24, 320, 128, device="cuda").bfloat16() for _ in range(3))
    # A padded text prefix per sample: the mask rows split with the samples.
    valid = torch.ones(4, 1, 1, 320, dtype=torch.bool, device="cuda")
    valid[1, ..., 40:64] = False
    for mask in (None, valid):
        alone = torch.cat(
            [
                F.scaled_dot_product_attention(
                    query[i : i + 1],
                    key[i : i + 1],
                    value[i : i + 1],
                    attn_mask=None if mask is None else mask[i : i + 1],
                )
                for i in range(4)
            ]
        )
        for batch in (2, 3, 4):
            batched = per_sample_attention(
                F.scaled_dot_product_attention,
                query[:batch],
                key[:batch],
                value[:batch],
                attn_mask=None if mask is None else mask[:batch],
            )
            assert torch.equal(batched, alone[:batch])


def _relative(actual: torch.Tensor, exact: torch.Tensor) -> float:
    return ((actual.double() - exact.double()).norm() / exact.double().norm()).item()


def _lora_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Linear(256, 64)).cuda()
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["0"]))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    return model


def _forward_backward(model: torch.nn.Module, tokens: torch.Tensor, *, autocast: bool):
    with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
        out = model(tokens)
    out.float().square().sum().backward()
    return out, {n: p.grad for n, p in model.named_parameters() if p.requires_grad}


@pytest.fixture
def ieee_fp32():
    """fp32 GEMMs at IEEE precision, whatever an earlier test left set (TF32 rounds them)."""

    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    matmul.fp32_precision = "ieee"
    yield
    matmul.fp32_precision = previous


@pytest.mark.parametrize(
    "autocast", [False, True], ids=["fp32-per-sample-cublas", "bf16-autocast-triton"]
)
@pytest.mark.usefixtures("ieee_fp32")
def test_installed_lora_linears_compute_peft_lora_and_its_gradients(autocast: bool) -> None:
    stock, invariant = _lora_model(2), _lora_model(2)
    assert install_batch_invariant_linears(invariant) == 5  # 2 bases, 2 adapter halves, 1 LoRA
    assert install_batch_invariant_linears(invariant) == 5
    assert type(invariant.base_model.model[1]) is PerSampleLinear
    tokens = torch.randn(3, 50, 256, device="cuda")
    if autocast:
        tokens = tokens.bfloat16()

    reference, reference_grads = _forward_backward(_lora_model(2), tokens.float(), autocast=False)
    expected, stock_grads = _forward_backward(stock, tokens, autocast=autocast)
    actual, invariant_grads = _forward_backward(invariant, tokens, autocast=autocast)
    assert stock_grads.keys() == invariant_grads.keys() == reference_grads.keys()
    if not autocast:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        for name, grad in stock_grads.items():
            torch.testing.assert_close(invariant_grads[name], grad, rtol=1e-4, atol=1e-5)
        return
    # bf16: the hand-written forward and backward are no further from the fp32
    # model than PEFT's own bf16 adapter.
    assert actual.dtype == expected.dtype
    assert _relative(actual, reference) <= 1.5 * _relative(expected, reference)
    for name, grad in reference_grads.items():
        assert _relative(invariant_grads[name], grad) <= 1.5 * _relative(
            stock_grads[name], grad
        ), name
