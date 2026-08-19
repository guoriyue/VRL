"""Fused linear+log-softmax+gather: parity with the eager path, and gradients.

The acceptance contract (SPRINT_triton_map_and_perf_ledger P4): the fused path
must match ``gather_categorical_log_probs(F.linear(h, W, b), ids)`` in value
and in gradient, on the torch fallback (fp64 gradcheck) and on the Triton
kernels (GPU parity at training dtypes).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vrl.math.token.logprob import gather_categorical_log_probs
from vrl.nn.kernels.fused_linear_logprob import fused_linear_logprob

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _eager(hidden, weight, bias, ids, temperature=1.0):
    return gather_categorical_log_probs(
        F.linear(hidden, weight, bias), ids, temperature=temperature
    )


def _rand_case(batch, length, vocab, dim, *, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(batch, length, dim, generator=g).to(device=device, dtype=dtype)
    weight = torch.randn(vocab, dim, generator=g).to(device=device, dtype=dtype)
    bias = torch.randn(vocab, generator=g).to(device=device, dtype=dtype)
    ids = torch.randint(0, vocab, (batch, length), generator=g).to(device)
    return hidden, weight, bias, ids


class TestTorchFallback:
    def test_matches_eager_gather(self) -> None:
        hidden, weight, bias, ids = _rand_case(2, 5, 33, 8)
        # chunk_rows=3 forces uneven chunk boundaries across the 10 rows.
        out = fused_linear_logprob(hidden, weight, ids, bias=bias, chunk_rows=3)
        torch.testing.assert_close(out, _eager(hidden, weight, bias, ids), rtol=1e-6, atol=1e-6)
        assert out.shape == ids.shape
        assert out.dtype == torch.float32

    def test_temperature_matches_eager(self) -> None:
        hidden, weight, bias, ids = _rand_case(1, 7, 19, 4, seed=1)
        out = fused_linear_logprob(hidden, weight, ids, bias=bias, temperature=0.7, chunk_rows=2)
        torch.testing.assert_close(
            out, _eager(hidden, weight, bias, ids, temperature=0.7), rtol=1e-6, atol=1e-6
        )

    def test_no_bias(self) -> None:
        hidden, weight, _, ids = _rand_case(2, 3, 11, 6, seed=2)
        out = fused_linear_logprob(hidden, weight, ids, chunk_rows=4)
        torch.testing.assert_close(out, _eager(hidden, weight, None, ids), rtol=1e-6, atol=1e-6)

    def test_gradcheck_fp64(self) -> None:
        """Analytic backward (recompute + (onehot - softmax)) against numeric."""
        g = torch.Generator().manual_seed(3)
        hidden = torch.randn(4, 6, generator=g, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(9, 6, generator=g, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(9, generator=g, dtype=torch.float64, requires_grad=True)
        ids = torch.randint(0, 9, (4,), generator=g)

        def fn(h, w, b):
            return fused_linear_logprob(h, w, ids, bias=b, temperature=0.8, chunk_rows=3)

        assert torch.autograd.gradcheck(fn, (hidden, weight, bias), raise_exception=True)

    def test_grads_match_eager_autograd(self) -> None:
        hidden, weight, bias, ids = _rand_case(2, 4, 17, 5, seed=4)
        upstream = torch.randn(2, 4)

        args_fused = [t.clone().requires_grad_(True) for t in (hidden, weight, bias)]
        fused_linear_logprob(
            args_fused[0], args_fused[1], ids, bias=args_fused[2], chunk_rows=3
        ).backward(upstream)

        args_eager = [t.clone().requires_grad_(True) for t in (hidden, weight, bias)]
        _eager(args_eager[0], args_eager[1], args_eager[2], ids).backward(upstream)

        for got, want in zip(args_fused, args_eager, strict=True):
            torch.testing.assert_close(got.grad, want.grad, rtol=1e-5, atol=1e-5)

    def test_shape_mismatch_raises(self) -> None:
        hidden, weight, bias, ids = _rand_case(2, 3, 11, 6)
        with pytest.raises(ValueError, match="leading shape"):
            fused_linear_logprob(hidden, weight, ids[:, :2], bias=bias)
        with pytest.raises(ValueError, match="vocab, hidden_dim"):
            fused_linear_logprob(hidden, weight[:, :3], ids, bias=bias)


@pytest.mark.gpu
class TestTritonKernels:
    """Tiny shapes on purpose: these run alongside live training jobs."""

    @requires_cuda
    def test_forward_matches_eager_fp32(self) -> None:
        # vocab 1021 (prime, > BLOCK boundary at 1024 masks the tail lanes).
        hidden, weight, bias, ids = _rand_case(3, 11, 1021, 32, device="cuda")
        out = fused_linear_logprob(hidden, weight, ids, bias=bias, temperature=0.9, chunk_rows=7)
        eager = _eager(hidden, weight, bias, ids, temperature=0.9)
        torch.testing.assert_close(out, eager, rtol=2e-5, atol=2e-5)

    @requires_cuda
    def test_forward_bf16_no_worse_than_eager(self) -> None:
        """bf16 cannot be compared path-vs-path: cuBLAS picks different tilings
        for the chunked [7, D]x[D, V] GEMMs than for the one-shot 3D linear, and
        the resulting logits already differ by a couple of bf16 ulps (measured
        0.125 on this case). Judge both paths against the fp64 truth instead and
        require the fused path to be no less accurate than eager."""
        hidden, weight, bias, ids = _rand_case(
            3, 11, 1021, 32, device="cuda", dtype=torch.bfloat16
        )
        out = fused_linear_logprob(hidden, weight, ids, bias=bias, temperature=0.9, chunk_rows=7)
        eager = _eager(hidden, weight, bias, ids, temperature=0.9)
        truth = _eager(
            hidden.double(), weight.double(), bias.double(), ids, temperature=0.9
        ).float()
        err_fused = (out - truth).abs().max().item()
        err_eager = (eager - truth).abs().max().item()
        assert err_fused <= 2.0 * err_eager + 1e-2, (err_fused, err_eager)

    @requires_cuda
    def test_grads_match_eager_autograd(self) -> None:
        hidden, weight, bias, ids = _rand_case(2, 9, 1021, 16, device="cuda", seed=5)
        upstream = torch.randn(2, 9, device="cuda")

        args_fused = [t.clone().requires_grad_(True) for t in (hidden, weight, bias)]
        fused_linear_logprob(
            args_fused[0], args_fused[1], ids, bias=args_fused[2], chunk_rows=5
        ).backward(upstream)

        args_eager = [t.clone().requires_grad_(True) for t in (hidden, weight, bias)]
        _eager(args_eager[0], args_eager[1], args_eager[2], ids).backward(upstream)

        for got, want in zip(args_fused, args_eager, strict=True):
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-4, atol=2e-4)

    @requires_cuda
    def test_frozen_weight_only_hidden_grad(self) -> None:
        """The janus shape: LoRA trains the trunk, the vocab head is frozen."""
        hidden, weight, bias, ids = _rand_case(1, 8, 257, 16, device="cuda", seed=6)
        hidden.requires_grad_(True)
        fused_linear_logprob(hidden, weight, ids, bias=bias).sum().backward()
        assert hidden.grad is not None
        assert weight.grad is None and bias.grad is None


class TestReplayContractDispatch:
    def test_segment_fused_payload_matches_logits_payload(self) -> None:
        from vrl.models.interfaces import ReplaySegmentResult

        hidden, weight, bias, ids = _rand_case(2, 4, 23, 6, seed=7)
        logits_form = ReplaySegmentResult(
            segment="image_tokens",
            values={"logits": F.linear(hidden, weight, bias), "token_ids": ids},
        )
        fused_form = ReplaySegmentResult(
            segment="image_tokens",
            values={
                "head_hidden": hidden,
                "head_weight": weight,
                "head_bias": bias,
                "token_ids": ids,
            },
        )
        torch.testing.assert_close(
            fused_form.logprobs(temperature=1.3),
            logits_form.logprobs(temperature=1.3),
            rtol=1e-6,
            atol=1e-6,
        )
