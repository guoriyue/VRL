"""Linear layers whose every output row is independent of the batch around it.

A rollout and its replay agree bitwise only when each sample's numbers do not
depend on what else shares its batch. cuBLAS picks its GEMM kernel from the
whole problem shape, so a sample's rows change in their last bits with the
batch around them whenever the per-sample row count is small: measured on the
RTX 5090, a conditioning projection (one vector per sample) and the text
stream of a joint-attention block (tens of tokens per sample) both move by
1e-2 in bf16 between batch 1, 2 and 4, as does a LoRA's thin rank-``r`` side.
Flash attention and the elementwise kernels already work per sample.

Both are fixed without giving up cuBLAS where it is the right kernel:

- every plain linear runs cuBLAS once per sample (:func:`per_sample_linear`):
  each call has the one-sample shape, so cuBLAS picks one kernel whatever the
  batch, at its own accuracy, for a few extra launches;
- a LoRA adapter at a half-precision compute dtype runs the Triton kernel
  here (:func:`lora_delta`), which computes ``x @ W^T`` with one tile
  configuration chosen from ``(N, K)`` alone and an in-order reduction over
  ``K`` inside each program, so an output row is a fixed function of its
  input row. It also fuses the adapter into the base output: the rank-``r``
  hidden stays fp32 and ``base + scaling * h @ B^T`` rounds once, where the
  eager adapter rounds the hidden, the product, the scaled product and the
  sum; measured twice as fast as the eager adapter and closer to float64. At
  an fp32 compute dtype, where cuBLAS's accumulation is the more accurate,
  the adapter runs cuBLAS once per sample instead.

:func:`install_batch_invariant_linears` routes a model's linears and its LoRA
adapters through these, in the rollout and in the replay alike.
Gradients are computed with torch ops (cuBLAS): only the forward has to agree
between the two sides.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_BLOCK_M = 64


@triton.jit(do_not_specialize=["M"])
def _linear_fwd(
    X,
    W,
    BASE,
    OUT,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_basem,
    stride_outm,
    alpha,
    HAS_BASE: tl.constexpr,
    W_AS_X: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # out = [base +] alpha * x @ w^T; one (BLOCK_M, BLOCK_N) tile per
    # program and the K reduction in order, so a row never meets another row.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + depth
        k_mask = k < K
        x = tl.load(
            X + rows[:, None] * stride_xm + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            W + cols[None, :] * stride_wn + k[:, None],
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        if W_AS_X:
            w = w.to(x.dtype)
        else:
            x = x.to(w.dtype)
        acc = tl.dot(x, w, acc, input_precision=INPUT_PRECISION)
    acc = acc * alpha
    mask = row_mask[:, None] & col_mask[None, :]
    if HAS_BASE:
        acc += tl.load(
            BASE + rows[:, None] * stride_basem + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
    tl.store(
        OUT + rows[:, None] * stride_outm + cols[None, :], acc.to(OUT.dtype.element_ty), mask=mask
    )


def _blocks(n: int, k: int, dtype: torch.dtype) -> tuple[int, int]:
    """Tile widths from the weight shape and dtype only, never from the row count."""

    block_k = 32 if dtype == torch.float32 else 64
    return min(max(triton.next_power_of_2(n), 16), 128), min(
        max(triton.next_power_of_2(k), 16), block_k
    )


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    base: torch.Tensor | None = None,
    alpha: float = 1.0,
    weight_as_input: bool = True,
) -> torch.Tensor:
    """``[base +] alpha * x @ weight^T`` over a 2-D ``x``.

    ``weight_as_input`` rounds the weight to ``x``'s dtype before the product
    (the cast autocast applies); otherwise ``x`` is widened to the weight's.
    """

    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty(m, n, device=x.device, dtype=out_dtype)
    compute = x.dtype if weight_as_input else weight.dtype
    block_n, block_k = _blocks(n, k, compute)
    grid = (triton.cdiv(m, _BLOCK_M), triton.cdiv(n, block_n))
    _linear_fwd[grid](
        x, weight, base if base is not None else x, out,
        m, n, k, x.stride(0), weight.stride(0), base.stride(0) if base is not None else 0,
        out.stride(0), alpha,
        HAS_BASE=base is not None, W_AS_X=weight_as_input,
        INPUT_PRECISION="ieee", BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=3,
    )  # fmt: skip
    return out


def _autocast_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype a linear on ``x`` computes in: the autocast dtype when one is on."""

    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return x.dtype


# A 2-D input is one conditioning vector per sample; more rows than this are
# flattened tokens, which a per-row loop would crawl through.
_MAX_VECTOR_ROWS = 256


def per_sample_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``F.linear`` over ``x`` one sample (``x[i]``) at a time.

    Each sample is its own cuBLAS call at the one-sample shape, so its result
    is the same whatever else shares the batch; the numbers are cuBLAS's own.
    A 2-D ``x`` holds one vector per sample. Without autograd (a rollout) the
    calls write straight into one output, the products ``F.linear`` forms
    bitwise; with it they are concatenated.
    """

    if x.ndim < 2 or x.shape[0] <= 1:
        return F.linear(x, weight, bias)
    if x.ndim == 2 and x.shape[0] > _MAX_VECTOR_ROWS:
        raise ValueError(
            f"per_sample_linear: a [{x.shape[0]}, {x.shape[1]}] input looks like flattened "
            "tokens, not one vector per sample; keep the batch axis"
        )
    tracked = torch.is_grad_enabled() and (
        x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    )
    if tracked or not x.is_cuda:
        return torch.cat([F.linear(sample, weight, bias) for sample in x.split(1)])
    compute = _autocast_dtype(x)
    weight_t = weight.to(compute).t()
    bias = None if bias is None else bias.to(compute)
    out = torch.empty(*x.shape[:-1], weight.shape[0], device=x.device, dtype=compute)
    for sample, target in zip(x.to(compute).unbind(0), out.unbind(0), strict=True):
        rows = sample.reshape(-1, sample.shape[-1])
        rows_out = target.view(-1, weight.shape[0])
        if bias is None:
            torch.mm(rows, weight_t, out=rows_out)
        else:
            torch.addmm(bias, rows, weight_t, out=rows_out)
    return out


class _LoraDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, x, lora_a, lora_b, scaling, compute):
        flat = x.reshape(-1, x.shape[-1]).to(compute).contiguous()
        base_flat = base.reshape(-1, base.shape[-1])
        if base_flat.stride(-1) != 1:
            base_flat = base_flat.contiguous()
        hidden = _launch(flat, lora_a.contiguous(), out_dtype=torch.float32)
        # lora_b rounds to the compute dtype, as autocast would round it, and
        # then multiplies the fp32 hidden in fp32: one computation whether a
        # side holds the adapter in fp32 or in that dtype.
        out = _launch(
            hidden,
            lora_b.to(compute).float().contiguous(),
            out_dtype=base.dtype,
            base=base_flat,
            alpha=scaling,
            weight_as_input=False,
        )
        ctx.save_for_backward(flat, hidden, lora_a, lora_b)
        ctx.scaling = scaling
        ctx.compute = compute
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        return out.view(base.shape)

    @staticmethod
    def backward(ctx, dout):
        flat, hidden, lora_a, lora_b = ctx.saved_tensors
        compute = ctx.compute
        dflat = dout.reshape(-1, dout.shape[-1]).to(compute)
        dx = dlora_a = dlora_b = None
        # The eager adapter's backward at its autocast dtypes.
        dhidden = (dflat @ lora_b.to(compute)) * ctx.scaling
        if ctx.needs_input_grad[3]:
            dlora_b = ((dflat.t() @ hidden.to(compute)) * ctx.scaling).to(lora_b.dtype)
        if ctx.needs_input_grad[2]:
            dlora_a = (dhidden.t() @ flat).to(lora_a.dtype)
        if ctx.needs_input_grad[1]:
            dx = (dhidden @ lora_a.to(compute)).to(ctx.x_dtype).view(ctx.x_shape)
        return dout if ctx.needs_input_grad[0] else None, dx, dlora_a, dlora_b, None, None


def lora_delta(
    base: torch.Tensor,
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    """``base + scaling * (x @ lora_a^T) @ lora_b^T`` with rows independent of the batch.

    The adapter computes where PEFT's would: at the CUDA autocast dtype, or
    else at the adapter's dtype (PEFT casts its input to it). At a
    half-precision compute dtype on CUDA, ``x @ lora_a^T`` multiplies at it
    and stays fp32, and the second product runs in fp32 and joins ``base`` in
    the same pass, rounding once. At fp32 it is PEFT's expression, one sample
    (``x[i]``) at a time; off CUDA it is PEFT's expression.
    """

    compute = (
        torch.get_autocast_dtype("cuda")
        if x.is_cuda and torch.is_autocast_enabled("cuda")
        else lora_a.dtype
    )
    if not x.is_cuda or compute not in (torch.bfloat16, torch.float16):
        samples = x.split(1) if x.is_cuda else (x,)
        delta = torch.cat([F.linear(F.linear(s.to(compute), lora_a), lora_b) for s in samples])
        return (base + delta * scaling).to(base.dtype)
    with torch.autocast("cuda", enabled=False):
        return _LoraDelta.apply(base, x, lora_a, lora_b, float(scaling), compute)


class PerSampleLinear(nn.Linear):
    """``nn.Linear`` whose forward is :func:`per_sample_linear`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return per_sample_linear(x, self.weight, self.bias)


def _lora_linear_class() -> type:
    from peft.tuners.lora.layer import Linear as LoraLinear

    class InvariantLoraLinear(LoraLinear):
        """PEFT ``lora.Linear`` whose plain-LoRA path runs :func:`lora_delta`.

        Anything else PEFT's forward handles (disabled or merged adapters,
        mixed adapter batches, variants such as DoRA, dropout, an adapter
        bias, several active adapters) takes PEFT's own forward.
        """

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            active = [name for name in self.active_adapters if name in self.lora_A]
            if (
                args
                or kwargs
                or self.disable_adapters
                or self.merged
                or len(active) != 1
                or active[0] in self.lora_variant
                or not isinstance(self.lora_dropout[active[0]], nn.Identity)
                or self.lora_B[active[0]].bias is not None
            ):
                return super().forward(x, *args, **kwargs)
            name = active[0]
            return lora_delta(
                self.base_layer(x),
                x,
                self.lora_A[name].weight,
                self.lora_B[name].weight,
                scaling=self.scaling[name],
            )

    return InvariantLoraLinear


_INVARIANT_LORA_LINEAR: type | None = None


def install_batch_invariant_linears(root: nn.Module) -> int:
    """Route ``root``'s plain linears and LoRA adapters through the invariant kernels.

    Plain ``nn.Linear`` modules become :class:`PerSampleLinear` and PEFT
    ``lora.Linear`` adapters take
    :func:`lora_delta`; the class swap keeps every parameter and state-dict
    name. Returns how many modules route through the kernels, those already
    swapped included. Install in the rollout and the replay alike.
    """

    global _INVARIANT_LORA_LINEAR
    from peft.tuners.lora.layer import Linear as LoraLinear

    if _INVARIANT_LORA_LINEAR is None:
        _INVARIANT_LORA_LINEAR = _lora_linear_class()
    count = 0
    for module in root.modules():
        kind = type(module)
        if kind in (PerSampleLinear, _INVARIANT_LORA_LINEAR):
            count += 1
        elif kind is nn.Linear:
            module.__class__ = PerSampleLinear
            count += 1
        elif kind is LoraLinear:
            module.__class__ = _INVARIANT_LORA_LINEAR
            count += 1
    return count


__all__ = [
    "PerSampleLinear",
    "install_batch_invariant_linears",
    "lora_delta",
    "per_sample_linear",
]
