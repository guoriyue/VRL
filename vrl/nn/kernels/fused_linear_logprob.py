"""Fused final-linear + log-softmax + gather for AR token replay.

AR replay needs one number per position — ``log p(sampled token)`` — but the
eager path materializes the full logits ``[B, L, V]`` through the vocab head
and keeps them alive for backward (janus: B16 x L576 x V16384 fp32 is ~600 MB
of logits plus the same again in gradient). This module computes the selected
log-probs directly from the head's input, chunking over positions so peak
extra memory is ``chunk_rows x V`` instead of ``B*L*V``:

  forward:  per chunk, logits = hidden @ W.T (+ b) in the model dtype, then a
            Triton pass does the fp32 online logsumexp + target gather; only
            the per-row logsumexp is saved.
  backward: per chunk, logits are RECOMPUTED, a Triton pass overwrites the
            chunk buffer in place with d(logprob)/d(logits) scaled by the
            incoming per-row gradient, and two GEMMs fold that into
            grad_hidden / grad_weight. Nothing of size ``rows x V`` outlives
            its chunk.

The GEMMs stay on cuBLAS (following Liger Kernel's fused_linear_cross_entropy,
the prior art for this shape); Triton owns only the memory-bound
softmax/gather fusion. On CPU (or when Triton is unavailable) a pure-torch
fallback runs the identical chunked math, which also makes fp64 gradcheck
possible.

Numerics contract: matches ``gather_categorical_log_probs`` applied to
``F.linear(hidden, weight, bias)`` — logits are produced in the input dtype,
then normalized in fp32 with the same temperature scaling, so GRPO old/new
ratios are preserved (see the temperature note in ``vrl/math/token/logprob.py``).
One caveat at reduced precision: chunked-GEMM logits can differ from a one-shot
linear by a few input-dtype ulps (cuBLAS tiles the shapes differently), exactly
as an eager microbatch-size change would; the rollout/replay mismatch machinery
already absorbs that class of drift. The Triton normalization itself matches
the torch reference to ~4e-6 on identical logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vrl.math.token.logprob import require_positive_temperature

# Keep chunk buffers ~16 MB fp32 regardless of vocab size.
_CHUNK_ELEMENTS = 4 * 1024 * 1024
_BLOCK_V = 1024


def _chunk_rows_for(vocab_size: int) -> int:
    return max(1, _CHUNK_ELEMENTS // max(vocab_size, 1))


def _use_triton(tensor: torch.Tensor) -> bool:
    # fp64 rows (gradcheck) stay on the torch path: the kernels compute in fp32.
    return tensor.is_cuda and tensor.dtype != torch.float64 and _lse_gather_fwd_kernel is not None


def _norm_dtype(z: torch.Tensor) -> torch.dtype:
    """fp32 normalization like the eager gather path; fp64 passes through."""
    return torch.float64 if z.dtype == torch.float64 else torch.float32


def _fwd_torch(z: torch.Tensor, targets: torch.Tensor, inv_temp: float):
    zf = z.to(_norm_dtype(z)) * inv_temp
    lse = torch.logsumexp(zf, dim=-1)
    picked = zf.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked - lse, lse


def _fwd_triton(z: torch.Tensor, targets: torch.Tensor, inv_temp: float):
    import triton

    rows = z.shape[0]
    out = torch.empty(rows, dtype=torch.float32, device=z.device)
    lse = torch.empty(rows, dtype=torch.float32, device=z.device)
    _lse_gather_fwd_kernel[(rows,)](
        z,
        targets,
        out,
        lse,
        z.shape[1],
        z.stride(0),
        inv_temp,
        BLOCK_V=_BLOCK_V,
        num_warps=8,
    )
    del triton
    return out, lse


def _bwd_torch(
    z: torch.Tensor,
    targets: torch.Tensor,
    lse: torch.Tensor,
    grad_rows: torch.Tensor,
    inv_temp: float,
) -> torch.Tensor:
    """Return d(logprob)/d(z) * grad_rows in z's dtype (new tensor)."""
    p = torch.exp(z.to(_norm_dtype(z)) * inv_temp - lse.unsqueeze(-1))
    onehot = torch.zeros_like(p)
    onehot.scatter_(-1, targets.unsqueeze(-1), 1.0)
    grad = (grad_rows.to(p.dtype) * inv_temp).unsqueeze(-1)
    return ((onehot - p) * grad).to(z.dtype)


def _bwd_triton(
    z: torch.Tensor,
    targets: torch.Tensor,
    lse: torch.Tensor,
    grad_rows: torch.Tensor,
    inv_temp: float,
) -> torch.Tensor:
    import triton

    rows, vocab = z.shape
    _grad_z_bwd_kernel[(rows, triton.cdiv(vocab, _BLOCK_V))](
        z,
        targets,
        lse,
        grad_rows,
        vocab,
        z.stride(0),
        inv_temp,
        BLOCK_V=_BLOCK_V,
    )
    return z  # overwritten in place


class _FusedLinearLogprob(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, D]
        weight: torch.Tensor,  # [V, D]
        bias: torch.Tensor | None,  # [V] or None
        targets: torch.Tensor,  # [N] long
        inv_temp: float,
        chunk_rows: int,
    ) -> torch.Tensor:
        rows = hidden.shape[0]
        norm_dtype = _norm_dtype(hidden)
        out = torch.empty(rows, dtype=norm_dtype, device=hidden.device)
        lse = torch.empty(rows, dtype=norm_dtype, device=hidden.device)
        with torch.no_grad():
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                z = F.linear(hidden[start:end], weight, bias)
                fwd = _fwd_triton if _use_triton(z) else _fwd_torch
                out[start:end], lse[start:end] = fwd(z, targets[start:end], inv_temp)
        ctx.save_for_backward(hidden, weight, bias, targets, lse)
        ctx.inv_temp = inv_temp
        ctx.chunk_rows = chunk_rows
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        hidden, weight, bias, targets, lse = ctx.saved_tensors
        inv_temp: float = ctx.inv_temp
        chunk_rows: int = ctx.chunk_rows
        need_hidden, need_weight, need_bias = (
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            bias is not None and ctx.needs_input_grad[2],
        )
        grad_hidden = torch.empty_like(hidden) if need_hidden else None
        # fp32 accumulators: chunks sum into these many times, and bf16
        # accumulation would drift with the chunk count.
        acc_dtype = _norm_dtype(weight)
        grad_weight = (
            torch.zeros(weight.shape, dtype=acc_dtype, device=weight.device)
            if need_weight
            else None
        )
        grad_bias = (
            torch.zeros(bias.shape, dtype=acc_dtype, device=bias.device) if need_bias else None
        )
        rows = hidden.shape[0]
        with torch.no_grad():
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                h = hidden[start:end]
                z = F.linear(h, weight, bias)
                bwd = _bwd_triton if _use_triton(z) else _bwd_torch
                grad_z = bwd(z, targets[start:end], lse[start:end], grad_out[start:end], inv_temp)
                if need_hidden:
                    torch.mm(grad_z, weight, out=grad_hidden[start:end])
                if grad_weight is not None:
                    grad_weight.addmm_(grad_z.mT.to(acc_dtype), h.to(acc_dtype))
                if grad_bias is not None:
                    grad_bias.add_(grad_z.sum(dim=0).to(acc_dtype))
        return (
            grad_hidden,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None,
            None,
            None,
        )


def fused_linear_logprob(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Log-probs of ``token_ids`` under ``softmax(F.linear(hidden, weight, bias) / T)``.

    ``hidden`` is ``[..., D]`` (the input of the final vocab projection),
    ``weight`` is the projection's ``[V, D]``, ``token_ids`` matches hidden's
    leading shape. Returns fp32 log-probs shaped like ``token_ids``, identical
    (up to fp32 reduction order) to materializing the logits and calling
    ``gather_categorical_log_probs``.
    """
    temp = require_positive_temperature(temperature)
    if hidden.shape[:-1] != token_ids.shape:
        raise ValueError(
            "hidden leading shape must match token_ids shape; "
            f"got hidden={tuple(hidden.shape)} token_ids={tuple(token_ids.shape)}",
        )
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError(
            "weight must be [vocab, hidden_dim] matching hidden's last dim; "
            f"got weight={tuple(weight.shape)} hidden={tuple(hidden.shape)}",
        )
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_ids = token_ids.to(device=hidden.device, dtype=torch.long).reshape(-1)
    rows = chunk_rows if chunk_rows is not None else _chunk_rows_for(weight.shape[0])
    out = _FusedLinearLogprob.apply(flat_hidden, weight, bias, flat_ids, 1.0 / temp, rows)
    return out.reshape(token_ids.shape)


def _define_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _lse_gather_fwd_kernel(
        z_ptr,
        targets_ptr,
        out_ptr,
        lse_ptr,
        V,
        stride_row,
        inv_temp,
        BLOCK_V: tl.constexpr,
    ):
        """Per row: fp32 online logsumexp over V plus the target-logit gather."""
        row = tl.program_id(0)
        base = z_ptr + row * stride_row
        running_max = -float("inf")
        running_sum = 0.0
        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            z = tl.load(base + cols, mask=cols < V, other=-float("inf")).to(tl.float32)
            z = z * inv_temp
            block_max = tl.max(z, axis=0)
            new_max = tl.maximum(running_max, block_max)
            running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
                tl.exp(z - new_max), axis=0
            )
            running_max = new_max
        lse = running_max + tl.log(running_sum)
        target = tl.load(targets_ptr + row)
        z_target = tl.load(base + target).to(tl.float32) * inv_temp
        tl.store(out_ptr + row, z_target - lse)
        tl.store(lse_ptr + row, lse)

    @triton.jit
    def _grad_z_bwd_kernel(
        z_ptr,
        targets_ptr,
        lse_ptr,
        grad_rows_ptr,
        V,
        stride_row,
        inv_temp,
        BLOCK_V: tl.constexpr,
    ):
        """Overwrite z in place with (onehot - softmax) * grad_row * inv_temp."""
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = cols < V
        base = z_ptr + row * stride_row
        z = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        lse = tl.load(lse_ptr + row)
        grad_row = tl.load(grad_rows_ptr + row).to(tl.float32)
        target = tl.load(targets_ptr + row)
        p = tl.exp(z * inv_temp - lse)
        grad = (tl.where(cols == target, 1.0, 0.0) - p) * (grad_row * inv_temp)
        tl.store(base + cols, grad.to(z_ptr.dtype.element_ty), mask=mask)

    return _lse_gather_fwd_kernel, _grad_z_bwd_kernel


try:  # Import-safe on CPU-only environments; callers fall back to torch.
    _lse_gather_fwd_kernel, _grad_z_bwd_kernel = _define_kernels()
except Exception:  # pragma: no cover - exercised only where triton is absent
    _lse_gather_fwd_kernel = _grad_z_bwd_kernel = None  # type: ignore[assignment]


__all__ = ["fused_linear_logprob"]
