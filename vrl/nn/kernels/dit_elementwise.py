"""Fused elementwise kernels of a modulated DiT block.

Between its GEMMs and its attention, a modulated DiT block (Qwen-Image,
Qwen-Image-2.1, MiniMax-H3) spends a fifth of its time in memory-bound
elementwise chains: a LayerNorm or RMSNorm whose output is modulated through
fp32 temporaries and cast back at every projection that reads it, a q/k
RMSNorm and a RoPE on each of q and k, gated residual adds, and the SwiGLU
product. Each function here does one chain in one pass, in fp32 registers,
rounding once to the input dtype on the way out:

- :func:`layer_norm_modulate`, :func:`rms_norm_modulate`: the norm and
  ``* (1 + scale) + shift``, where every token reads its row of a modulation
  table (per batch item, per packed row as MiniMax-H3's ``adaln_indices``, or
  per sample and position as Qwen-Image-2.1's causal condition);
- :func:`gated_residual`: ``residual + gate * y``, or ``tanh(gate)``;
- :func:`joint_rms_norm_rope`: q/k RMSNorm and the interleaved complex RoPE
  of Qwen-Image, written straight into the joint attention sequence;
- :func:`rms_norm_rotate_half`: q/k RMSNorm and MiniMax-H3's rotate-half RoPE
  of the leading rotary channels;
- :func:`silu_mul`, :func:`swiglu`: ``silu(gate) * up``.

The rollout and the trainer run these same kernels, so a rollout and its
replay stay one computation (``trainer.replay_parity``); ``torch.compile``
gives a comparable speedup, but it partitions an inference graph and a
training graph differently, so the two sides round differently. Every kernel
works on one token row at a time and uses no atomics: a row's result is a
deterministic function of that row, whatever its batch.

Gradients of the modulation tables, gates and norm weights (full finetuning;
LoRA on the attention projections needs none) are reduced with torch ops from
the saved inputs, outside the kernels, and into table rows by masked sums
rather than atomic ``index_add_``.

Rounding once is also closer to the exact math than the eager chains, which
round the modulation factor ``1 + scale``, the normalized q and k, the RoPE
table (MiniMax-H3 rotates in bf16) and the gated product on the way. On CUDA
half and fp32 tensors the Triton kernels run; on any other device or dtype
(a float64 reference) the ``*_reference`` functions compute the same math
with torch ops.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _runs_triton(x: torch.Tensor) -> bool:
    """CUDA tensors at a kernel dtype; float64 (a reference) takes the torch path."""

    return x.is_cuda and x.dtype in (torch.bfloat16, torch.float16, torch.float32)


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.promote_types(x.dtype, torch.float32)


def _row_block(width: int) -> tuple[int, int]:
    """Block width and warp count for one row of ``width`` elements per program."""

    block = triton.next_power_of_2(width)
    return block, min(max(block // 256, 1), 8)


@triton.jit
def _table_row(ROWS, row, rows_per_batch, rows_batch_stride, HAS_ROWS: tl.constexpr):
    # The modulation table row of token ``row``: its batch index, or the
    # entry an index tensor gives its (batch, position).
    batch = row // rows_per_batch
    if HAS_ROWS:
        return tl.load(ROWS + batch * rows_batch_stride + row % rows_per_batch).to(tl.int64)
    return batch


@triton.jit
def _norm_modulate_fwd(
    X,
    WEIGHT,
    SCALE,
    SHIFT,
    ROWS,
    Y,
    MEAN,
    RSTD,
    rows_per_batch,
    rows_batch_stride,
    width,
    scale_stride,
    shift_stride,
    eps,
    RMS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    HAS_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    entry = _table_row(ROWS, row, rows_per_batch, rows_batch_stride, HAS_ROWS)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    x = tl.load(X + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    if RMS:
        mean = 0.0
        centered = x
    else:
        mean = tl.sum(x, axis=0) / width
        centered = tl.where(mask, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / width + eps)
    normalized = centered * rstd
    if HAS_WEIGHT:
        normalized *= tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(SCALE + entry * scale_stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = normalized * (1.0 + scale)
    if HAS_SHIFT:
        y += tl.load(SHIFT + entry * shift_stride + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * width + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)


@triton.jit
def _norm_modulate_bwd(
    DY,
    X,
    WEIGHT,
    SCALE,
    ROWS,
    MEAN,
    RSTD,
    DX,
    rows_per_batch,
    rows_batch_stride,
    width,
    scale_stride,
    RMS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    entry = _table_row(ROWS, row, rows_per_batch, rows_batch_stride, HAS_ROWS)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    dy = tl.load(DY + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    normalized = tl.where(mask, (x - tl.load(MEAN + row)) * rstd, 0.0)
    scale = tl.load(SCALE + entry * scale_stride + cols, mask=mask, other=0.0).to(tl.float32)
    grad = dy * (1.0 + scale)
    if HAS_WEIGHT:
        grad *= tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
    projection = tl.sum(grad * normalized, axis=0) / width
    if RMS:
        dx = (grad - normalized * projection) * rstd
    else:
        dx = (grad - tl.sum(grad, axis=0) / width - normalized * projection) * rstd
    tl.store(DX + row * width + cols, dx.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _gated_residual_fwd(
    RESIDUAL,
    GATE,
    ROWS,
    Y,
    OUT,
    rows_per_batch,
    rows_batch_stride,
    width,
    gate_stride,
    HAS_ROWS: tl.constexpr,
    TANH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    entry = _table_row(ROWS, row, rows_per_batch, rows_batch_stride, HAS_ROWS)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    residual = tl.load(RESIDUAL + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + entry * gate_stride + cols, mask=mask, other=0.0).to(tl.float32)
    if TANH:
        gate = 2.0 * tl.sigmoid(2.0 * gate) - 1.0
    tl.store(OUT + row * width + cols, (residual + gate * y).to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _gate_rows(
    DOUT,
    GATE,
    ROWS,
    DY,
    rows_per_batch,
    rows_batch_stride,
    width,
    gate_stride,
    HAS_ROWS: tl.constexpr,
    TANH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    entry = _table_row(ROWS, row, rows_per_batch, rows_batch_stride, HAS_ROWS)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    dout = tl.load(DOUT + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + entry * gate_stride + cols, mask=mask, other=0.0).to(tl.float32)
    if TANH:
        gate = 2.0 * tl.sigmoid(2.0 * gate) - 1.0
    tl.store(DY + row * width + cols, (dout * gate).to(DY.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_rope_fwd(
    X,
    WEIGHT,
    FREQS,
    OUT,
    rows,
    heads,
    seq,
    out_seq,
    out_offset,
    eps,
    HALF: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # One row is one head vector; its (2j, 2j + 1) pairs are the complex
    # numbers RoPE rotates, FREQS holds (cos, sin) per position and pair.
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < rows
    head = row % heads
    position = (row // heads) % seq
    batch = row // (heads * seq)
    pair = tl.arange(0, HALF)
    mask = row_mask[:, None]
    source = row[:, None] * (2 * HALF) + 2 * pair[None, :]
    even = tl.load(X + source, mask=mask, other=0.0).to(tl.float32)
    odd = tl.load(X + source + 1, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(even * even, axis=1) + tl.sum(odd * odd, axis=1)) / (2 * HALF) + eps)
    real = even * rstd[:, None] * tl.load(WEIGHT + 2 * pair).to(tl.float32)[None, :]
    imag = odd * rstd[:, None] * tl.load(WEIGHT + 2 * pair + 1).to(tl.float32)[None, :]
    angle = position[:, None] * (2 * HALF) + 2 * pair[None, :]
    cos = tl.load(FREQS + angle, mask=mask, other=0.0)
    sin = tl.load(FREQS + angle + 1, mask=mask, other=0.0)
    target = ((batch * out_seq + out_offset + position) * heads + head)[:, None] * (
        2 * HALF
    ) + 2 * pair[None, :]
    tl.store(OUT + target, (real * cos - imag * sin).to(OUT.dtype.element_ty), mask=mask)
    tl.store(OUT + target + 1, (real * sin + imag * cos).to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_rope_bwd(
    DOUT,
    X,
    WEIGHT,
    FREQS,
    DX,
    rows,
    heads,
    seq,
    out_seq,
    out_offset,
    eps,
    HALF: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < rows
    head = row % heads
    position = (row // heads) % seq
    batch = row // (heads * seq)
    pair = tl.arange(0, HALF)
    mask = row_mask[:, None]
    target = ((batch * out_seq + out_offset + position) * heads + head)[:, None] * (
        2 * HALF
    ) + 2 * pair[None, :]
    dreal_out = tl.load(DOUT + target, mask=mask, other=0.0).to(tl.float32)
    dimag_out = tl.load(DOUT + target + 1, mask=mask, other=0.0).to(tl.float32)
    angle = position[:, None] * (2 * HALF) + 2 * pair[None, :]
    cos = tl.load(FREQS + angle, mask=mask, other=0.0)
    sin = tl.load(FREQS + angle + 1, mask=mask, other=0.0)
    # The rotation's transpose, then the weight: the gradient at x * rstd.
    deven = (dreal_out * cos + dimag_out * sin) * tl.load(WEIGHT + 2 * pair).to(tl.float32)[
        None, :
    ]
    dodd = (dimag_out * cos - dreal_out * sin) * tl.load(WEIGHT + 2 * pair + 1).to(tl.float32)[
        None, :
    ]
    source = row[:, None] * (2 * HALF) + 2 * pair[None, :]
    even = tl.load(X + source, mask=mask, other=0.0).to(tl.float32)
    odd = tl.load(X + source + 1, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(even * even, axis=1) + tl.sum(odd * odd, axis=1)) / (2 * HALF) + eps)
    projection = (tl.sum(deven * even, axis=1) + tl.sum(dodd * odd, axis=1)) / (2 * HALF)
    correction = (rstd * rstd * rstd * projection)[:, None]
    tl.store(
        DX + source, (rstd[:, None] * deven - even * correction).to(DX.dtype.element_ty), mask=mask
    )
    tl.store(
        DX + source + 1,
        (rstd[:, None] * dodd - odd * correction).to(DX.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _rms_norm_rotate_half_fwd(
    X,
    WEIGHT,
    COS,
    SIN,
    OUT,
    rows,
    heads,
    seq,
    rotary,
    eps,
    DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # One row is one head vector: RMSNorm over all DIM channels, then the
    # rotate-half RoPE on the leading ``rotary`` channels (channel j pairs
    # with j +- rotary/2); the rest pass through normalized.
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    position = (row // heads) % seq
    cols = tl.arange(0, DIM)
    half = rotary // 2
    partner = tl.where(cols < half, cols + half, tl.where(cols < rotary, cols - half, cols))
    sign = tl.where(cols < half, -1.0, 1.0)
    mask = (row < rows)[:, None]
    x = tl.load(X + row[:, None] * DIM + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(X + row[:, None] * DIM + partner[None, :], mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / DIM + eps)[:, None]
    normalized = x * rstd * tl.load(WEIGHT + cols).to(tl.float32)[None, :]
    partnered = xp * rstd * tl.load(WEIGHT + partner).to(tl.float32)[None, :]
    angle = position[:, None] * rotary + cols[None, :]
    rotating = mask & (cols < rotary)[None, :]
    cos = tl.load(COS + angle, mask=rotating, other=1.0)
    sin = tl.load(SIN + angle, mask=rotating, other=0.0)
    out = normalized * cos + sign[None, :] * partnered * sin
    tl.store(OUT + row[:, None] * DIM + cols[None, :], out.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_rotate_half_bwd(
    DOUT,
    X,
    WEIGHT,
    COS,
    SIN,
    DX,
    rows,
    heads,
    seq,
    rotary,
    eps,
    DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    position = (row // heads) % seq
    cols = tl.arange(0, DIM)
    half = rotary // 2
    partner = tl.where(cols < half, cols + half, tl.where(cols < rotary, cols - half, cols))
    sign = tl.where(cols < half, -1.0, 1.0)
    mask = (row < rows)[:, None]
    dout = tl.load(DOUT + row[:, None] * DIM + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    doutp = tl.load(DOUT + row[:, None] * DIM + partner[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    rotating = mask & (cols < rotary)[None, :]
    cos = tl.load(COS + position[:, None] * rotary + cols[None, :], mask=rotating, other=1.0)
    sinp = tl.load(SIN + position[:, None] * rotary + partner[None, :], mask=rotating, other=0.0)
    # The rotation's transpose (the partner's sign is minus this channel's),
    # then the weight: the gradient at x * rstd.
    grad = (dout * cos - sign[None, :] * doutp * sinp) * tl.load(WEIGHT + cols).to(tl.float32)[
        None, :
    ]
    x = tl.load(X + row[:, None] * DIM + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / DIM + eps)[:, None]
    projection = (tl.sum(grad * x, axis=1) / DIM)[:, None]
    dx = rstd * grad - x * (rstd * rstd * rstd * projection)
    tl.store(DX + row[:, None] * DIM + cols[None, :], dx.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _silu_mul_fwd(
    GATE,
    UP,
    OUT,
    width,
    gate_stride,
    up_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    gate = tl.load(GATE + row * gate_stride + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + row * up_stride + cols, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(OUT + row * width + cols, out.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _silu_mul_bwd(
    DOUT,
    GATE,
    UP,
    DGATE,
    DUP,
    width,
    gate_stride,
    up_stride,
    dgate_stride,
    dup_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    dout = tl.load(DOUT + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + row * gate_stride + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + row * up_stride + cols, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    dgate = dout * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    tl.store(DGATE + row * dgate_stride + cols, dgate.to(DGATE.dtype.element_ty), mask=mask)
    tl.store(
        DUP + row * dup_stride + cols, (dout * gate * sigmoid).to(DUP.dtype.element_ty), mask=mask
    )


def _table_rows(
    x: torch.Tensor, table: torch.Tensor, rows: torch.Tensor | None
) -> tuple[torch.Tensor | None, int]:
    """Check a ``[R, D]`` modulation table against ``x``; ``(rows, batch stride of rows)``.

    ``rows`` is ``None`` (token ``(b, l)`` reads table row ``b``, so ``R`` is
    the batch), an ``[L]`` index shared by the batch, or a ``[B, L]`` index.
    """

    if x.ndim != 3 or table.ndim != 2 or table.shape[1] != x.shape[2]:
        raise ValueError(
            f"expected x [B, L, D] and a [R, D] modulation table, got {tuple(x.shape)} "
            f"and {tuple(table.shape)}"
        )
    batch, length = x.shape[:2]
    if rows is None:
        if table.shape[0] != batch:
            raise ValueError(
                f"a per-batch modulation needs {batch} table rows, got {table.shape[0]}"
            )
        return None, 0
    if rows.dtype.is_floating_point or rows.is_complex():
        raise TypeError(f"modulation rows must be integers, got {rows.dtype}")
    if tuple(rows.shape) == (length,):
        return rows.contiguous(), 0
    if tuple(rows.shape) == (batch, length):
        return rows.contiguous(), length
    raise ValueError(f"modulation rows must be [L] or [B, L], got {tuple(rows.shape)}")


def _gather_table(table: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
    """The ``[B or 1, L or 1, D]`` per-token view of a modulation table."""

    if rows is None:
        return table[:, None]
    return table[rows.long()] if rows.ndim == 2 else table[rows.long()][None]


def _reduce_to_table(
    per_token: torch.Tensor, rows: torch.Tensor | None, entries: int
) -> torch.Tensor:
    """Sum ``[B, L, D]`` per-token gradients into their table rows, deterministically.

    One masked fp32 sum per table row (a table has a handful): no atomic
    ``index_add_``, and no GEMM that ``float32_precision: tf32`` would round.
    """

    if rows is None:
        return per_token.sum(1)
    index = rows.expand(per_token.shape[:2])
    return torch.stack(
        [
            torch.where((index == entry)[..., None], per_token, 0).sum((0, 1))
            for entry in range(entries)
        ]
    )


def _last_dim_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.stride(-1) == 1 else tensor.contiguous()


class _NormModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, shift, rows, eps, rms):
        x = x.contiguous()
        scale = _last_dim_contiguous(scale)
        shift = None if shift is None else _last_dim_contiguous(shift)
        weight = None if weight is None else weight.contiguous()
        rows, rows_batch_stride = _table_rows(x, scale, rows)
        if shift is not None and shift.shape != scale.shape:
            raise ValueError(f"shift {tuple(shift.shape)} and scale {tuple(scale.shape)} differ")
        count, width = x.shape[0] * x.shape[1], x.shape[2]
        y = torch.empty_like(x)
        mean = torch.empty(count, device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        block, warps = _row_block(width)
        _norm_modulate_fwd[(count,)](
            x, weight if weight is not None else x, scale, shift if shift is not None else scale,
            rows if rows is not None else mean, y, mean, rstd, x.shape[1], rows_batch_stride,
            width, scale.stride(0), shift.stride(0) if shift is not None else 0, eps,
            RMS=rms, HAS_WEIGHT=weight is not None, HAS_SHIFT=shift is not None,
            HAS_ROWS=rows is not None, BLOCK=block, num_warps=warps,
        )  # fmt: skip
        ctx.save_for_backward(x, weight, scale, rows, mean, rstd)
        ctx.rows_batch_stride = rows_batch_stride
        ctx.rms = rms
        ctx.has_shift = shift is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, scale, rows, mean, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        count, width = x.shape[0] * x.shape[1], x.shape[2]
        dx = dweight = dscale = dshift = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            block, warps = _row_block(width)
            _norm_modulate_bwd[(count,)](
                dy, x, weight if weight is not None else x, scale,
                rows if rows is not None else mean, mean, rstd, dx, x.shape[1],
                ctx.rows_batch_stride, width, scale.stride(0), RMS=ctx.rms,
                HAS_WEIGHT=weight is not None, HAS_ROWS=rows is not None,
                BLOCK=block, num_warps=warps,
            )  # fmt: skip
        if any(ctx.needs_input_grad[1:4]):
            dy32 = dy.float()
            normalized = (x.float() - mean.view(x.shape[:2])[..., None]) * rstd.view(x.shape[:2])[
                ..., None
            ]
            if ctx.needs_input_grad[1]:
                factor = 1 + _gather_table(scale.float(), rows)
                dweight = (dy32 * factor * normalized).sum((0, 1)).to(weight.dtype)
            if ctx.needs_input_grad[2]:
                weighted = normalized if weight is None else normalized * weight.float()
                dscale = _reduce_to_table(dy32 * weighted, rows, scale.shape[0]).to(scale.dtype)
            if ctx.has_shift and ctx.needs_input_grad[3]:
                dshift = _reduce_to_table(dy32, rows, scale.shape[0]).to(scale.dtype)
        return dx, dweight, dscale, dshift, None, None, None


class _GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, gate, y, rows, tanh):
        residual = residual.contiguous()
        y = y.contiguous()
        gate = _last_dim_contiguous(gate)
        rows, rows_batch_stride = _table_rows(residual, gate, rows)
        if y.shape != residual.shape:
            raise ValueError(f"residual {tuple(residual.shape)} and y {tuple(y.shape)} differ")
        count, width = residual.shape[0] * residual.shape[1], residual.shape[2]
        out = torch.empty_like(residual)
        block, warps = _row_block(width)
        _gated_residual_fwd[(count,)](
            residual, gate, rows if rows is not None else gate, y, out, residual.shape[1],
            rows_batch_stride, width, gate.stride(0), HAS_ROWS=rows is not None, TANH=tanh,
            BLOCK=block, num_warps=warps,
        )  # fmt: skip
        ctx.save_for_backward(gate, rows, y if ctx.needs_input_grad[1] else None)
        ctx.rows_batch_stride = rows_batch_stride
        ctx.tanh = tanh
        ctx.y_dtype = y.dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, rows, y = ctx.saved_tensors
        dout = dout.contiguous()
        dgate = dy = None
        if ctx.needs_input_grad[2]:
            count, width = dout.shape[0] * dout.shape[1], dout.shape[2]
            dy = torch.empty_like(dout, dtype=ctx.y_dtype)
            block, warps = _row_block(width)
            _gate_rows[(count,)](
                dout, gate, rows if rows is not None else gate, dy, dout.shape[1],
                ctx.rows_batch_stride, width, gate.stride(0), HAS_ROWS=rows is not None,
                TANH=ctx.tanh, BLOCK=block, num_warps=warps,
            )  # fmt: skip
        if ctx.needs_input_grad[1]:
            per_token = dout.float() * y.float()
            if ctx.tanh:
                per_token = per_token * (1 - torch.tanh(_gather_table(gate.float(), rows)) ** 2)
            dgate = _reduce_to_table(per_token, rows, gate.shape[0]).to(gate.dtype)
        return dout if ctx.needs_input_grad[0] else None, dgate, dy, None, None


_ROPE_BLOCK_ROWS = 32


def _rope_table(freqs: torch.Tensor) -> torch.Tensor:
    """``[S, D/2]`` complex rotations as a contiguous ``[S, D/2, 2]`` (cos, sin) fp32 table."""

    if not freqs.is_complex():
        raise TypeError(f"expected complex RoPE frequencies, got {freqs.dtype}")
    return torch.view_as_real(freqs.to(torch.complex64).contiguous())


class _JointRmsNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, count, *tensors):
        inputs = [x.contiguous() for x in tensors[:count]]
        weights = [w.contiguous() for w in tensors[count : 2 * count]]
        tables = [_rope_table(f) for f in tensors[2 * count :]]
        batch, _, heads, dim = inputs[0].shape
        lengths = [x.shape[1] for x in inputs]
        out = inputs[0].new_empty(batch, sum(lengths), heads, dim)
        offset = 0
        for x, weight, table in zip(inputs, weights, tables, strict=True):
            rows = x.shape[0] * x.shape[1] * heads
            _rms_norm_rope_fwd[(triton.cdiv(rows, _ROPE_BLOCK_ROWS),)](
                x, weight, table, out, rows, heads, x.shape[1], out.shape[1], offset, eps,
                HALF=dim // 2, BLOCK_ROWS=_ROPE_BLOCK_ROWS,
            )  # fmt: skip
            offset += x.shape[1]
        ctx.save_for_backward(*inputs, *weights, *tables)
        ctx.eps = eps
        ctx.count = count
        return out

    @staticmethod
    def backward(ctx, dout):
        count = ctx.count
        saved = ctx.saved_tensors
        inputs, weights, tables = saved[:count], saved[count : 2 * count], saved[2 * count :]
        dout = dout.contiguous()
        heads, dim = dout.shape[2], dout.shape[3]
        dinputs: list[torch.Tensor | None] = []
        dweights: list[torch.Tensor | None] = []
        offset = 0
        for index, (x, weight, table) in enumerate(zip(inputs, weights, tables, strict=True)):
            dx = None
            if ctx.needs_input_grad[2 + index]:
                dx = torch.empty_like(x)
                rows = x.shape[0] * x.shape[1] * heads
                _rms_norm_rope_bwd[(triton.cdiv(rows, _ROPE_BLOCK_ROWS),)](
                    dout, x, weight, table, dx, rows, heads, x.shape[1], dout.shape[1],
                    offset, ctx.eps, HALF=dim // 2, BLOCK_ROWS=_ROPE_BLOCK_ROWS,
                )  # fmt: skip
            dinputs.append(dx)
            dweight = None
            if ctx.needs_input_grad[2 + count + index]:
                part = dout[:, offset : offset + x.shape[1]].float().unflatten(-1, (-1, 2))
                cos, sin = table[:, None, :, 0], table[:, None, :, 1]
                rotated = torch.stack(
                    [
                        part[..., 0] * cos + part[..., 1] * sin,
                        part[..., 1] * cos - part[..., 0] * sin,
                    ],
                    dim=-1,
                ).flatten(-2)
                x32 = x.float()
                normalized = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + ctx.eps)
                dweight = (rotated * normalized).sum((0, 1, 2)).to(weight.dtype)
            dweights.append(dweight)
            offset += x.shape[1]
        return None, None, *dinputs, *dweights, *([None] * count)


class _RmsNormRotateHalf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cos, sin, eps):
        x = x.contiguous()
        weight = weight.contiguous()
        cos = cos.float().contiguous()
        sin = sin.float().contiguous()
        out = torch.empty_like(x)
        rows = x.shape[0] * x.shape[1] * x.shape[2]
        _rms_norm_rotate_half_fwd[(triton.cdiv(rows, _ROPE_BLOCK_ROWS),)](
            x, weight, cos, sin, out, rows, x.shape[2], x.shape[1], cos.shape[1], eps,
            DIM=x.shape[3], BLOCK_ROWS=_ROPE_BLOCK_ROWS,
        )  # fmt: skip
        ctx.save_for_backward(x, weight, cos, sin)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, cos, sin = ctx.saved_tensors
        dout = dout.contiguous()
        dx = dweight = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            rows = x.shape[0] * x.shape[1] * x.shape[2]
            _rms_norm_rotate_half_bwd[(triton.cdiv(rows, _ROPE_BLOCK_ROWS),)](
                dout, x, weight, cos, sin, dx, rows, x.shape[2], x.shape[1], cos.shape[1],
                ctx.eps, DIM=x.shape[3], BLOCK_ROWS=_ROPE_BLOCK_ROWS,
            )  # fmt: skip
        if ctx.needs_input_grad[1]:
            rotary = cos.shape[1]
            grad = dout.float()
            first, second = grad[..., : rotary // 2], grad[..., rotary // 2 : rotary]
            c, s = cos[None, :, None], sin[None, :, None]
            # The rotation's transpose on the rotated channels.
            unrotated = torch.cat(
                [
                    first * c[..., : rotary // 2] + second * s[..., rotary // 2 :],
                    second * c[..., rotary // 2 :] - first * s[..., : rotary // 2],
                    grad[..., rotary:],
                ],
                dim=-1,
            )
            x32 = x.float()
            normalized = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + ctx.eps)
            dweight = (unrotated * normalized).sum((0, 1, 2)).to(weight.dtype)
        return dx, dweight, None, None, None


_SILU_BLOCK = 1024


def _silu_mul_launch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    width = gate.shape[-1]
    gate2, up2 = gate.reshape(-1, width), up.reshape(-1, width)
    out = torch.empty(gate2.shape, device=gate.device, dtype=gate.dtype)
    _silu_mul_fwd[(gate2.shape[0], triton.cdiv(width, _SILU_BLOCK))](
        gate2, up2, out, width, gate2.stride(0), up2.stride(0), BLOCK=_SILU_BLOCK, num_warps=4
    )
    return out.view(gate.shape)


def _silu_mul_grads(
    dout: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    dgate: torch.Tensor,
    dup: torch.Tensor,
) -> None:
    width = gate.shape[-1]
    gate2, up2 = gate.reshape(-1, width), up.reshape(-1, width)
    dgate2, dup2 = dgate.view(-1, *dgate.shape[-1:]), dup.view(-1, *dup.shape[-1:])
    _silu_mul_bwd[(gate2.shape[0], triton.cdiv(width, _SILU_BLOCK))](
        dout.contiguous().view(-1, width), gate2, up2, dgate2, dup2, width, gate2.stride(0),
        up2.stride(0), dgate2.stride(0), dup2.stride(0), BLOCK=_SILU_BLOCK, num_warps=4,
    )  # fmt: skip


def _rows_strided(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` with a contiguous last dim whose leading dims flatten into one row stride."""

    tensor = _last_dim_contiguous(tensor)
    try:
        tensor.view(-1, tensor.shape[-1])
    except RuntimeError:
        tensor = tensor.contiguous()
    return tensor


class _SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if gate.shape != up.shape:
            raise ValueError(f"gate {tuple(gate.shape)} and up {tuple(up.shape)} differ")
        gate, up = _rows_strided(gate), _rows_strided(up)
        ctx.save_for_backward(gate, up)
        return _silu_mul_launch(gate, up)

    @staticmethod
    def backward(ctx, dout):
        gate, up = ctx.saved_tensors
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        _silu_mul_grads(dout, gate, up, dgate, dup)
        return dgate, dup


class _PackedSwiGlu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected):
        projected = projected.contiguous()
        up, gate = projected.chunk(2, dim=-1)
        ctx.save_for_backward(projected)
        return _silu_mul_launch(gate, up)

    @staticmethod
    def backward(ctx, dout):
        (projected,) = ctx.saved_tensors
        up, gate = projected.chunk(2, dim=-1)
        dprojected = torch.empty_like(projected)
        dup, dgate = dprojected.chunk(2, dim=-1)
        _silu_mul_grads(dout, gate, up, dgate, dup)
        return dprojected


def _norm_modulate_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor | None,
    rows: torch.Tensor | None,
    *,
    eps: float,
    rms: bool,
) -> torch.Tensor:
    compute = _compute_dtype(x)
    wide = x.to(compute)
    if rms:
        normalized = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + eps)
    else:
        normalized = F.layer_norm(wide, (x.shape[-1],), eps=eps)
    if weight is not None:
        normalized = normalized * weight.to(compute)
    out = normalized * (1 + _gather_table(scale.to(compute), rows))
    if shift is not None:
        out = out + _gather_table(shift.to(compute), rows)
    return out.to(x.dtype)


def layer_norm_modulate_reference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor | None,
    *,
    eps: float,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`layer_norm_modulate` in torch ops, in fp32 or the input's wider dtype."""

    return _norm_modulate_reference(x, None, scale, shift, rows, eps=eps, rms=False)


def layer_norm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor | None,
    *,
    eps: float,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """``LayerNorm(x) * (1 + scale) + shift`` with no LayerNorm affine.

    ``x`` is ``[B, L, D]``; ``scale`` and ``shift`` (``None``: no shift) are
    ``[R, D]`` modulation tables. Token ``(b, l)`` reads table row ``b``
    without ``rows`` (``R`` is the batch), else row ``rows[l]`` (an ``[L]``
    index) or ``rows[b, l]`` (``[B, L]``). The result has ``x``'s dtype.
    """

    if _runs_triton(x):
        return _NormModulate.apply(x, None, scale, shift, rows, eps, False)
    return layer_norm_modulate_reference(x, scale, shift, eps=eps, rows=rows)


def rms_norm_modulate_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor | None,
    *,
    eps: float,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`rms_norm_modulate` in torch ops, in fp32 or the input's wider dtype."""

    return _norm_modulate_reference(x, weight, scale, shift, rows, eps=eps, rms=True)


def rms_norm_modulate(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor | None,
    *,
    eps: float,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """``RMSNorm(x) * weight * (1 + scale) + shift``, tables as :func:`layer_norm_modulate`."""

    if _runs_triton(x):
        return _NormModulate.apply(x, weight, scale, shift, rows, eps, True)
    return rms_norm_modulate_reference(x, weight, scale, shift, eps=eps, rows=rows)


def gated_residual_reference(
    residual: torch.Tensor,
    gate: torch.Tensor,
    y: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
    tanh: bool = False,
) -> torch.Tensor:
    """:func:`gated_residual` in torch ops, in fp32 or the input's wider dtype."""

    compute = _compute_dtype(residual)
    gate = _gather_table(gate.to(compute), rows)
    if tanh:
        gate = gate.tanh()
    return (residual.to(compute) + gate * y.to(compute)).to(residual.dtype)


def gated_residual(
    residual: torch.Tensor,
    gate: torch.Tensor,
    y: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
    tanh: bool = False,
) -> torch.Tensor:
    """``residual + gate * y`` (``tanh(gate)`` with ``tanh``) over ``[B, L, D]`` tensors.

    ``gate`` is an ``[R, D]`` table addressed per token as in
    :func:`layer_norm_modulate`.
    """

    if _runs_triton(residual):
        return _GatedResidual.apply(residual, gate, y, rows, tanh)
    return gated_residual_reference(residual, gate, y, rows=rows, tanh=tanh)


def rms_norm_rotate_half_reference(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, eps: float
) -> torch.Tensor:
    """:func:`rms_norm_rotate_half` in torch ops, in fp32 or the input's wider dtype."""

    compute = _compute_dtype(x)
    wide = x.to(compute)
    normalized = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + eps) * weight.to(compute)
    rotary = cos.shape[-1]
    rotating, passing = normalized[..., :rotary], normalized[..., rotary:]
    first, second = rotating.chunk(2, dim=-1)
    cos, sin = cos.to(compute)[None, :, None], sin.to(compute)[None, :, None]
    rotated = rotating * cos + torch.cat([-second, first], dim=-1) * sin
    return torch.cat([rotated, passing], dim=-1).to(x.dtype)


def rms_norm_rotate_half(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, eps: float
) -> torch.Tensor:
    """RMSNorm each ``[B, S, H, D]`` head vector with ``weight``, then rotate-half RoPE.

    ``cos`` and ``sin`` are ``[S, R]`` (``R <= D``, even): the leading ``R``
    channels rotate in the ``rotate_half`` convention (channel ``j`` pairs
    with ``j + R/2``), the remaining ``D - R`` pass through normalized.
    """

    if x.ndim != 4 or cos.shape != (x.shape[1], cos.shape[-1]) or cos.shape != sin.shape:
        raise ValueError(
            f"input {tuple(x.shape)} needs [S, R] cos and sin, got {tuple(cos.shape)} "
            f"and {tuple(sin.shape)}"
        )
    if cos.shape[1] % 2 or cos.shape[1] > x.shape[3]:
        raise ValueError(f"rotary width {cos.shape[1]} must be even and at most {x.shape[3]}")
    if _runs_triton(x):
        if x.shape[3] != triton.next_power_of_2(x.shape[3]):
            raise ValueError(
                f"the fused RoPE kernel needs a power-of-two head_dim, got {x.shape[3]}"
            )
        return _RmsNormRotateHalf.apply(x, weight, cos, sin, eps)
    return rms_norm_rotate_half_reference(x, weight, cos, sin, eps=eps)


def silu_mul_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """:func:`silu_mul` in torch ops, in fp32 or the input's wider dtype."""

    compute = _compute_dtype(gate)
    return (F.silu(gate.to(compute)) * up.to(compute)).to(gate.dtype)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` in one pass; either may be a view with strided rows."""

    if _runs_triton(gate):
        return _SiluMul.apply(gate, up)
    return silu_mul_reference(gate, up)


def swiglu(projected: torch.Tensor) -> torch.Tensor:
    """diffusers' ``SwiGLU`` after its projection: ``up * silu(gate)`` of ``projected.chunk(2)``."""

    if _runs_triton(projected):
        return _PackedSwiGlu.apply(projected)
    up, gate = projected.chunk(2, dim=-1)
    return silu_mul_reference(gate, up)


def joint_rms_norm_rope_reference(
    inputs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    freqs: Sequence[torch.Tensor],
    *,
    eps: float,
) -> torch.Tensor:
    """:func:`joint_rms_norm_rope` in torch ops, in fp32 or the input's wider dtype."""

    parts = []
    for x, weight, rotation in zip(inputs, weights, freqs, strict=True):
        compute = _compute_dtype(x)
        wide = x.to(compute)
        normalized = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + eps)
        normalized = normalized * weight.to(compute)
        rotated = torch.view_as_complex(normalized.unflatten(-1, (-1, 2)).contiguous())
        rotated = rotated * rotation.to(compute.to_complex())[None, :, None, :]
        parts.append(torch.view_as_real(rotated).flatten(-2).to(x.dtype))
    return torch.cat(parts, dim=1)


def joint_rms_norm_rope(
    inputs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    freqs: Sequence[torch.Tensor],
    *,
    eps: float,
) -> torch.Tensor:
    """RMSNorm each ``[B, S_i, H, D]`` input, rotate it, and concatenate along S.

    Input ``i`` is normalized over its head dim with ``weights[i]`` and
    rotated by ``freqs[i]`` (``[S_i, D/2]`` complex, the interleaved-pair
    convention of diffusers' ``apply_rotary_emb_qwen(use_real=False)``); the
    parts are written straight into one ``[B, sum S_i, H, D]`` tensor, the
    joint sequence attention reads.
    """

    if len(inputs) != len(weights) or len(inputs) != len(freqs):
        raise ValueError("joint_rms_norm_rope takes one weight and one rotation per input")
    for x, rotation in zip(inputs, freqs, strict=True):
        if x.ndim != 4 or rotation.shape != (x.shape[1], x.shape[3] // 2):
            raise ValueError(
                f"input {tuple(x.shape)} needs [S, D/2] rotations, got {tuple(rotation.shape)}"
            )
    if _runs_triton(inputs[0]) and inputs[0].shape[3] // 2 != triton.next_power_of_2(
        inputs[0].shape[3] // 2
    ):
        raise ValueError(
            f"the fused RoPE kernel needs a power-of-two head_dim / 2, got {inputs[0].shape[3]}"
        )
    if _runs_triton(inputs[0]):
        return _JointRmsNormRope.apply(eps, len(inputs), *inputs, *weights, *freqs)
    return joint_rms_norm_rope_reference(inputs, weights, freqs, eps=eps)


__all__ = [
    "gated_residual",
    "gated_residual_reference",
    "joint_rms_norm_rope",
    "joint_rms_norm_rope_reference",
    "layer_norm_modulate",
    "layer_norm_modulate_reference",
    "rms_norm_modulate",
    "rms_norm_modulate_reference",
    "rms_norm_rotate_half",
    "rms_norm_rotate_half_reference",
    "silu_mul",
    "silu_mul_reference",
    "swiglu",
]
