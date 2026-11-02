"""Fused elementwise kernels of a modulated double-stream DiT block.

Between its GEMMs and its attention, a Qwen-Image block spends a fifth of its
time in memory-bound elementwise chains: an fp32 LayerNorm whose output is
modulated in fp32 and cast back to bf16 at every projection that reads it, an
RMSNorm and a complex RoPE on each of q and k through fp32 temporaries, and
gated residual adds. Each function here does one chain in one pass, in fp32
registers, rounding once to the input dtype on the way out.

The rollout and the trainer run these same kernels, so a rollout and its
replay stay one computation (``trainer.replay_parity``); ``torch.compile``
gives a comparable speedup, but it partitions an inference graph and a
training graph differently, so the two sides round differently. No kernel
uses atomics: every result is a deterministic function of its inputs.

Gradients of the modulation, gate and norm weights (full finetuning; LoRA on
the attention projections needs none) are reduced with torch ops from the
saved inputs, outside the kernels.

Rounding once is also closer to the exact math than the eager chains, which
round the modulation factor ``1 + scale``, the normalized q and k, and the
gated product to bf16 on the way. On CUDA half and fp32 tensors the Triton
kernels run; on any other device or dtype (a float64 reference) the
``*_reference`` functions compute the same math with torch ops.
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
def _layer_norm_modulate_fwd(
    X,
    SCALE,
    SHIFT,
    Y,
    MEAN,
    RSTD,
    rows_per_batch,
    width,
    scale_stride,
    shift_stride,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // rows_per_batch
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    x = tl.load(X + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / width
    centered = tl.where(mask, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / width + eps)
    scale = tl.load(SCALE + batch * scale_stride + cols, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(SHIFT + batch * shift_stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * (1.0 + scale) + shift
    tl.store(Y + row * width + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)


@triton.jit
def _layer_norm_modulate_bwd(
    DY,
    X,
    SCALE,
    MEAN,
    RSTD,
    DX,
    rows_per_batch,
    width,
    scale_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // rows_per_batch
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    dy = tl.load(DY + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    normalized = tl.where(mask, (x - tl.load(MEAN + row)) * rstd, 0.0)
    scale = tl.load(SCALE + batch * scale_stride + cols, mask=mask, other=0.0).to(tl.float32)
    grad = dy * (1.0 + scale)
    projection = tl.sum(grad * normalized, axis=0) / width
    mean_grad = tl.sum(grad, axis=0) / width
    dx = (grad - mean_grad - normalized * projection) * rstd
    tl.store(DX + row * width + cols, dx.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _gated_residual_fwd(
    RESIDUAL,
    GATE,
    Y,
    OUT,
    rows_per_batch,
    width,
    gate_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // rows_per_batch
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    residual = tl.load(RESIDUAL + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + batch * gate_stride + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + row * width + cols, (residual + gate * y).to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _gate_rows(
    DOUT,
    GATE,
    DY,
    rows_per_batch,
    width,
    gate_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // rows_per_batch
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    dout = tl.load(DOUT + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + batch * gate_stride + cols, mask=mask, other=0.0).to(tl.float32)
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


def _rows(x: torch.Tensor, modulation: torch.Tensor) -> tuple[int, int, int]:
    """``(rows, rows per batch, width)`` of a ``[B, L, D]`` input modulated per batch."""

    if x.ndim != 3 or modulation.shape != (x.shape[0], x.shape[2]):
        raise ValueError(
            f"expected x [B, L, D] and a [B, D] modulation, got {tuple(x.shape)} "
            f"and {tuple(modulation.shape)}"
        )
    return x.shape[0] * x.shape[1], x.shape[1], x.shape[2]


def _last_dim_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.stride(-1) == 1 else tensor.contiguous()


class _LayerNormModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, shift, eps):
        x = x.contiguous()
        scale = _last_dim_contiguous(scale)
        shift = _last_dim_contiguous(shift)
        rows, rows_per_batch, width = _rows(x, scale)
        y = torch.empty_like(x)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        block, warps = _row_block(width)
        _layer_norm_modulate_fwd[(rows,)](
            x, scale, shift, y, mean, rstd, rows_per_batch, width,
            scale.stride(0), shift.stride(0), eps, BLOCK=block, num_warps=warps,
        )  # fmt: skip
        ctx.save_for_backward(x, scale, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, scale, mean, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        rows, rows_per_batch, width = _rows(x, scale)
        dx = dscale = dshift = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            block, warps = _row_block(width)
            _layer_norm_modulate_bwd[(rows,)](
                dy, x, scale, mean, rstd, dx, rows_per_batch, width, scale.stride(0),
                BLOCK=block, num_warps=warps,
            )  # fmt: skip
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            dy32 = dy.float()
            if ctx.needs_input_grad[1]:
                normalized = (x.float() - mean.view(x.shape[:2])[..., None]) * rstd.view(
                    x.shape[:2]
                )[..., None]
                dscale = (dy32 * normalized).sum(1).to(scale.dtype)
            if ctx.needs_input_grad[2]:
                dshift = dy32.sum(1).to(scale.dtype)
        return dx, dscale, dshift, None


class _GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, gate, y):
        residual = residual.contiguous()
        y = y.contiguous()
        gate = _last_dim_contiguous(gate)
        rows, rows_per_batch, width = _rows(residual, gate)
        if y.shape != residual.shape:
            raise ValueError(f"residual {tuple(residual.shape)} and y {tuple(y.shape)} differ")
        out = torch.empty_like(residual)
        block, warps = _row_block(width)
        _gated_residual_fwd[(rows,)](
            residual, gate, y, out, rows_per_batch, width, gate.stride(0),
            BLOCK=block, num_warps=warps,
        )  # fmt: skip
        ctx.save_for_backward(gate, y if ctx.needs_input_grad[1] else None)
        ctx.y_dtype = y.dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, y = ctx.saved_tensors
        dout = dout.contiguous()
        dgate = dy = None
        if ctx.needs_input_grad[2]:
            rows, rows_per_batch, width = _rows(dout, gate)
            dy = torch.empty_like(dout, dtype=ctx.y_dtype)
            block, warps = _row_block(width)
            _gate_rows[(rows,)](
                dout, gate, dy, rows_per_batch, width, gate.stride(0),
                BLOCK=block, num_warps=warps,
            )  # fmt: skip
        if ctx.needs_input_grad[1]:
            dgate = (dout.float() * y.float()).sum(1).to(gate.dtype)
        return dout if ctx.needs_input_grad[0] else None, dgate, dy


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


def layer_norm_modulate_reference(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, *, eps: float
) -> torch.Tensor:
    """:func:`layer_norm_modulate` in torch ops, in fp32 or the input's wider dtype."""

    compute = _compute_dtype(x)
    normalized = F.layer_norm(x.to(compute), (x.shape[-1],), eps=eps)
    scale, shift = scale.to(compute)[:, None], shift.to(compute)[:, None]
    return (normalized * (1 + scale) + shift).to(x.dtype)


def layer_norm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, *, eps: float
) -> torch.Tensor:
    """``LayerNorm(x) * (1 + scale) + shift`` with no LayerNorm affine.

    ``x`` is ``[B, L, D]``; ``scale`` and ``shift`` are ``[B, D]`` and apply to
    every token of their batch row. The result has ``x``'s dtype.
    """

    if _runs_triton(x):
        return _LayerNormModulate.apply(x, scale, shift, eps)
    return layer_norm_modulate_reference(x, scale, shift, eps=eps)


def gated_residual_reference(
    residual: torch.Tensor, gate: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """:func:`gated_residual` in torch ops, in fp32 or the input's wider dtype."""

    compute = _compute_dtype(residual)
    out = residual.to(compute) + gate.to(compute)[:, None] * y.to(compute)
    return out.to(residual.dtype)


def gated_residual(residual: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``residual + gate * y`` with a ``[B, D]`` gate over ``[B, L, D]`` tensors."""

    if _runs_triton(residual):
        return _GatedResidual.apply(residual, gate, y)
    return gated_residual_reference(residual, gate, y)


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
]
