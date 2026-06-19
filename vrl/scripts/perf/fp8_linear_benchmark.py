"""Is fp8 GEMM actually faster than bf16 at DiT shapes? (the efficiency gate)

WHY: before wiring fp8 into the rollout engine (SPRINT_fp8_rollout_gemm_kernel.md),
confirm the premise — that an fp8 `_scaled_mm` linear, *including* the per-call
dynamic activation quantization overhead, beats a plain bf16 linear at the matmul
shapes a diffusion DiT actually runs. On consumer Blackwell (sm_120) the quant +
amax-reduce can eat the tensor-core win for small/memory-bound GEMMs, so this is
measured, not assumed. If fp8 does not win here, the integration is not worth it.

WHAT: for each (M, K, N) it times, with CUDA events (warmup + median over many
iters): a bf16 `nn.Linear` forward vs an fp8 forward that pre-quantizes the weight
once (inference-realistic) and dynamically quantizes the activation per call, then
`torch._scaled_mm(..., out_dtype=bf16)`. Reports fp8 speedup and the output
relative error (drift) per shape. Shapes cover the four DiT GEMMs (QKV, attn-out,
MLP-up 4x, MLP-down 4x) at SD3.5/cosmos-ish hidden sizes and image/video token
counts.

FAITHFULNESS: the fp8 path is the exact recipe the swap module will use
(per-tensor amax e4m3 scaling, weight pre-quantized, `_scaled_mm` bf16 accumulate),
so the measured latency includes the real overhead the rollout forward would pay.

Usage:  python -m vrl.scripts.perf.fp8_linear_benchmark
"""

from __future__ import annotations

import torch
from torch import nn

FP8_E4M3_MAX = 448.0


def _amax_scale(t: torch.Tensor) -> torch.Tensor:
    return (t.abs().amax() / FP8_E4M3_MAX).clamp_min(1e-12).to(torch.float32)


class Fp8DynamicLinear(nn.Module):
    """nn.Linear replacement: fp8 weight (pre-quantized) + dynamic fp8 activation.

    Per-tensor amax e4m3 scaling, ``_scaled_mm`` with bf16 accumulation. The weight
    is quantized once at construction (inference: weights are frozen during a
    rollout); only the activation is quantized per forward.
    """

    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        w = lin.weight.data  # [out, in]
        w_scale = _amax_scale(w)
        self.register_buffer("w_fp8", (w / w_scale).to(torch.float8_e4m3fn))
        self.register_buffer("w_scale", w_scale)
        self.bias = lin.bias
        self.out_features = lin.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        x_scale = _amax_scale(x2d)
        x_fp8 = (x2d / x_scale).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8, self.w_fp8.t(), scale_a=x_scale, scale_b=self.w_scale, out_dtype=x.dtype,
        )
        out = out.reshape(*shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


def _time_ms(fn, x, iters: int = 100, warmup: int = 25) -> float:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _drift(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-12))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device with fp8 support (Hopper/Blackwell)")
    torch.manual_seed(0)
    dev = "cuda"
    print(f"== fp8 vs bf16 linear on {torch.cuda.get_device_name(0)} ==")
    print("M=tokens*batch, K=in, N=out; fp8 includes per-call activation quant\n")
    print(f"{'shape (M,K,N)':>22} | {'bf16 ms':>8} | {'fp8 ms':>8} | {'speedup':>7} | {'drift':>8}")
    print("-" * 70)

    # (M, K, N): the four DiT GEMMs at a few hidden sizes / token counts.
    hiddens = [1536, 2048, 4096]
    token_counts = [4096, 8192]  # ~1024px image latent / small video
    shapes: list[tuple[int, int, int]] = []
    for h in hiddens:
        for m in token_counts:
            shapes.append((m, h, 3 * h))  # fused QKV
            shapes.append((m, h, h))      # attn out
            shapes.append((m, h, 4 * h))  # MLP up
            shapes.append((m, 4 * h, h))  # MLP down

    speedups, drifts = [], []
    for m, k, n in shapes:
        lin = nn.Linear(k, n, bias=True).cuda().to(torch.bfloat16)
        fp8 = Fp8DynamicLinear(lin).cuda()
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            ref, q = lin(x), fp8(x)
            bf16_ms = _time_ms(lambda t: lin(t), x)
            fp8_ms = _time_ms(lambda t: fp8(t), x)
        sp, dr = bf16_ms / fp8_ms, _drift(q, ref)
        speedups.append(sp)
        drifts.append(dr)
        print(f"{f'({m},{k},{n})':>22} | {bf16_ms:8.4f} | {fp8_ms:8.4f} | {sp:6.2f}x | {dr:8.4f}")

    geomean = float(torch.tensor(speedups).log().mean().exp())
    print("-" * 70)
    print(f"geomean speedup={geomean:.2f}x   mean drift={sum(drifts) / len(drifts):.4f}   "
          f"max drift={max(drifts):.4f}")
    print("\n-- verdict --")
    won = geomean > 1.05
    print(f"  fp8 {'WINS' if won else 'does NOT win'} at these DiT shapes "
          f"(geomean {geomean:.2f}x). "
          + ("Worth wiring." if won else "Not worth wiring as-is."))


if __name__ == "__main__":
    main()
