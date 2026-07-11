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

FAITHFULNESS: the fp8 path IS the production swap module (``Fp8Linear``,
tensorwise recipe: per-tensor amax e4m3 scaling, weight pre-quantized,
`_scaled_mm` bf16 accumulate), so the measured latency includes the real
overhead the rollout forward pays.

Usage:  python -m vrl.scripts.perf.fp8_linear_benchmark
"""

from __future__ import annotations

import torch
from torch import nn

from vrl.nn.quantization.fp8 import Fp8Linear
from vrl.scripts.perf.common.fp8_math import relative_l1_drift
from vrl.scripts.perf.common.timing import cuda_mean_ms


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
        fp8 = Fp8Linear(lin, recipe="tensorwise")
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            ref, q = lin(x), fp8(x)
            bf16_ms = cuda_mean_ms(lambda lin=lin, x=x: lin(x), iters=100, warmup=25)
            fp8_ms = cuda_mean_ms(lambda fp8=fp8, x=x: fp8(x), iters=100, warmup=25)
        sp, dr = bf16_ms / fp8_ms, relative_l1_drift(q, ref)
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
