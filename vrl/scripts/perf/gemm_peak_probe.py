"""Measure the ACHIEVABLE dense bf16/fp16 GEMM throughput of the current GPU.

WHY: every MFU number is `achieved_TFLOPS / peak_TFLOPS`, so a wrong peak silently
mis-diagnoses everything. The vendor "TFLOPS"/"AI TOPS" headline is usually the
fp8/fp4 (often sparse) number, NOT the bf16-dense-with-fp32-accumulate rate that a
training/inference DiT actually runs at — and consumer GeForce cards (RTX 40/50) run
bf16 tensor ops with fp32 accumulate at HALF the fp16-accumulate rate. So the real
peak must be MEASURED per machine, not hardcoded.

Concrete example this catches: an RTX 5090's headline is ~419 bf16 / ~3352 fp4-sparse
"AI TOPS", but the achievable bf16-dense (fp32 accumulate) GEMM is ~232 TFLOPS. Using
419 made a saturated (~94%) cosmos DiT look like 51% MFU and sent us chasing a
nonexistent "Blackwell GEMM kernel" lever.

Use the printed `bf16 achieved peak` as `--peak-tflops` for dit_mfu_probe /
backward_mfu_probe / video_dit_mfu_probe on THIS machine. Also prints the build-arch
match check (torch must be built for the GPU's SM, else PTX-JIT fallback / slow).

Run: python -m vrl.scripts.perf.gemm_peak_probe
"""

from __future__ import annotations

import argparse

import torch


def _bench(m: int, n: int, k: int, dtype: torch.dtype, iters: int = 50, warmup: int = 10) -> float:
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        c = a @ b  # noqa: F841
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return 2 * m * n * k / (ms / 1e3) / 1e12


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192, 12288, 16384])
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    # bf16/fp16 inputs ignore tf32; disabled so an fp32 reference (if added) is honest.
    torch.backends.cuda.matmul.allow_tf32 = False

    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    print(f"device: {torch.cuda.get_device_name(0)}  ({sm})")
    print(f"torch {torch.__version__}  cuda(build) {torch.version.cuda}")
    print(f"build arch_list: {archs}")
    if sm in archs:
        print(f"  OK: torch is built for {sm} (native SASS, no PTX-JIT fallback)")
    else:
        print(f"  WARNING: {sm} NOT in build arch_list -> PTX-JIT fallback or failure. "
              f"Install a torch build whose CUDA covers {sm}.")

    print("\nbf16 dense GEMM (cuBLAS, fp32 accumulate — the real training/inference rate):")
    best_bf16 = 0.0
    for nsz in args.sizes:
        t = _bench(nsz, nsz, nsz, torch.bfloat16)
        best_bf16 = max(best_bf16, t)
        print(f"  {nsz:>6}^3   {t:6.0f} TFLOPS")
    print(f"  --> bf16 achieved peak = {best_bf16:.0f} TFLOPS  (USE THIS as --peak-tflops)")

    t_fp16 = _bench(8192, 8192, 8192, torch.float16)
    print(f"\nfp16 ref @8192^3: {t_fp16:.0f} TFLOPS")
    print("\nnote: vendor 'AI TOPS' headline is fp8/fp4 (often sparse) — do NOT use it as the")
    print("      bf16 MFU denominator. Going past this bf16 peak requires fp8/fp4 (LOSSY ->")
    print("      off the RL policy path only) — there is no lossless kernel beyond the bf16 ceiling.")


if __name__ == "__main__":
    main()
