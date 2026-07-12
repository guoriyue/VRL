"""Are fp8/fp4 linears actually faster than bf16 at DiT shapes? (the gate)

WHY: before wiring a low-precision linear into the rollout engine, confirm the
premise — that the production module, *including* per-call dynamic activation
quantization, beats a plain bf16 linear at the matmul shapes a diffusion DiT
actually runs. On consumer Blackwell (sm_120), quantization can eat the
tensor-core win, so this is measured rather than inferred from GEMM-only timing.

WHAT: for each (M, K, N), it times a bf16 `nn.Linear` and the default production
`Fp8Linear` recipe (rowwise). On MLP rows, where the live FP4 targeting policy
actually swaps modules, it also times `Fp4Linear`. It reports speedup and output
drift per shape, then evaluates fp8 and fp4 independently. A supported scheme
that misses the net-speed gate makes the command fail, so a negative result
cannot be hidden behind another scheme's positive verdict.

FAITHFULNESS: both paths use the production swap modules. Weights are quantized
once, while activation quantization remains inside every timed forward, matching
the rollout cost rather than the GEMM-only hardware ceiling.

Usage:  python -m vrl.scripts.perf.quantized_linear_benchmark
"""

from __future__ import annotations

import torch
from torch import nn

from vrl.nn.quantization import Fp4Linear, Fp8Linear, nvfp4_available
from vrl.scripts.perf.common.fp8_math import relative_l1_drift
from vrl.scripts.perf.common.timing import cuda_mean_ms


def _summarize(
    scheme: str,
    speedups: list[float],
    drifts: list[float],
    *,
    min_speedup: float,
) -> bool:
    """Print one scheme's independent efficiency verdict and return its gate result."""
    geomean = float(torch.tensor(speedups).log().mean().exp())
    passed = geomean >= min_speedup
    print(
        f"{scheme}: geomean speedup={geomean:.2f}x   "
        f"mean drift={sum(drifts) / len(drifts):.4f}   max drift={max(drifts):.4f}",
    )
    print(
        f"  {scheme} {'PASSES' if passed else 'FAILS'} the {min_speedup:.2f}x "
        "net-linear speed gate. "
        + ("Eligible for later end-to-end gates." if passed else "Not worth wiring as-is."),
    )
    return passed


def main(*, schemes: tuple[str, ...] = ("fp8", "fp4")) -> None:
    """Benchmark the requested schemes; the canonical CLI runs both by default."""

    schemes = tuple(dict.fromkeys(schemes))
    unknown = sorted(set(schemes) - {"fp8", "fp4"})
    if not schemes or unknown:
        raise ValueError(f"schemes must be a non-empty fp8/fp4 subset; got {schemes!r}")
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device with fp8 support (Hopper/Blackwell)")
    torch.manual_seed(0)
    dev = "cuda"
    run_fp8 = "fp8" in schemes
    run_fp4 = "fp4" in schemes and nvfp4_available()
    print(f"== {'/'.join(schemes)} vs bf16 linear on {torch.cuda.get_device_name(0)} ==")
    recipes = []
    if run_fp8:
        recipes.append("fp8=rowwise(default)")
    if "fp4" in schemes:
        recipes.append("fp4=nvfp4")
    print(f"M=tokens*batch, K=in, N=out; {', '.join(recipes)}; includes activation quant")
    if "fp4" in schemes and not run_fp4:
        print("(nvfp4 _scaled_mm unavailable on this device/build — fp4 columns skipped)")
    print()
    header = f"{'projection':>10} | {'shape (M,K,N)':>22} | {'bf16 ms':>8}"
    if run_fp8:
        header += f" | {'fp8 ms':>8} | {'speedup':>7} | {'drift':>8}"
    if run_fp4:
        header += f" | {'fp4 ms':>8} | {'fp4 spd':>7} | {'fp4 drift':>9}"
    print(header)
    print("-" * len(header))

    # (M, K, N): the four DiT GEMMs at a few hidden sizes / token counts.
    hiddens = [1536, 2048, 4096]
    token_counts = [4096, 8192]  # ~1024px image latent / small video
    shapes: list[tuple[str, int, int, int]] = []
    for h in hiddens:
        for m in token_counts:
            shapes.append(("qkv", m, h, 3 * h))
            shapes.append(("attn_out", m, h, h))
            shapes.append(("mlp_up", m, h, 4 * h))
            shapes.append(("mlp_down", m, 4 * h, h))

    fp8_speedups: list[float] = []
    fp8_drifts: list[float] = []
    fp4_speedups: list[float] = []
    fp4_drifts: list[float] = []
    for projection, m, k, n in shapes:
        lin = nn.Linear(k, n, bias=True).cuda().to(torch.bfloat16)
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            ref = lin(x)
            bf16_ms = cuda_mean_ms(lambda lin=lin, x=x: lin(x), iters=100, warmup=25)
        row = f"{projection:>10} | {f'({m},{k},{n})':>22} | {bf16_ms:8.4f}"
        if run_fp8:
            # Use the module default because that is also the loader's default
            # production recipe. Historical tensorwise numbers are not a valid
            # claim for an unspecified ``precision.rollout_recipe``.
            fp8 = Fp8Linear(lin)
            with torch.no_grad():
                q8 = fp8(x)
                fp8_ms = cuda_mean_ms(lambda fp8=fp8, x=x: fp8(x), iters=100, warmup=25)
            fp8_speedup = bf16_ms / fp8_ms
            fp8_drift = relative_l1_drift(q8, ref)
            fp8_speedups.append(fp8_speedup)
            fp8_drifts.append(fp8_drift)
            row += f" | {fp8_ms:8.4f} | {fp8_speedup:6.2f}x | {fp8_drift:8.4f}"
        if run_fp4 and projection.startswith("mlp_"):
            fp4_lin = Fp4Linear(lin)
            with torch.no_grad():
                q4 = fp4_lin(x)
                fp4_ms = cuda_mean_ms(
                    lambda fp4_lin=fp4_lin, x=x: fp4_lin(x), iters=100, warmup=25
                )
            fp4_speedup = bf16_ms / fp4_ms
            fp4_drift = relative_l1_drift(q4, ref)
            fp4_speedups.append(fp4_speedup)
            fp4_drifts.append(fp4_drift)
            row += f" | {fp4_ms:8.4f} | {fp4_speedup:6.2f}x | {fp4_drift:9.4f}"
        elif run_fp4:
            row += f" | {'-':>8} | {'-':>7} | {'-':>9}"
        print(row)

    print("-" * len(header))
    print("\n-- verdict --")
    results: dict[str, bool] = {}
    if run_fp8:
        results["fp8"] = _summarize("fp8", fp8_speedups, fp8_drifts, min_speedup=1.05)
    if run_fp4:
        results["fp4"] = _summarize(
            "fp4 (MLP-only)",
            fp4_speedups,
            fp4_drifts,
            min_speedup=1.05,
        )
    failed = [scheme for scheme, passed in results.items() if not passed]
    if failed:
        print(f"performance gate failed: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
