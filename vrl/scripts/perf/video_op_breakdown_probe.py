"""Where does video DiT GPU time go: matmul, attention, or norm/elementwise work?

WHY: this is an op-time attribution probe, not a saturation proof. The historical
"cosmos is ~51% MFU" conclusion came from dividing analytic FLOPs by an inflated
vendor peak. Nsight Compute later showed the main bf16 GEMMs run at the same tensor-pipe
SOL as a square GEMM on RTX 5090, so the compute kernels are already near the real
bf16+fp32-accumulate ceiling. This probe still answers a different useful question:
which CUDA kernel families dominate eager/compiled wall time?

It profiles one forward (eager AND compiled) with torch.profiler and buckets every CUDA
kernel's self-time into:

  - MATMUL    : GEMM (linear/proj/MLP) — the useful linear FLOPs
  - ATTENTION : SDPA / flash / fmha — the useful attention FLOPs
  - NORM_ELEM : layernorm / RoPE / pointwise mul-add / gate / fused triton pointwise —
                the bandwidth-bound AdaLN-Zero + modulation work a fused kernel targets
  - OTHER     : copies / cat / reshape / misc

If NORM_ELEM is a large slice of the COMPILED time, a hand-fused AdaLN-Zero kernel has
real headroom (inductor did not fuse it away). Top-N kernels are printed raw so the
bucketing is auditable, not a black box.

Run: python -m vrl.scripts.perf.video_op_breakdown_probe --family cosmos-predict2.5 --frames 8
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from vrl.scripts.perf.common.synthetic_diffusion import build_synthetic_inputs

# Substring buckets, checked in order; first match wins. Lowercased kernel name.
_BUCKETS = (
    ("ATTENTION", ("flash", "fmha", "attention", "scaled_dot", "mem_eff", "_sdp", "sdpa")),
    ("MATMUL", ("gemm", "cutlass", "cublas", "ampere", "sm80", "sm90", "sm120", "addmm",
                "matmul", "wgrad", "dgrad", "s16816", "h16816", "_mm_", "gett", "implicit")),
    ("NORM_ELEM", ("norm", "rope", "rotary", "elementwise", "triton_poi", "triton_red",
                   "triton_per", "mul", "add", "layer_norm", "rms", "silu", "gelu", "vectorized")),
)


def _bucket(name: str) -> str:
    low = name.lower()
    for label, keys in _BUCKETS:
        if any(k in low for k in keys):
            return label
    return "OTHER"


def _profile(fwd, iters: int, warmup: int = 5) -> tuple[dict, list]:
    for _ in range(warmup):
        fwd()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fwd()
        torch.cuda.synchronize()
    totals: dict[str, float] = defaultdict(float)
    rows = []
    for ev in prof.key_averages():
        # torch>=2.11 renamed self_cuda_time_total -> self_device_time_total.
        us = getattr(ev, "self_device_time_total", None)
        if us is None:
            us = getattr(ev, "self_cuda_time_total", 0.0)
        t = us / 1e3  # ms (summed over iters)
        if t <= 0:
            continue
        totals[_bucket(ev.key)] += t
        rows.append((ev.key, t))
    rows.sort(key=lambda r: -r[1])
    return totals, rows


def _report(tag: str, totals: dict, rows: list, iters: int) -> None:
    total = sum(totals.values()) or 1.0
    print(f"\n=== {tag} — GPU self-time by bucket (per forward) ===")
    for label in ("MATMUL", "ATTENTION", "NORM_ELEM", "OTHER"):
        ms = totals.get(label, 0.0) / iters
        print(f"  {label:<10} {ms:8.2f} ms  ({totals.get(label,0.0)/total*100:5.1f}%)")
    useful = (totals.get("MATMUL", 0) + totals.get("ATTENTION", 0)) / total * 100
    print(f"  -- useful FLOP kernels (MATMUL+ATTN) = {useful:.1f}% of GPU time; "
          f"the rest is fusion/overhead headroom --")
    print("  top kernels:")
    for name, t in rows[:10]:
        print(f"    {t/iters:7.2f} ms  [{_bucket(name):<9}] {name[:70]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", default="cosmos-predict2.5",
                   choices=["cosmos-predict2", "cosmos-predict2.5", "wan_2_1"])
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    model, kwargs = build_synthetic_inputs(args.family, batch=args.batch, device=device, dtype=dtype)
    B, C, _, h, w = kwargs["hidden_states"].shape
    kwargs["hidden_states"] = torch.randn(B, C, args.frames, h, w, device=device, dtype=dtype)
    cfg = model.config
    tok = args.frames * (h // cfg.patch_size[1]) * (w // cfg.patch_size[2])
    print(f"{args.family}: {sum(q.numel() for q in model.parameters())/1e9:.2f}B, "
          f"{cfg.num_layers} layers, frames={args.frames}, vid_tok={tok}")

    def _fwd():
        with torch.no_grad():
            model(**kwargs)

    eager_tot, eager_rows = _profile(_fwd, args.iters)
    _report("EAGER", eager_tot, eager_rows, args.iters)

    print("\ncompiling (inductor default, ~1-3 min) ...")
    compiled = torch.compile(model, mode="default")

    def _fwd_c():
        with torch.no_grad():
            compiled(**kwargs)

    comp_tot, comp_rows = _profile(_fwd_c, args.iters, warmup=8)
    _report("COMPILED", comp_tot, comp_rows, args.iters)

    nm = comp_tot.get("NORM_ELEM", 0.0) / (sum(comp_tot.values()) or 1.0) * 100
    print(f"\nverdict: after compile, NORM_ELEM (AdaLN/RoPE/modulation/pointwise) = {nm:.1f}% "
          f"of GPU time.\n  >20% -> fused AdaLN-Zero kernel has real headroom (P1 priority); "
          f"check ATTENTION% for FA-3 priority.")


if __name__ == "__main__":
    main()
