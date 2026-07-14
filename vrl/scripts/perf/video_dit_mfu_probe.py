"""Video DiT analytic FLOP/time probe for Cosmos/Wan shapes.

This is a screening tool: it estimates achieved TFLOPS from model-shape FLOP formulas
and divides by a measured bf16 GEMM peak. That is useful for trend checks, but it is
not enough to prove tensor-core saturation. Final "is there kernel headroom?" calls must
be checked with Nsight Compute tensor-pipe SOL against a square-GEMM baseline on the same
GPU. The old "cosmos post-compile ~51% MFU" result was caused by using an inflated vendor
peak instead of the real bf16+fp32-accumulate peak.

This sweeps the latent FRAME count for a real-dims (synthetic-weight; only shapes drive
kernel time) Cosmos/Wan transformer and reports, per frame count: token count,
attention-FLOP fraction, eager ms; then an eager-vs-compile analytic MFU A/B at a
representative video length.

Reuses build_synthetic_inputs (the single source of production-faithful dims shared with
the GEMM breakdown + compile A/B), so shapes match the real model.

Run: python -m vrl.scripts.perf.video_dit_mfu_probe --family cosmos-predict2.5 --frames 1 4 8 16
"""

from __future__ import annotations

import argparse

import torch

from vrl.scripts.perf.common.synthetic_diffusion import build_synthetic_inputs
from vrl.scripts.perf.common.timing import cuda_mean_ms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--family",
        default="cosmos-predict2.5",
        choices=["cosmos-predict2", "cosmos-predict2.5", "wan_2_1"],
    )
    p.add_argument("--frames", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--compile-at", type=int, default=8, help="frame count for the compile A/B")
    p.add_argument(
        "--peak-tflops",
        type=float,
        default=0.0,
        help="bf16 MFU denominator; 0 = measure this GPU's real bf16 peak "
        "(gpu_preflight). NEVER hardcode the vendor fp8/sparse headline "
        "(419 on a 5090 is fp8/sparse; real bf16 dense is ~232).",
    )
    p.add_argument("--iters", type=int, default=6)
    args = p.parse_args()

    if args.peak_tflops <= 0:
        from vrl.scripts.perf.gpu_preflight import measured_bf16_peak_tflops

        args.peak_tflops = measured_bf16_peak_tflops()
        print(f"measured bf16 peak (MFU denominator) = {args.peak_tflops:.0f} TFLOPS")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model, kwargs = build_synthetic_inputs(
        args.family, batch=args.batch, device=device, dtype=dtype
    )
    cfg = model.config
    n_params = sum(q.numel() for q in model.parameters())
    n_layers = cfg.num_layers
    d_model = cfg.num_attention_heads * cfg.attention_head_dim
    base_hs = kwargs["hidden_states"]  # [B, C, t, h, w]
    B, C, _, h, w = base_hs.shape
    ph, pw = cfg.patch_size[1], cfg.patch_size[2]
    tok_per_frame = (h // ph) * (w // pw)
    text_tok = kwargs["encoder_hidden_states"].shape[1]

    print(
        f"{args.family}: {n_params / 1e9:.2f}B, {n_layers} layers, d_model={d_model}, "
        f"patch={tuple(cfg.patch_size)}, latent {h}x{w} -> {tok_per_frame} tok/frame, text_tok={text_tok}"
    )
    print(
        f"\n{'frames':>6} | {'vid_tok':>7} | {'lin TFLOP':>9} | {'attn TFLOP':>10} | "
        f"{'attn%':>5} | {'ms/fwd':>7} | regime"
    )

    def _run(frames: int):
        hs = torch.randn(B, C, frames, h, w, device=device, dtype=dtype)
        kw = dict(kwargs)
        kw["hidden_states"] = hs
        return kw

    for frames in args.frames:
        vid_tok = frames * tok_per_frame
        seq = vid_tok + text_tok  # joint/self+cross attention sequence
        flops_linear = 2 * n_params * vid_tok * B
        flops_attn = n_layers * 2 * 2 * (seq * seq) * d_model * B
        attn_frac = flops_attn / (flops_linear + flops_attn)
        kw = _run(frames)

        def _fwd(kw=kw):
            with torch.no_grad():
                model(**kw)

        try:
            ms = cuda_mean_ms(_fwd, iters=args.iters, warmup=4)
            ms_s = f"{ms:7.1f}"
        except torch.cuda.OutOfMemoryError:
            ms_s = "    OOM"
            torch.cuda.empty_cache()
        regime = (
            "ATTN-DOMINATED"
            if attn_frac >= 0.5
            else ("attn rising" if attn_frac >= 0.25 else "linear-bound")
        )
        print(
            f"{frames:>6} | {vid_tok:>7} | {flops_linear / 1e12:>9.1f} | {flops_attn / 1e12:>10.1f} | "
            f"{attn_frac * 100:>4.0f}% | {ms_s} | {regime}"
        )

    # eager vs compile MFU at a representative video length
    f = args.compile_at
    kw = _run(f)
    vid_tok = f * tok_per_frame
    seq = vid_tok + text_tok
    flops = 2 * n_params * vid_tok * B + n_layers * 2 * 2 * (seq * seq) * d_model * B

    def _fwd(kw=kw):
        with torch.no_grad():
            model(**kw)

    try:
        eager_ms = cuda_mean_ms(_fwd, iters=args.iters, warmup=4)
        eager_tflops = flops / (eager_ms / 1e3) / 1e12
        print(f"\n=== eager (frames={f}, vid_tok={vid_tok}) ===")
        print(
            f"  {eager_ms:.1f} ms -> {eager_tflops:.0f} TFLOPS -> MFU ~ {eager_tflops / args.peak_tflops * 100:.0f}%"
        )
        print("compiling (inductor default, ~1-3 min) ...")
        compiled = torch.compile(model, mode="default")

        def _fwd_c(kw=kw):
            with torch.no_grad():
                compiled(**kw)

        comp_ms = cuda_mean_ms(_fwd_c, iters=args.iters, warmup=6)
        comp_tflops = flops / (comp_ms / 1e3) / 1e12
        print("=== torch.compile ===")
        print(
            f"  {comp_ms:.1f} ms -> {comp_tflops:.0f} TFLOPS -> MFU ~ {comp_tflops / args.peak_tflops * 100:.0f}%"
        )
        print(f"  measured compile speedup = {eager_ms / comp_ms:.2f}x")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        print(f"\ncompile A/B at frames={f} failed: {type(exc).__name__}: {str(exc)[:160]}")

    print("\nread: attn% rising with frames -> for VIDEO the lossless compute lever shifts to")
    print("      attention (FA-3); fused AdaLN (AdaptiveLoad) also targets video DiT. Contrast")
    print("      SD3.5 image where attn stays <13%; measure compile gain with a direct A/B.")


if __name__ == "__main__":
    main()
