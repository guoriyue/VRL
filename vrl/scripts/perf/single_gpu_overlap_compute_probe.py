"""Probe whether two tensor-heavy diffusion forwards overlap on one GPU.

Run:
    python -m vrl.scripts.perf.single_gpu_overlap_compute_probe

This answers the rollout-stage question directly: can reward-class tensor compute
hide behind the next denoise step on a single GPU, or do the tensor cores simply
serialize both forwards?
"""

from __future__ import annotations

import argparse

import torch

from vrl.scripts.perf.gemm_projection_breakdown import build_synthetic_inputs


def _synchronize() -> None:
    torch.cuda.synchronize()


def _elapsed_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    _synchronize()
    return start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="cosmos-predict2.5")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--denoise-frames", type=int, default=33)
    parser.add_argument("--reward-proxy-frames", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this overlap probe")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    model_a, kwargs_a = build_synthetic_inputs(
        args.family,
        batch=args.batch,
        device=device,
        dtype=dtype,
    )
    batch, channels, _frames, height, width = kwargs_a["hidden_states"].shape
    kwargs_a["hidden_states"] = torch.randn(
        batch,
        channels,
        args.denoise_frames,
        height,
        width,
        device=device,
        dtype=dtype,
    )

    model_b, kwargs_b = build_synthetic_inputs(
        args.family,
        batch=args.batch,
        device=device,
        dtype=dtype,
    )
    kwargs_b["hidden_states"] = torch.randn(
        batch,
        channels,
        args.reward_proxy_frames,
        height,
        width,
        device=device,
        dtype=dtype,
    )

    def forward_a() -> torch.Tensor:
        with torch.no_grad():
            return model_a(**kwargs_a)[0]

    def forward_b() -> torch.Tensor:
        with torch.no_grad():
            return model_b(**kwargs_b)[0]

    for _ in range(args.warmup):
        forward_a()
        forward_b()
    _synchronize()

    def run_a_only() -> None:
        for _ in range(args.iters):
            forward_a()

    def run_b_only() -> None:
        for _ in range(args.iters):
            forward_b()

    def run_serial() -> None:
        for _ in range(args.iters):
            forward_a()
            forward_b()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    def run_concurrent() -> None:
        for _ in range(args.iters):
            with torch.cuda.stream(stream_a):
                forward_a()
            with torch.cuda.stream(stream_b):
                forward_b()
            _synchronize()

    a_ms = _elapsed_ms(run_a_only) / args.iters
    b_ms = _elapsed_ms(run_b_only) / args.iters
    serial_ms = _elapsed_ms(run_serial) / args.iters
    concurrent_ms = _elapsed_ms(run_concurrent) / args.iters

    delta_ms = serial_ms - concurrent_ms
    delta_pct = delta_ms / serial_ms * 100.0
    if concurrent_ms < serial_ms * 0.95:
        verdict = "OVERLAP HELPS"
    elif concurrent_ms < serial_ms * 1.05:
        verdict = "NEUTRAL (tensor-heavy forwards serialize)"
    else:
        verdict = "OVERLAP HURTS"

    print("=== single-GPU compute overlap probe ===")
    print(f"family:                  {args.family}")
    print(f"A denoise frames:        {args.denoise_frames}")
    print(f"B reward-proxy frames:   {args.reward_proxy_frames}")
    print(f"A alone:                 {a_ms:.1f} ms/iter")
    print(f"B alone:                 {b_ms:.1f} ms/iter")
    print(f"serial A then B:         {serial_ms:.1f} ms/iter")
    print(f"concurrent two streams:  {concurrent_ms:.1f} ms/iter")
    print(f"delta:                   {delta_ms:+.1f} ms/iter ({delta_pct:+.1f}%)")
    print(f"verdict:                 {verdict}")


if __name__ == "__main__":
    main()
