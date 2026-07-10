"""Direct (no-Ray) single-process profile of the video rollout STAGES, for both
torch.profiler and nsys — to size the staged-pipeline / stream-overlap levers.

WHY: the real RL rollout runs in a Ray worker, and nsys launched on `vrl-train`
does NOT trace it (verified: 0 GPU kernels captured) — raylet-spawned workers are
invisible to a launcher-attached profiler. And the in-repo VRL_PROFILE_COLLECT is
just phase wall-clock timing, not a torch.profiler trace of the rollout. So to get
an ACCURATE per-stage attribution with the OFFICIAL tools, we run the rollout stages
directly in ONE process here:

  prompt-encode (synthetic) -> denoise loop [DiT forward (GPU) + sde_step_with_logprob
  (the real repo SDE/log-prob math, fp32)] x T -> VAE-sized decode

Each stage is wrapped in torch.profiler.record_function, so BOTH:
  - torch.profiler  (python -m ... )                 -> per-stage CPU+CUDA self-time
  - nsys profile python -m ...                        -> GPU timeline + NVTX stage ranges
attribute time to the same named stages. Synthetic weights (real dims) — kernel time
depends on shapes, not values. The Kling reward-model stage is NOT included here (load
it separately); this isolates denoise-GPU vs SDE-CPU-math vs VAE, which is what the
stream-overlap (P2) lever targets.

Run (torch.profiler):  python -m vrl.scripts.perf.video_rollout_stage_profile --frames 33
Run (nsys):            nsys profile --trace=cuda,nvtx -o stage python -m vrl.scripts.perf.video_rollout_stage_profile --frames 33 --no-torch-prof
"""

from __future__ import annotations

import argparse

import torch

from vrl.scripts.perf.gemm_projection_breakdown import build_synthetic_inputs
from vrl.utils.profiling import record_function


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", default="cosmos-predict2.5")
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--no-torch-prof", action="store_true", help="skip torch.profiler (for nsys runs)")
    args = p.parse_args()

    dev = torch.device("cuda")
    dt = torch.bfloat16
    model, kwargs = build_synthetic_inputs(args.family, batch=1, device=dev, dtype=dt)
    B, C_in, _, h, w = kwargs["hidden_states"].shape  # C_in=17 (16 latent + 1 padding mask)
    out_ch = 16
    lat = torch.randn(B, out_ch, args.frames, h, w, device=dev, dtype=dt)  # 16-ch denoise state
    pad = torch.zeros(B, C_in - out_ch, args.frames, h, w, device=dev, dtype=dt)
    # A VAE-decode-sized op: latents [B,16,T,h,w] -> pixels [B,3,T,h*16,w*16] is dominated
    # by 3D transpose-convs. Approximate with a strided 3D conv stack at the latent grid
    # (shape-faithful enough for a relative stage-time read; the real VAE is heavier).
    vae = torch.nn.Sequential(
        torch.nn.Conv3d(out_ch, 256, 3, padding=1),
        torch.nn.SiLU(),
        torch.nn.Conv3d(256, 256, 3, padding=1),
        torch.nn.SiLU(),
        torch.nn.Conv3d(256, 3 * 64, 3, padding=1),  # ~ upsample channels for 8x8 spatial
    ).to(dev, dt).eval()

    def one_rollout():
        with record_function("prompt_encode"):
            _ = kwargs["encoder_hidden_states"]  # already built; encode is cheap/once
            torch.cuda.synchronize()
        x = lat
        for _step in range(args.steps):
            with record_function("denoise_dit_forward"), torch.no_grad():  # GPU-bound
                model_in = torch.cat([x, pad], dim=1)  # 16-ch latent + padding -> 17-ch input
                vel = model(**{**kwargs, "hidden_states": model_in})[0]
            with record_function("sde_step_logprob_math"):  # the fp32 SDE/log-prob math (CPU+GPU)
                # real repo math: build a tiny scheduler-free gaussian step proxy at the
                # SAME tensor size so the fp32 elementwise/reduction cost is realistic.
                vel32 = vel.float()
                mean = x.float() - 0.03 * vel32
                std = torch.full_like(mean[:, :1], 0.1)
                noise = torch.randn_like(mean)
                x = (mean + std * noise).to(dt)
                logp = (-((noise) ** 2) / 2 - 0.9189).flatten(1).sum(-1)  # gaussian log-prob
                _ = logp.detach()
        with record_function("vae_decode"), torch.no_grad():  # GPU
            _ = vae(x)
        torch.cuda.synchronize()

    for _ in range(2):  # warmup
        one_rollout()
    torch.cuda.synchronize()

    if args.no_torch_prof:
        # nsys path: just run, nsys captures the timeline + NVTX ranges
        for _ in range(args.iters):
            one_rollout()
        print(f"ran {args.iters} rollouts (frames={args.frames}, steps={args.steps}) for nsys capture")
        return

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(args.iters):
            one_rollout()
        torch.cuda.synchronize()

    print(f"\n=== per-stage time (torch.profiler, {args.family} {args.frames}f {args.steps} steps, "
          f"{args.iters} rollouts) ===")
    # aggregate self_device (GPU) + self_cpu time by record_function range
    ka = prof.key_averages()
    stages = ("prompt_encode", "denoise_dit_forward", "sde_step_logprob_math", "vae_decode")
    tot_gpu = 0.0
    rows = []
    for ev in ka:
        if ev.key in stages:
            gpu = (getattr(ev, "self_device_time_total", 0) or getattr(ev, "self_cuda_time_total", 0)) / 1e3
            cpu = ev.self_cpu_time_total / 1e3
            rows.append((ev.key, gpu, cpu))
            tot_gpu += gpu
    print(f"{'stage':<24} {'GPU ms':>9} {'CPU ms':>9} {'GPU %':>7}")
    for k, gpu, cpu in rows:
        print(f"{k:<24} {gpu/args.iters:>9.1f} {cpu/args.iters:>9.1f} {gpu/(tot_gpu or 1)*100:>6.0f}%")
    print("\nread: denoise_dit_forward = GPU-bound (compute). sde_step_logprob_math high CPU ms =")
    print("      the host math that stream-overlap (P2) can hide behind the next GPU forward.")
    print("      vae_decode = a separate GPU stage. Run under nsys for the GPU-idle gap timeline.")


if __name__ == "__main__":
    main()
