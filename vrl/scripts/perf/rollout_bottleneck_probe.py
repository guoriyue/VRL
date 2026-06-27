"""Where does diffusion rollout time go: compute, launch overhead, or VAE?

Decisive test for compute-bound vs launch-bound: time the DiT forward at growing
batch sizes. If ms/sample drops as batch grows, the GPU was idle (launch/overhead
bound) and you get extra samples nearly free -> bigger batch / compile fixes it.
If ms/sample is flat, you are compute-bound and only fewer FLOPs (steps / fp8 /
resolution / smaller model) helps.

Also decomposes one rollout into DiT-forward vs VAE-decode vs scheduler/python.

Run: python -m vrl.scripts.perf.rollout_bottleneck_probe --steps 28 --height 1024
"""

from __future__ import annotations

import argparse

import torch


def _cuda_time(fn, iters: int, warmup: int = 3) -> float:
    """Mean wall-time (ms) of fn() over iters, GPU-synced, after warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    from diffusers import StableDiffusion3Pipeline

    print(f"loading {args.model} (bf16, cuda) ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    transformer = pipe.transformer
    device, dtype = transformer.device, transformer.dtype

    (embeds, neg, pooled, neg_pooled) = pipe.encode_prompt(
        prompt="a photorealistic astronaut riding a horse on mars",
        prompt_2=None, prompt_3=None, do_classifier_free_guidance=True, device=device,
    )
    full_embeds = torch.cat([neg, embeds])      # CFG -> 2 rows per sample
    full_pooled = torch.cat([neg_pooled, pooled])
    num_ch = transformer.config.in_channels
    h, w = args.height // 8, args.width // 8

    pipe.scheduler.set_timesteps(args.steps, device=device)
    t = pipe.scheduler.timesteps[len(pipe.scheduler.timesteps) // 2]

    print(f"\n=== DiT forward: batch-size scaling (steps={args.steps}, {args.height}^2) ===")
    print(f"{'batch':>6} | {'ms/forward':>11} | {'ms/sample':>10} | {'tokens':>8} | verdict")
    base_per_sample = None
    for b in args.batches:
        latents = torch.randn(b, num_ch, h, w, device=device, dtype=dtype)
        emb = full_embeds.repeat_interleave(b, dim=0) if full_embeds.shape[0] == 2 else full_embeds
        pol = full_pooled.repeat_interleave(b, dim=0) if full_pooled.shape[0] == 2 else full_pooled
        model_in = torch.cat([latents] * 2)      # CFG
        ts = t.expand(model_in.shape[0])

        def _fwd(model_in=model_in, ts=ts, emb=emb, pol=pol):
            with torch.no_grad():
                transformer(
                    hidden_states=model_in, timestep=ts,
                    encoder_hidden_states=emb, pooled_projections=pol,
                    return_dict=False,
                )

        try:
            ms = _cuda_time(_fwd, args.iters)
        except torch.cuda.OutOfMemoryError:
            print(f"{b:>6} | OOM")
            torch.cuda.empty_cache()
            continue
        per_sample = ms / b
        if base_per_sample is None:
            base_per_sample = per_sample
        ratio = per_sample / base_per_sample
        verdict = "launch/overhead-bound (free headroom)" if ratio < 0.7 else (
            "compute-bound (saturated)" if ratio > 0.9 else "transition")
        print(f"{b:>6} | {ms:>11.2f} | {per_sample:>10.2f} | {h * w:>8} | {verdict}")

    # Per-phase decomposition for one rollout at batch=group size.
    g = max(args.batches)
    print(f"\n=== one rollout decomposition (batch={g}, {args.steps} steps) ===")
    latents = torch.randn(g, num_ch, h, w, device=device, dtype=dtype)
    emb = full_embeds.repeat_interleave(g, dim=0)
    pol = full_pooled.repeat_interleave(g, dim=0)

    def _denoise():
        pipe.scheduler.set_timesteps(args.steps, device=device)  # reset step_index
        lat = latents.clone()
        for tt in pipe.scheduler.timesteps:
            mi = torch.cat([lat] * 2)
            with torch.no_grad():
                out = transformer(
                    hidden_states=mi, timestep=tt.expand(mi.shape[0]),
                    encoder_hidden_states=emb, pooled_projections=pol, return_dict=False,
                )[0]
            u, c = out.chunk(2)
            lat = pipe.scheduler.step(u + 4.5 * (c - u), tt, lat, return_dict=False)[0]
        return lat

    def _vae(lat):
        with torch.no_grad():
            z = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            return pipe.vae.decode(z, return_dict=False)[0]

    denoise_ms = _cuda_time(_denoise, iters=3, warmup=1)
    final = _denoise()
    vae_ms = _cuda_time(lambda: _vae(final), iters=3, warmup=1)
    total = denoise_ms + vae_ms
    print(f"  denoise ({args.steps} DiT fwd): {denoise_ms:>9.1f} ms  ({denoise_ms/total*100:4.1f}%)")
    print(f"  VAE decode               : {vae_ms:>9.1f} ms  ({vae_ms/total*100:4.1f}%)")
    print(f"  per DiT forward (avg)    : {denoise_ms/args.steps:>9.2f} ms")
    print(f"  total / sample           : {total/g:>9.1f} ms")
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  peak memory              : {mem:>9.2f} GB")


if __name__ == "__main__":
    main()
