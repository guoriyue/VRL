"""Backward-pass cost: forward vs backward time, MFU, and the grad-checkpoint tax.

Training replays the DiT with grad. Backward is ~2x forward FLOPs, so it is the
training cost center. Two questions this answers on a real model:

  1. Is backward at peak (high MFU) or launch-bound at small microbatch?
  2. What does gradient_checkpointing actually cost (extra recompute time) and
     save (peak memory)? -> decides whether off-by-default is safe.

Run: python -m vrl.scripts.perf.backward_mfu_probe --height 1024 --batches 1 2 4
"""

from __future__ import annotations

import argparse

import torch


def _cuda_time(fn, iters: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--peak-tflops", type=float, default=419.0)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the transformer (measures training-side fusion "
                        "headroom: backward MFU eager vs compiled). Restricts to ckpt=False "
                        "since compile+grad-checkpointing recompiles/collides.")
    args = p.parse_args()

    from diffusers import StableDiffusion3Pipeline

    # Load weights on CPU, then move ONLY the transformer to GPU. Putting the full
    # pipeline on CUDA pins the ~12GB T5-XXL text encoder in VRAM, which is NOT the
    # training footprint (embeds are precomputed/offloaded) and falsely OOMs the
    # backward. Training-time GPU memory = transformer + grads + activations only.
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    tf = pipe.transformer.to("cuda")
    device, dtype = tf.device, tf.dtype
    if args.compile:
        # Same transformer object serves rollout AND replay (base.py replay_forward ->
        # forward_step), so compiling it is exactly the training-side fusion the real
        # recipe gets when model.torch_compile.enable=true. Backward graph is compiled
        # too. mode=default to avoid the CUDA-graph/PEFT collisions noted for Wan.
        tf = torch.compile(tf, mode="default")
    cfg = tf.config
    n_params = sum(p.numel() for p in tf.parameters())
    d_model = cfg.num_attention_heads * cfg.attention_head_dim
    text_tok = 333  # SD3 joint sequence text length
    # Random embeds of the right shape — avoids loading the text encoders to GPU.
    embeds = torch.randn(1, text_tok, cfg.joint_attention_dim, device=device, dtype=dtype)
    pooled = torch.randn(1, cfg.pooled_projection_dim, device=device, dtype=dtype)
    h, w = args.height // 8, args.width // 8
    img_tok = (h // 2) * (w // 2)
    seq = img_tok + text_tok

    def _fwd_flops(rows: int) -> float:
        lin = 2 * n_params * seq * rows
        attn = cfg.num_layers * 2 * 2 * (seq * seq) * d_model * rows
        return lin + attn

    print(f"model {n_params/1e9:.2f}B, seq={seq}, peak~{args.peak_tflops:.0f} TFLOPS bf16")
    print(f"\n{'batch':>5} | {'ckpt':>5} | {'fwd ms':>7} | {'bwd ms':>7} | "
          f"{'bwd/fwd':>7} | {'fwd+bwd MFU':>11} | {'peak GB':>7}")

    for b in args.batches:
        # No CFG in training replay here; one row per sample.
        emb = embeds.repeat_interleave(b, dim=0)
        pol = pooled.repeat_interleave(b, dim=0)
        latents = torch.randn(b, cfg.in_channels, h, w, device=device, dtype=dtype)
        pipe.scheduler.set_timesteps(28, device=device)
        ts = pipe.scheduler.timesteps[14].expand(b)
        # fwd+bwd FLOPs ~ 3x forward (fwd 2ND + bwd 4ND).
        total_flops = 3.0 * _fwd_flops(b)

        # compile + grad-checkpointing recompiles/collides; isolate to ckpt=False.
        ckpt_modes = (False,) if args.compile else (False, True)
        for ckpt in ckpt_modes:
            if ckpt:
                tf.enable_gradient_checkpointing()
            else:
                tf.disable_gradient_checkpointing()

            def _fwd_nograd(emb=emb, pol=pol, latents=latents, ts=ts):
                with torch.no_grad():
                    tf(hidden_states=latents, timestep=ts, encoder_hidden_states=emb,
                       pooled_projections=pol, return_dict=False)

            def _step(emb=emb, pol=pol, latents=latents, ts=ts):
                tf.zero_grad(set_to_none=True)
                out = tf(hidden_states=latents, timestep=ts, encoder_hidden_states=emb,
                         pooled_projections=pol, return_dict=False)[0]
                out.float().pow(2).mean().backward()
                del out

            try:
                warmup = 6 if args.compile else 2  # inductor needs more warmup per shape
                fwd_ms = _cuda_time(_fwd_nograd, iters=4, warmup=warmup)
                torch.cuda.reset_peak_memory_stats()
                step_ms = _cuda_time(_step, iters=4, warmup=warmup)
                peak = torch.cuda.max_memory_allocated() / 1024**3
                bwd_ms = max(step_ms - fwd_ms, 0.01)
                mfu = total_flops / (step_ms / 1e3) / 1e12 / args.peak_tflops * 100
                print(f"{b:>5} | {ckpt!s:>5} | {fwd_ms:>7.1f} | {bwd_ms:>7.1f} | "
                      f"{bwd_ms/fwd_ms:>7.2f} | {mfu:>10.0f}% | {peak:>7.2f}")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                print(f"{b:>5} | {ckpt!s:>5} | OOM/err ({type(exc).__name__})")
            finally:
                tf.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    print("\nreads: bwd/fwd~2 expected; if MFU low at batch=1 -> launch-bound "
          "(batch up / compile helps); ckpt True = slower (recompute) but lower peak GB.")


if __name__ == "__main__":
    main()
