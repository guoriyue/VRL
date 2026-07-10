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

# Measure the EXACT production policy, not a copy: selective_checkpoint_func is the
# same SAC helper that enable_transformer_gradient_checkpointing applies to every
# diffusion family, so probe numbers describe what real training actually runs.
# Imported from the trainers module (not the online runner) so the probe stays a
# lightweight perf script and does not pull in Ray/launcher.
from vrl.scripts.perf.common.timing import cuda_mean_ms
from vrl.trainers.activation_checkpointing import selective_checkpoint_func


def _apply_ckpt(tf, mode: str) -> None:
    """off | full | selective — full recomputes every block, selective uses SAC."""
    tf.disable_gradient_checkpointing()
    if mode == "full":
        tf.enable_gradient_checkpointing()
    elif mode == "selective":
        tf.enable_gradient_checkpointing(gradient_checkpointing_func=selective_checkpoint_func)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--peak-tflops", type=float, default=0.0,
                   help="bf16 MFU denominator; 0 = measure real bf16 peak (gpu_preflight). "
                        "419 is the fp8/sparse headline; real 5090 bf16 dense is ~232.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the transformer (measures training-side fusion "
                        "headroom: backward MFU eager vs compiled). Restricts to ckpt=False "
                        "since compile+grad-checkpointing recompiles/collides.")
    args = p.parse_args()

    if args.peak_tflops <= 0:
        from vrl.scripts.perf.gpu_preflight import measured_bf16_peak_tflops
        args.peak_tflops = measured_bf16_peak_tflops()
        print(f"measured bf16 peak (MFU denominator) = {args.peak_tflops:.0f} TFLOPS")

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
    print(f"\n{'batch':>5} | {'ckpt':>9} | {'fwd ms':>7} | {'bwd ms':>7} | "
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

        # Eager measures all three. Under --compile we skip full (compile + full
        # block checkpointing recompiles/collides) but DO measure selective: SAC
        # goes through AOTAutograd's min-cut partitioner, so compiled x selective
        # is the P1 question — does it compose and beat compiled x off at a larger
        # batch? (off / selective only; errors are caught and printed per row.)
        ckpt_modes = ("off", "selective") if args.compile else ("off", "full", "selective")
        for ckpt in ckpt_modes:
            _apply_ckpt(tf, ckpt)

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
                fwd_ms = cuda_mean_ms(_fwd_nograd, iters=4, warmup=warmup)
                torch.cuda.reset_peak_memory_stats()
                step_ms = cuda_mean_ms(_step, iters=4, warmup=warmup)
                peak = torch.cuda.max_memory_allocated() / 1024**3
                bwd_ms = max(step_ms - fwd_ms, 0.01)
                mfu = total_flops / (step_ms / 1e3) / 1e12 / args.peak_tflops * 100
                print(f"{b:>5} | {ckpt:>9} | {fwd_ms:>7.1f} | {bwd_ms:>7.1f} | "
                      f"{bwd_ms/fwd_ms:>7.2f} | {mfu:>10.0f}% | {peak:>7.2f}")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                print(f"{b:>5} | {ckpt:>9} | OOM/err ({type(exc).__name__})")
            finally:
                tf.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    print("\nreads: bwd/fwd~2 expected; if MFU low at batch=1 -> launch-bound "
          "(batch up / compile helps). full = lower peak GB but recompute tax; "
          "selective should sit between off and full on time, near full on peak GB "
          "-> the win is reaching larger batch (higher MFU) where off would OOM.")


if __name__ == "__main__":
    main()
