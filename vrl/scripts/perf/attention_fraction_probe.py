"""Where does the DiT spend FLOPs — linear (MLP/proj) vs attention — as resolution grows?

WHY: the lossless-research report (docs/sprints/reading/SPRINT_lossless_diffusion_rl_research.md)
says FlashAttention-3 is an EXACT compute lever, but attention is O(seq^2) while the
linear/MLP work is O(seq). At image 1024^2 attention is a small FLOP slice, so an
attention-kernel upgrade (FA-3) has a small ceiling; at video token counts (5D latent,
many frames) attention can overtake linear and FA-3 becomes worth it. This probe finds
the CROSSOVER: the sequence length / resolution where attention FLOPs >= linear FLOPs,
and times the eager forward so the analytic fraction is grounded in real ms.

Lossless only: pure measurement, changes nothing.

Run: python -m vrl.scripts.perf.attention_fraction_probe --model stabilityai/stable-diffusion-3.5-medium
"""

from __future__ import annotations

import argparse

import torch


def _cuda_time(fn, iters: int, warmup: int = 3) -> float:
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
    p.add_argument("--batch", type=int, default=1)
    # image side lengths to sweep; the last few stand in for video token counts
    # (a 1024^2 frame is 4096 img tokens; a short video clip is several x that).
    p.add_argument("--sides", type=int, nargs="+", default=[512, 768, 1024, 1536, 2048])
    p.add_argument("--iters", type=int, default=8)
    args = p.parse_args()

    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    tf = pipe.transformer
    device, dtype = tf.device, tf.dtype
    cfg = tf.config
    n_params = sum(q.numel() for q in tf.parameters())
    n_layers = cfg.num_layers
    d_model = cfg.num_attention_heads * cfg.attention_head_dim

    embeds, neg, pooled, neg_pooled = pipe.encode_prompt(
        prompt="a photorealistic astronaut riding a horse on mars",
        prompt_2=None, prompt_3=None, do_classifier_free_guidance=True, device=device,
    )
    text_tok = embeds.shape[1]
    b = args.batch
    full_embeds = torch.cat([neg, embeds]).repeat_interleave(b, dim=0)
    full_pooled = torch.cat([neg_pooled, pooled]).repeat_interleave(b, dim=0)
    rows = 2 * b
    pipe.scheduler.set_timesteps(28, device=device)
    ts = pipe.scheduler.timesteps[14].expand(rows)

    print(f"model {n_params/1e9:.2f}B, {n_layers} layers, d_model={d_model}, text_tok={text_tok}")
    print(f"\n{'side':>5} | {'img_tok':>7} | {'seq':>6} | {'lin TFLOP':>9} | {'attn TFLOP':>10} | "
          f"{'attn%':>5} | {'ms/fwd':>7} | crossover")
    for side in args.sides:
        h, w = side // 8, side // 8
        img_tok = (h // 2) * (w // 2)  # SD3 patch size 2
        seq = img_tok + text_tok
        flops_linear = 2 * n_params * seq * rows
        flops_attn = n_layers * 2 * 2 * (seq * seq) * d_model * rows
        attn_frac = flops_attn / (flops_linear + flops_attn)
        latents = torch.randn(b, cfg.in_channels, h, w, device=device, dtype=dtype)
        model_in = torch.cat([latents] * 2)

        def _fwd(model_in=model_in, ts=ts):
            with torch.no_grad():
                tf(hidden_states=model_in, timestep=ts, encoder_hidden_states=full_embeds,
                   pooled_projections=full_pooled, return_dict=False)
        try:
            ms = _cuda_time(_fwd, args.iters)
            ms_s = f"{ms:7.1f}"
        except torch.cuda.OutOfMemoryError:
            ms_s = "    OOM"
            torch.cuda.empty_cache()
        tag = "ATTN-DOMINATED" if attn_frac >= 0.5 else ("attn rising" if attn_frac >= 0.25 else "linear-bound")
        print(f"{side:>5} | {img_tok:>7} | {seq:>6} | {flops_linear/1e12:>9.1f} | "
              f"{flops_attn/1e12:>10.1f} | {attn_frac*100:>4.0f}% | {ms_s} | {tag}")

    print("\nread: attn% < 25% -> FA-3/attention-kernel upgrade has a small ceiling (linear-bound);")
    print("      attn% >= 50% -> attention dominates (high-res / video token counts) -> FA-3 worth it.")
    print("      The lossless compute lever shifts from MLP-fusion (compile) to attention (FA-3) as seq grows.")


if __name__ == "__main__":
    main()
