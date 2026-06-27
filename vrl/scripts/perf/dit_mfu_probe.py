"""Is the DiT forward at peak tensor-core throughput, or leaving headroom?

batch-scaling tells you launch- vs compute-bound; it does NOT tell you MFU
(achieved FLOPs / peak FLOPs). A DiT can be "compute-bound" yet sit at 30-50%
MFU because the non-matmul ops (norm/softmax/SiLU/AdaLN/elementwise) are
bandwidth-bound on CUDA cores and serialize with the tensor-core GEMMs, leaving
tensor cores idle. This probe measures:

  - achieved TFLOPS and approx MFU of one eager forward
  - the torch.compile speedup (fusion headroom: how much of the slack is the
    serialized bandwidth-bound work that can overlap/fuse away)
  - which scaled-dot-product-attention backend is actually running

Run: python -m vrl.scripts.perf.dit_mfu_probe --height 1024 --batch 4
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
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--peak-tflops", type=float, default=419.0,
                   help="RTX 5090 dense bf16 tensor-core peak (~419) for MFU.")
    p.add_argument("--compile-mode", default="default",
                   help="torch.compile mode: 'default' (inductor fusion) or "
                        "'reduce-overhead' (adds CUDA graphs — captures kernel-launch "
                        "overhead; helps launch-bound, marginal for compute-bound).")
    args = p.parse_args()

    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    tf = pipe.transformer
    device, dtype = tf.device, tf.dtype

    n_params = sum(p.numel() for p in tf.parameters())
    cfg = tf.config
    n_layers = cfg.num_layers
    d_model = cfg.num_attention_heads * cfg.attention_head_dim

    embeds, neg, pooled, neg_pooled = pipe.encode_prompt(
        prompt="a photorealistic astronaut riding a horse on mars",
        prompt_2=None, prompt_3=None, do_classifier_free_guidance=True, device=device,
    )
    b = args.batch
    full_embeds = torch.cat([neg, embeds]).repeat_interleave(b, dim=0)
    full_pooled = torch.cat([neg_pooled, pooled]).repeat_interleave(b, dim=0)
    text_tok = embeds.shape[1]
    h, w = args.height // 8, args.width // 8
    img_tok = (h // 2) * (w // 2)            # SD3 patch size 2
    seq = img_tok + text_tok                  # joint attention sequence length
    rows = 2 * b                              # CFG doubles the batch

    latents = torch.randn(b, cfg.in_channels, h, w, device=device, dtype=dtype)
    model_in = torch.cat([latents] * 2)
    ts = pipe.scheduler.timesteps[0].expand(rows) if len(
        pipe.scheduler.timesteps) else torch.zeros(rows, device=device)
    pipe.scheduler.set_timesteps(28, device=device)
    ts = pipe.scheduler.timesteps[14].expand(rows)

    def _fwd(model):
        with torch.no_grad():
            model(hidden_states=model_in, timestep=ts,
                  encoder_hidden_states=full_embeds, pooled_projections=full_pooled,
                  return_dict=False)

    # FLOPs of one forward call (the whole CFG batch of `rows`):
    #   linear/proj  ~ 2 * params * tokens
    #   attention    ~ n_layers * 2 (QK^T, PV) * 2 (mul+add) * seq^2 * d_model
    flops_linear = 2 * n_params * seq * rows
    flops_attn = n_layers * 2 * 2 * (seq * seq) * d_model * rows
    flops = flops_linear + flops_attn

    print(f"model: {n_params/1e9:.2f}B params, {n_layers} layers, d_model={d_model}")
    print(f"tokens: img={img_tok} + text={text_tok} = {seq}, CFG rows={rows}")
    print(f"approx FLOPs/forward = {flops/1e12:.1f} TFLOP "
          f"(linear {flops_linear/1e12:.1f} + attn {flops_attn/1e12:.1f})")

    # backend in use
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: F401
    print(f"\nSDPA available: flash={torch.backends.cuda.flash_sdp_enabled()}, "
          f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
          f"math={torch.backends.cuda.math_sdp_enabled()}")

    eager_ms = _cuda_time(lambda: _fwd(tf), iters=10)
    eager_tflops = flops / (eager_ms / 1e3) / 1e12
    print("\n=== eager ===")
    print(f"  {eager_ms:.2f} ms/forward  ->  {eager_tflops:.0f} TFLOPS achieved  "
          f"->  MFU ~ {eager_tflops/args.peak_tflops*100:.0f}%  (peak~{args.peak_tflops:.0f})")

    print(f"\ncompiling (inductor, mode={args.compile_mode!r}, ~1-3 min) ...")
    compiled = torch.compile(tf, mode=args.compile_mode)
    # reduce-overhead (CUDA graphs) needs more warmup to capture/replay the graph.
    comp_ms = _cuda_time(lambda: _fwd(compiled), iters=10, warmup=8)
    comp_tflops = flops / (comp_ms / 1e3) / 1e12
    print(f"=== torch.compile (mode={args.compile_mode}) ===")
    print(f"  {comp_ms:.2f} ms/forward  ->  {comp_tflops:.0f} TFLOPS  "
          f"->  MFU ~ {comp_tflops/args.peak_tflops*100:.0f}%")
    print(f"\nfusion headroom (compile speedup) = {eager_ms/comp_ms:.2f}x")
    if eager_ms / comp_ms > 1.2:
        print("  -> real slack: bandwidth-bound non-matmul work was serializing; "
              "fusion reclaims it. You were NOT at tensor-core peak.")
    else:
        print("  -> little slack: GEMMs already dominate; near matmul-bound.")


if __name__ == "__main__":
    main()
