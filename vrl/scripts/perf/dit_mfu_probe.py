"""Screen DiT throughput with analytic MFU and a measured compile A/B.

Batch scaling screens amortization; it does not prove launch- or compute-bound
behavior, and it does NOT tell you MFU
(achieved FLOPs / peak FLOPs). A DiT can be "compute-bound" yet sit at 30-50%
MFU because the non-matmul ops (norm/softmax/SiLU/AdaLN/elementwise) are
bandwidth-bound on CUDA cores and serialize with the tensor-core GEMMs, leaving
tensor cores idle. This probe measures:

  - achieved TFLOPS and approx MFU of one eager forward
  - the measured torch.compile speedup for this shape
  - which scaled-dot-product-attention backends are enabled

Run: python -m vrl.scripts.perf.dit_mfu_probe --height 1024 --batch 4
"""

from __future__ import annotations

import argparse

import torch

from vrl.scripts.perf.common.timing import cuda_mean_ms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument(
        "--peak-tflops",
        type=float,
        default=0.0,
        help="bf16 MFU denominator; 0 = measure this GPU's real bf16 peak "
        "(gpu_preflight). The 419 'RTX 5090 peak' is fp8/sparse — the real "
        "bf16 dense peak is ~232, and using 419 mis-diagnoses MFU by ~1.8x.",
    )
    p.add_argument(
        "--compile-mode",
        default="default",
        help="torch.compile mode: 'default' (inductor fusion) or "
        "'reduce-overhead' (adds CUDA graphs — captures kernel-launch "
        "overhead; helps launch-bound, marginal for compute-bound).",
    )
    args = p.parse_args()

    if args.peak_tflops <= 0:
        from vrl.scripts.perf.gpu_preflight import measured_bf16_peak_tflops

        args.peak_tflops = measured_bf16_peak_tflops()
        print(f"measured bf16 peak (MFU denominator) = {args.peak_tflops:.0f} TFLOPS")

    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(
        "cuda"
    )
    pipe.set_progress_bar_config(disable=True)
    tf = pipe.transformer
    device, dtype = tf.device, tf.dtype

    n_params = sum(p.numel() for p in tf.parameters())
    cfg = tf.config
    n_layers = cfg.num_layers
    d_model = cfg.num_attention_heads * cfg.attention_head_dim

    embeds, neg, pooled, neg_pooled = pipe.encode_prompt(
        prompt="a photorealistic astronaut riding a horse on mars",
        prompt_2=None,
        prompt_3=None,
        do_classifier_free_guidance=True,
        device=device,
    )
    b = args.batch
    full_embeds = torch.cat([neg, embeds]).repeat_interleave(b, dim=0)
    full_pooled = torch.cat([neg_pooled, pooled]).repeat_interleave(b, dim=0)
    text_tok = embeds.shape[1]
    h, w = args.height // 8, args.width // 8
    img_tok = (h // 2) * (w // 2)  # SD3 patch size 2
    seq = img_tok + text_tok  # joint attention sequence length
    rows = 2 * b  # CFG doubles the batch

    latents = torch.randn(b, cfg.in_channels, h, w, device=device, dtype=dtype)
    model_in = torch.cat([latents] * 2)
    pipe.scheduler.set_timesteps(28, device=device)
    ts = pipe.scheduler.timesteps[14].expand(rows)

    def _fwd(model):
        with torch.no_grad():
            model(
                hidden_states=model_in,
                timestep=ts,
                encoder_hidden_states=full_embeds,
                pooled_projections=full_pooled,
                return_dict=False,
            )

    # FLOPs of one forward call (the whole CFG batch of `rows`):
    #   linear/proj  ~ 2 * params * tokens
    #   attention    ~ n_layers * 2 (QK^T, PV) * 2 (mul+add) * seq^2 * d_model
    flops_linear = 2 * n_params * seq * rows
    flops_attn = n_layers * 2 * 2 * (seq * seq) * d_model * rows
    flops = flops_linear + flops_attn

    print(f"model: {n_params / 1e9:.2f}B params, {n_layers} layers, d_model={d_model}")
    print(f"tokens: img={img_tok} + text={text_tok} = {seq}, CFG rows={rows}")
    print(
        f"approx FLOPs/forward = {flops / 1e12:.1f} TFLOP "
        f"(linear {flops_linear / 1e12:.1f} + attn {flops_attn / 1e12:.1f})"
    )

    print(
        f"\nSDPA available: flash={torch.backends.cuda.flash_sdp_enabled()}, "
        f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
        f"math={torch.backends.cuda.math_sdp_enabled()}"
    )

    eager_ms = cuda_mean_ms(lambda: _fwd(tf), iters=10)
    eager_tflops = flops / (eager_ms / 1e3) / 1e12
    print("\n=== eager ===")
    print(
        f"  {eager_ms:.2f} ms/forward  ->  {eager_tflops:.0f} TFLOPS achieved  "
        f"->  MFU ~ {eager_tflops / args.peak_tflops * 100:.0f}%  (peak~{args.peak_tflops:.0f})"
    )

    print(f"\ncompiling (inductor, mode={args.compile_mode!r}, ~1-3 min) ...")
    compiled = torch.compile(tf, mode=args.compile_mode)
    # reduce-overhead (CUDA graphs) needs more warmup to capture/replay the graph.
    comp_ms = cuda_mean_ms(lambda: _fwd(compiled), iters=10, warmup=8)
    comp_tflops = flops / (comp_ms / 1e3) / 1e12
    print(f"=== torch.compile (mode={args.compile_mode}) ===")
    print(
        f"  {comp_ms:.2f} ms/forward  ->  {comp_tflops:.0f} TFLOPS  "
        f"->  MFU ~ {comp_tflops / args.peak_tflops * 100:.0f}%"
    )
    print(f"\nmeasured compile speedup = {eager_ms / comp_ms:.2f}x")
    if eager_ms / comp_ms > 1.2:
        print(
            "  -> compile produced a material gain at this shape; this does not "
            "classify the main GEMM's tensor-pipe saturation"
        )
    else:
        print(
            "  -> compile produced little gain at this shape; use NCU against a "
            "same-machine GEMM baseline for a saturation conclusion"
        )


if __name__ == "__main__":
    main()
