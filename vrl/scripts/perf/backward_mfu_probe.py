"""Backward-pass cost: forward vs backward time, MFU/HFU, and the grad-checkpoint tax.

Training replays the DiT with grad. Backward dominates the update, so it is the
training cost center. Three questions this answers on a real model:

  1. Is backward at peak (high MFU) or launch-bound at small microbatch?
  2. What does gradient_checkpointing actually cost (extra recompute time) and
     save (peak memory)? -> decides whether off-by-default is safe.
  3. What does the production LoRA path (frozen base, adapter-only grads) do to
     the fwd/bwd ratio and MFU? Frozen params skip their weight-grad GEMMs, so
     LoRA backward is ~1x forward FLOPs, not the full-param ~2x.

FLOPs are counted, not derived: FlopCounterMode records the matmul/conv/SDPA ops
the module actually dispatches, so frozen-base LoRA, attention backward cost and
family structure are captured exactly instead of approximated by a 2ND formula.
MFU's numerator is the REQUIRED FLOPs (counted with checkpointing off — recompute
is excluded by definition); HFU's numerator is the EXECUTED FLOPs of the measured
mode (recompute included). The two split the grad-checkpoint tax: full ckpt shows
a lower MFU at a similar HFU.

Run: python -m vrl.scripts.perf.backward_mfu_probe --height 1024 --batches 1 2 4
"""

from __future__ import annotations

import argparse

import torch

from vrl.scripts.perf.common.timing import cuda_mean_ms

# Measure the EXACT production policy, not a copy: selective_checkpoint_func is the
# same SAC helper that enable_transformer_gradient_checkpointing applies to every
# diffusion family, so probe numbers describe what real training actually runs.
# Imported from the trainers module (not the online runner) so the probe stays a
# lightweight perf script and does not pull in Ray/launcher.
from vrl.trainers.activation_checkpointing import selective_checkpoint_func


def _apply_ckpt(tf, mode: str) -> None:
    """off | full | selective — full recomputes every block, selective uses SAC."""
    tf.disable_gradient_checkpointing()
    if mode == "full":
        tf.enable_gradient_checkpointing()
    elif mode == "selective":
        tf.enable_gradient_checkpointing(gradient_checkpointing_func=selective_checkpoint_func)


def _count_flops(fn) -> int:
    """Exact FLOPs of one call: every matmul/conv/SDPA the dispatcher executes.

    FlopCounterMode is a dispatch mode, so backward ops are counted too —
    including checkpoint recompute, which replays real forward ops inside
    backward — and ops that never run (the weight-grad GEMMs of frozen params
    under LoRA) are not.
    """
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument(
        "--peak-tflops",
        type=float,
        default=0.0,
        help="bf16 MFU denominator; 0 = measure real bf16 peak (gpu_preflight). "
        "Pass the datasheet dense bf16 number instead for paper-comparable MFU "
        "(the measured peak is ~80-90%% of datasheet, so measured-MFU reads higher).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the transformer (measures training-side fusion "
        "headroom: backward MFU eager vs compiled). Restricts to ckpt=off/"
        "selective since compile+full block checkpointing recompiles/collides.",
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="attach a fresh PEFT LoRA adapter (frozen base) exactly like "
        "production training does; 0 probes the full-param backward instead.",
    )
    p.add_argument("--lora-alpha", type=int, default=0, help="0 = 2x rank (the wan preset shape)")
    p.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
        help="attention projections — the cross-family experiment preset default",
    )
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
    cfg = tf.config
    if args.lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        # Mirror the production attach (vrl/models/steps/denoise/common/lora.py):
        # frozen base, fresh adapter, no fp32 adapter upcast — so this backward
        # dispatches exactly the GEMM set real LoRA training runs.
        tf.requires_grad_(False)
        tf = get_peft_model(
            tf,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha or 2 * args.lora_rank,
                init_lora_weights="gaussian",
                target_modules=args.lora_target_modules,
            ),
            autocast_adapter_dtype=False,
        )
    eager = tf
    if args.compile:
        # Same transformer object serves rollout AND replay (base.py replay_forward ->
        # forward_step), so compiling it is exactly the training-side fusion the real
        # recipe gets when model.torch_compile.enable=true. Backward graph is compiled
        # too. mode=default to avoid the CUDA-graph/PEFT collisions noted for Wan.
        tf = torch.compile(tf, mode="default")
    n_params = sum(p.numel() for p in eager.parameters())
    n_trainable = sum(p.numel() for p in eager.parameters() if p.requires_grad)
    text_tok = 333  # SD3 joint sequence text length
    h, w = args.height // 8, args.width // 8
    seq = (h // 2) * (w // 2) + text_tok

    def _inputs(b: int):
        # Random embeds of the right shape — avoids loading the text encoders to GPU.
        emb = torch.randn(b, text_tok, cfg.joint_attention_dim, device=device, dtype=dtype)
        pol = torch.randn(b, cfg.pooled_projection_dim, device=device, dtype=dtype)
        latents = torch.randn(b, cfg.in_channels, h, w, device=device, dtype=dtype)
        pipe.scheduler.set_timesteps(28, device=device)
        ts = pipe.scheduler.timesteps[14].expand(b)
        return emb, pol, latents, ts

    def _fwd_fn(module, emb, pol, latents, ts):
        def _fwd():
            with torch.no_grad():
                module(
                    hidden_states=latents,
                    timestep=ts,
                    encoder_hidden_states=emb,
                    pooled_projections=pol,
                    return_dict=False,
                )

        return _fwd

    def _step_fn(module, emb, pol, latents, ts):
        # No CFG in training replay here; one row per sample.
        def _step():
            module.zero_grad(set_to_none=True)
            out = module(
                hidden_states=latents,
                timestep=ts,
                encoder_hidden_states=emb,
                pooled_projections=pol,
                return_dict=False,
            )[0]
            out.float().pow(2).mean().backward()
            del out

        return _step

    # MFU numerator: required fwd+bwd FLOPs, counted once from the real module
    # with checkpointing OFF (recompute is MFU waste; it lands in HFU instead).
    # Counted at b=1 and b=2 and required to match exactly, so b * per_sample is
    # exact for every probed batch — including batches whose ckpt=off run OOMs.
    _apply_ckpt(eager, "off")
    per_sample_flops = _count_flops(_step_fn(eager, *_inputs(1)))
    try:
        doubled_flops = _count_flops(_step_fn(eager, *_inputs(2)))
    except torch.OutOfMemoryError:
        # The b=2 probe VALIDATES linearity; it is not a prerequisite. b=1 rows
        # need no scaling at all, and when b=2 does not fit at this shape the
        # loop below counts any larger batch directly instead of scaling.
        doubled_flops = None
        print("note: b=2 ckpt=off does not fit; larger batches count FLOPs directly")
    if doubled_flops is not None and doubled_flops != 2 * per_sample_flops:
        raise RuntimeError(
            f"step FLOPs are not linear in batch (b=1: {per_sample_flops}, "
            f"b=2: {doubled_flops}); refusing to scale — count each batch directly",
        )
    eager.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    print(
        f"model {n_params / 1e9:.2f}B ({n_trainable / 1e6:.1f}M trainable), seq={seq}, "
        f"fwd+bwd {per_sample_flops / 1e12:.2f} TFLOPs/sample, "
        f"peak~{args.peak_tflops:.0f} TFLOPS bf16"
    )
    print(
        f"\n{'batch':>5} | {'ckpt':>9} | {'fwd ms':>7} | {'bwd ms':>7} | "
        f"{'bwd/fwd':>7} | {'MFU':>5} | {'HFU':>5} | {'peak GB':>7}"
    )

    for b in args.batches:
        emb, pol, latents, ts = _inputs(b)
        if doubled_flops is None and b > 1:
            # Linearity unvalidated at this shape: count this batch directly
            # (guarded — a batch that cannot even be counted gets no MFU).
            _apply_ckpt(eager, "off")
            try:
                mfu_flops = float(_count_flops(_step_fn(eager, emb, pol, latents, ts)))
            except torch.OutOfMemoryError:
                mfu_flops = None
                torch.cuda.empty_cache()
        else:
            mfu_flops = float(b * per_sample_flops)

        # Eager measures all three. Under --compile we skip full (compile + full
        # block checkpointing recompiles/collides) but DO measure selective: SAC
        # goes through AOTAutograd's min-cut partitioner, so compiled x selective
        # is the P1 question — does it compose and beat compiled x off at a larger
        # batch? (off / selective only; errors are caught and printed per row.)
        ckpt_modes = ("off", "selective") if args.compile else ("off", "full", "selective")
        for ckpt in ckpt_modes:
            _apply_ckpt(tf, ckpt)
            try:
                # HFU numerator: executed FLOPs of THIS mode (recompute included).
                # Counted on the eager module — a dispatch mode cannot see inside
                # compiled kernels, and fusion does not change the GEMM/SDPA set.
                hfu_flops = float(_count_flops(_step_fn(eager, emb, pol, latents, ts)))
                warmup = 6 if args.compile else 2  # inductor needs more warmup per shape
                fwd_ms = cuda_mean_ms(_fwd_fn(tf, emb, pol, latents, ts), iters=4, warmup=warmup)
                torch.cuda.reset_peak_memory_stats()
                step_ms = cuda_mean_ms(_step_fn(tf, emb, pol, latents, ts), iters=4, warmup=warmup)
                peak = torch.cuda.max_memory_allocated() / 1024**3
                bwd_ms = max(step_ms - fwd_ms, 0.01)
                hfu = hfu_flops / (step_ms / 1e3) / 1e12 / args.peak_tflops * 100
                if mfu_flops is None:
                    mfu_cell = "   —"
                else:
                    mfu = mfu_flops / (step_ms / 1e3) / 1e12 / args.peak_tflops * 100
                    mfu_cell = f"{mfu:>4.0f}%"
                print(
                    f"{b:>5} | {ckpt:>9} | {fwd_ms:>7.1f} | {bwd_ms:>7.1f} | "
                    f"{bwd_ms / fwd_ms:>7.2f} | {mfu_cell} | {hfu:>4.0f}% | {peak:>7.2f}"
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                print(f"{b:>5} | {ckpt:>9} | OOM/err ({type(exc).__name__})")
            finally:
                tf.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    print(
        "\nreads: bwd/fwd ~2 expected full-param, ~1 under --lora-rank (frozen "
        "base skips weight-grad GEMMs). HFU-MFU gap = recompute tax. If MFU low "
        "at batch=1 -> launch-bound (batch up / compile helps). full = lower "
        "peak GB but recompute tax; selective should sit between off and full "
        "on time, near full on peak GB -> the win is reaching larger batch "
        "(higher MFU) where off would OOM."
    )


if __name__ == "__main__":
    main()
