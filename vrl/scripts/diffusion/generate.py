"""Family-agnostic rollout probe for registry-descriptor diffusion families.

Builds the family bundle straight from the registry descriptor (the same
``build_family_runtime_bundle`` path the worker uses), runs the PRODUCTION
denoise loop — ``forward_step`` + ``sde_step_with_logprob``, the exact math
the generation executor runs — decodes, and reports output statistics. With
``--check-replay`` it additionally runs the first-step logprob parity check in
miniature: restore the eval state from exported replay tensors, recompute the
step-0 forward, and assert the noise prediction and SDE log-prob match the
rollout's.

This is the per-family GPU verification tool for model landings (no Ray, no
trainer):

    python -m vrl.scripts.diffusion.generate --family sana \\
        --path Efficient-Large-Model/Sana_1600M_1024px_diffusers \\
        --prompt "a red fox in the snow" --steps 8 --height 512 --width 512 \\
        --check-replay --out /tmp/sana_probe.png
"""

from __future__ import annotations

import argparse
from typing import Any


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--path", required=True, help="checkpoint repo or local dir")
    parser.add_argument("--prompt", default="a photo of a red fox sitting in fresh snow")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="outer rollout autocast dtype; family-owned parameter storage still wins",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda/cpu (default: cuda if available). cpu lets a >VRAM model "
        "verify rollout correctness slowly",
    )
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--out", default=None, help="save first output image/frame here")
    parser.add_argument(
        "--sde-type",
        default="flow_grpo",
        choices=["flow_grpo", "cps", "ddim"],
        help="log-prob math family (ddim for alphas-ladder checkpoints)",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        choices=["fp8", "nvfp4"],
        help="selective rollout GEMM quantization",
    )
    parser.add_argument("--check-replay", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="ODE sampling (no SDE noise) — matches the reference pipeline output",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="move the text encoder to CPU after encode (large-family probes)",
    )
    return parser


def _resolve_probe_model_build(args: argparse.Namespace, family: str, device: Any) -> Any:
    """Project CLI inputs through the production model-build resolver.

    The family descriptor owns parameter storage while the public precision
    policy owns autocast, prompt-encoder precision, and quantized rollout behavior.
    Reusing that boundary keeps this direct probe from growing a second dtype
    policy that can drift from Ray workers.
    """

    from dataclasses import replace

    from omegaconf import OmegaConf

    from vrl.models.diffusion.build import resolve_family_model_build
    from vrl.models.dtypes import dtype_to_precision_token, resolve_torch_dtype

    autocast_precision = dtype_to_precision_token(resolve_torch_dtype(args.dtype))
    precision: dict[str, Any] = {"training": {"dtype": autocast_precision}}
    if args.quantize is not None:
        precision["rollout"] = {"quantization": {"format": args.quantize}}
    cfg = OmegaConf.create(
        {
            "model": {
                "family": family,
                "path": args.path,
                "use_lora": False,
                # Probe-scale decode safety: tiled/sliced VAE decode keeps the
                # decode inside whatever VRAM the resident transformer left over.
                "memory": {"vae_decode": {"tiling": True, "slicing": True}},
            },
            "sampling": {"num_steps": args.steps},
            "precision": precision,
        },
    )
    build = resolve_family_model_build(cfg, device)
    build.rollout = replace(build.require_rollout(), base_weight_sync=False)
    return build


def main() -> None:
    args = _build_arg_parser().parse_args()

    import torch

    from vrl.generation.diffusion.layout import VideoGenerationRequest
    from vrl.math.diffusion.flow_matching import sde_step_with_logprob
    from vrl.models.diffusion.build import build_family_runtime_bundle
    from vrl.rollouts.families.registry import (
        get_rollout_family_entry,
        normalize_rollout_family,
    )

    family = normalize_rollout_family(args.family)
    entry = get_rollout_family_entry(family)
    if entry.capability.trajectory_kind != "diffusion":
        raise SystemExit(
            f"--family {family} is an AR family; this probe drives the "
            "diffusion denoise loop only (use the AR entrypoints instead)",
        )
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    build = _resolve_probe_model_build(args, family, device)
    print(f"[probe] building {family} bundle from {args.path} ...")
    bundle = build_family_runtime_bundle(build)
    model = bundle.model

    encode_kwargs: dict[str, Any] = {"guidance_scale": args.guidance_scale}
    if args.max_sequence_length is not None:
        encode_kwargs["max_sequence_length"] = args.max_sequence_length
    encoded = model.encode_prompt(
        [args.prompt],
        [args.negative_prompt] if args.guidance_scale > 1.0 else None,
        **encode_kwargs,
    )
    if args.offload:
        enc = getattr(model.pipeline, "text_encoder", None)
        if enc is not None:
            enc.to("cpu")
            torch.cuda.empty_cache()

    request = VideoGenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        frame_count=args.frames,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    state = model.prepare_sampling(request, encoded)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # The probe verifies the forward rollout only — no backward pass — so the
    # whole loop runs under inference_mode (the trainable transformer would
    # otherwise accumulate 8 steps of activation graphs and OOM at decode).
    ctx = torch.inference_mode()
    ctx.__enter__()
    first_step: dict[str, Any] = {}
    logprobs = []
    for step_idx in range(args.steps):
        with torch.amp.autocast(
            device.type,
            dtype=model.autocast_dtype,
            enabled=(device.type == "cuda" and model.autocast_dtype != torch.float32),
        ):
            out = model.forward_step(state, step_idx)
        timestep = state.timesteps[step_idx]
        sde = sde_step_with_logprob(
            state.scheduler,
            out["noise_pred"].float(),
            timestep.unsqueeze(0),
            state.latents.float(),
            generator=None if args.deterministic else generator,
            deterministic=args.deterministic,
            sde_type=args.sde_type,
            step_index=step_idx,
        )
        if step_idx == 0 and args.check_replay:
            first_step = {
                "latents": state.latents.detach().clone(),
                "noise_pred": out["noise_pred"].detach().clone(),
                "prev_sample": sde.prev_sample.detach().clone(),
                "log_prob": sde.log_prob.detach().clone(),
                "timestep": timestep.detach().clone(),
            }
        logprobs.append(sde.log_prob.detach())
        state.latents = sde.prev_sample
        print(
            f"[probe] step {step_idx + 1}/{args.steps} "
            f"logprob={sde.log_prob.mean().item():.2f} "
            f"latent_std={state.latents.std().item():.4f}",
        )

    image = model.decode_latents(state.latents)
    finite = bool(torch.isfinite(image).all())
    print(
        f"[probe] decoded output shape={tuple(image.shape)} "
        f"range=[{image.min().item():.3f}, {image.max().item():.3f}] "
        f"std={image.std().item():.4f} finite={finite}",
    )
    if not finite or image.std().item() < 1e-3:
        raise SystemExit("[probe] FAIL: output is not finite or has ~zero variance")

    if args.out:
        from torchvision.utils import save_image

        frame = image
        if frame.dim() == 5:  # [B, C, T, H, W] -> middle frame
            frame = frame[:, :, frame.shape[2] // 2]
        save_image(frame[0].float().cpu().clamp(0, 1), args.out)
        print(f"[probe] saved {args.out}")

    if args.check_replay:
        # Embeds are step-invariant, so exporting from the final state gives
        # the same replay tensors step 0 saw.
        replay_tensors = dict(model.export_replay_tensors(state))
        batch_context = model.export_batch_context(state)
        bsz = first_step["latents"].shape[0]
        replay_tensors["timesteps"] = first_step["timestep"].expand(bsz).clone()
        restored = model.restore_eval_state(
            replay_tensors,
            batch_context,
            first_step["latents"],
            0,
        )
        with torch.amp.autocast(
            device.type,
            dtype=model.autocast_dtype,
            enabled=(device.type == "cuda" and model.autocast_dtype != torch.float32),
        ):
            out = model.forward_step(restored, 0)
        pred_err = (out["noise_pred"].float() - first_step["noise_pred"].float()).abs().max()
        sde = sde_step_with_logprob(
            state.scheduler,
            out["noise_pred"].float(),
            first_step["timestep"].unsqueeze(0),
            first_step["latents"].float(),
            prev_sample=first_step["prev_sample"].float(),
            sde_type=args.sde_type,
            step_index=0,
        )
        lp_err = (sde.log_prob - first_step["log_prob"]).abs().max()
        print(
            f"[probe] replay parity: noise_pred max_err={pred_err.item():.3e} "
            f"logprob max_err={lp_err.item():.3e}",
        )
        if pred_err.item() > 5e-2 or lp_err.item() > 5e-1:
            raise SystemExit("[probe] FAIL: replay parity out of tolerance")

    print(f"[probe] PASS: {family} rollout verified")


if __name__ == "__main__":
    main()
