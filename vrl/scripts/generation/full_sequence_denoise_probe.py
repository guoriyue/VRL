"""Family-agnostic rollout probe for registry-descriptor diffusion families.

Builds the family bundle through the registry entry (the same ``build_rollout``
path the worker uses), runs the PRODUCTION
denoise loop — ``forward_step`` + ``sde_step_with_logprob``, the exact math
the generation executor runs — decodes, and reports output statistics. With
``--check-replay`` it additionally runs the first-step logprob parity check in
miniature: restore the eval state from exported replay tensors, recompute the
step-0 forward, and assert the noise prediction and SDE log-prob match the
rollout's.

This is the per-family GPU verification tool for model landings (no Ray, no
trainer):

    python -m vrl.scripts.generation.full_sequence_denoise_probe --family sana \\
        --path Efficient-Large-Model/Sana_1600M_1024px_diffusers \\
        --dtype fp16 --float32-precision ieee --no-outer-autocast \\
        --prompt "a red fox in the snow" --steps 8 --height 512 --width 512 \\
        --check-replay --out /tmp/sana_probe.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def _nonnegative_finite(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def _check_replay_errors(pred_err: float, lp_err: float, *, pred_atol: float, lp_atol: float):
    if not all(math.isfinite(value) for value in (pred_err, lp_err)):
        raise SystemExit("[probe] FAIL: replay errors are not finite")
    if pred_err > pred_atol or lp_err > lp_atol:
        raise SystemExit("[probe] FAIL: replay parity out of tolerance")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--path", required=True, help="checkpoint repo or local dir")
    parser.add_argument("--model-preset", default=None, help="optional existing model YAML preset")
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
        required=True,
        help="rollout role dtype",
    )
    parser.add_argument(
        "--float32-precision",
        choices=["ieee", "tf32"],
        required=True,
        help="FP32 matmul mode",
    )
    parser.add_argument(
        "--outer-autocast",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="whether the shared diffusion boundary applies role-dtype autocast",
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
    parser.add_argument("--replay-noise-atol", type=_nonnegative_finite, default=5e-2)
    parser.add_argument("--replay-logprob-atol", type=_nonnegative_finite, default=5e-1)
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="new directory for conditioning, decoded video/image and result JSON",
    )
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


def _resolve_probe_model_build(args: argparse.Namespace, entry: Any, device: Any) -> Any:
    """Project CLI inputs through the production model-build resolver.

    The explicit public precision policy owns parameter storage, autocast,
    prompt-encoder precision, and quantized rollout behavior. Reusing the
    production resolver keeps the probe and Ray workers on the same contract.
    """

    from dataclasses import replace

    from omegaconf import OmegaConf

    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.dtypes import dtype_to_precision_token, resolve_torch_dtype

    role_precision = dtype_to_precision_token(resolve_torch_dtype(args.dtype))
    precision: dict[str, Any] = {
        "float32_precision": args.float32_precision,
        "training": {
            "dtype": role_precision,
            "outer_autocast": args.outer_autocast,
        },
    }
    if args.quantize is not None:
        precision["rollout"] = {"quantization": {"format": args.quantize}}
    cfg = OmegaConf.create(
        {
            "model": {
                "family": entry.family,
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
    if args.model_preset:
        preset = OmegaConf.load(args.model_preset)
        if not OmegaConf.is_dict(preset) or not OmegaConf.is_dict(preset.get("model")):
            raise ValueError("model preset must contain a model mapping")
        if preset.model.get("family", entry.family) != entry.family:
            raise ValueError("model preset family must match --family")
        cfg.model = OmegaConf.merge(
            cfg.model, preset.model, {"family": entry.family, "path": args.path}
        )
    root = parse_config(cfg)
    precision_policy = PrecisionPolicy.from_section(root.precision)
    build = entry.resolve_model_build(
        root,
        device,
        precision=precision_policy,
    )
    build.rollout = replace(build.require_rollout(), base_weight_sync=False)
    return build


def main() -> None:
    args = _build_arg_parser().parse_args()
    started = time.perf_counter()
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=False)

    import torch

    from vrl.generation.types import DenoiseRequest
    from vrl.math.denoise.flow_matching import sde_step_with_logprob
    from vrl.models.families.registry import get_model_family_entry

    entry = get_model_family_entry(args.family)
    family = entry.family
    if entry.policy_semantics.generation_regime != "full_sequence":
        raise SystemExit(
            f"--family {family} does not expose a full-sequence denoise policy; this probe "
            "drives that execution shape only",
        )
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    build = _resolve_probe_model_build(args, entry, device)
    print(f"[probe] building {family} bundle from {args.path} ...")
    bundle = entry.build_rollout(build)
    model = bundle.model

    encode_kwargs: dict[str, Any] = {"guidance_scale": args.guidance_scale}
    if args.max_sequence_length is not None:
        encode_kwargs["max_sequence_length"] = args.max_sequence_length
    encoded = model.encode_prompt(
        [args.prompt],
        [args.negative_prompt] if args.guidance_scale > 1.0 else None,
        **encode_kwargs,
    )
    if artifact_dir is not None:
        from vrl.trainers.weight_sync import to_cpu_snapshot

        torch.save(
            {
                "family": family,
                "model_path": args.path,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "encoded": to_cpu_snapshot(encoded),
            },
            artifact_dir / "conditioning.pt",
        )
    if args.offload:
        enc = getattr(model.pipeline, "text_encoder", None)
        if enc is not None:
            enc.to("cpu")
            torch.cuda.empty_cache()

    request = DenoiseRequest(
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
    step_records = []
    for step_idx in range(args.steps):
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
        state.latents = sde.prev_sample
        if not bool(torch.isfinite(state.latents).all()) or not bool(
            torch.isfinite(sde.log_prob).all()
        ):
            raise SystemExit(f"[probe] FAIL: nonfinite latents/log-probs at step {step_idx}")
        step_records.append(
            {
                "step": step_idx + 1,
                "logprob": sde.log_prob.mean().item(),
                "latent_std": state.latents.std().item(),
            }
        )
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

    replay_result = None
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
        _check_replay_errors(
            pred_err.item(),
            lp_err.item(),
            pred_atol=args.replay_noise_atol,
            lp_atol=args.replay_logprob_atol,
        )
        replay_result = {"noise_max_abs": pred_err.item(), "logprob_max_abs": lp_err.item()}

    if artifact_dir is not None:
        from vrl.utils.media import write_mp4, write_png

        if image.ndim == 5:
            write_mp4(
                image[0].float().cpu(),
                artifact_dir / "video.mp4",
                fps=getattr(state, "fps", None) or 16,
            )
        else:
            write_png(image[0].float().cpu(), artifact_dir / "image.png")
        result = {
            "passed": True,
            "arguments": vars(args),
            "steps": step_records,
            "decoded_shape": list(image.shape),
            "decoded_finite": finite,
            "decoded_std": image.std().item(),
            "replay": replay_result,
            "seconds_including_load": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        (artifact_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"[probe] PASS: {family} rollout verified")


if __name__ == "__main__":
    main()
