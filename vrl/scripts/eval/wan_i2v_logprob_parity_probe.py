"""Wan 2.1 I2V rollout-vs-replay log-prob parity probe (real weights, 1 GPU).

Why this exists (SPRINT_wan_2_1_i2v_proof_run): the single-GPU GRPO smoke is
structurally blocked on 32 GB — the I2V transformer is 16.4B params (65.6 GB
fp32 shards -> ~32.8 GB bf16), larger than the card, so the trainer-side replay
bundle can never hold it for a backward pass. The generation side, however,
runs fine under ``offload_mode=sequential``. This probe therefore validates the
sprint's §6 hard contract — the only I2V-specific new contract face — on real
weights without a train step:

1. rollout: family ``prepare_sampling`` -> ``forward_step`` -> scheduler step ->
   ``sde_step_with_logprob`` per denoise step, recording old_log_prob exactly
   like ``run_denoise_steps`` (native mode);
2. export ``condition`` + ``image_embeds`` via ``export_replay_tensors``,
   round-trip them through CPU (the trajectory-store wire);
3. replay: ``restore_eval_state`` -> ``forward`` -> ``sde_step_with_logprob``
   with the recorded actions, exactly like ``DiffusionSDELogProbEvaluator``;
4. assert mean |fresh - old| log-prob <= threshold (same 0.01 bar as the
   trainer's ``debug.first_step`` parity check).

The training step itself is family-agnostic (shared GRPO loss/trainer) and is
already run-validated on other families; what it consumes from this family is
precisely the replayed log-prob this probe checks.

Usage (defaults match the single-GPU smoke recipe shapes):
  python -m vrl.scripts.eval.wan_i2v_logprob_parity_probe \
      --image data/external/videophy_i2v_smoke/images/train/000.png \
      --prompt "A wooden spoon stirs the hot soup in the pot."
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vrl.math.denoise.flow_matching import sde_step_with_logprob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    parser.add_argument("--image", default="data/external/videophy_i2v_smoke/images/train/000.png")
    parser.add_argument("--prompt", default="A wooden spoon stirs the hot soup in the pot.")
    parser.add_argument("--num-samples", type=int, default=2, help="GRPO-group-like batch size")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--sde-type", default="cps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", choices=["none", "model", "sequential"], default="sequential")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="mean |fresh-old| log-prob bar (trainer debug.first_step)",
    )
    parser.add_argument("--out-dir", default="outputs/wan_i2v_parity_probe")
    parser.add_argument(
        "--decode-video",
        action="store_true",
        help="also VAE-decode and save the rollout clips for eyeballing",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from PIL import Image

    from vrl.families.registry import get_model_family_entry
    from vrl.generation.types import VideoGenerationRequest

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model_path} (offload={args.offload})", flush=True)
    t0 = time.time()
    entry = get_model_family_entry("wan_2_1_i2v")
    build = entry.resolve_model_build(
        OmegaConf.create(
            {
                "model": {
                    "family": "wan_2_1_i2v",
                    "path": args.model_path,
                    "use_lora": False,
                    "offload_mode": args.offload,
                },
                "sampling": {"num_steps": args.steps},
                "precision": {
                    "float32_precision": "tf32",
                    "training": {"dtype": "bf16"},
                },
            },
        ),
        torch.device("cuda"),
    )
    bundle = entry.build_rollout(build)
    model = bundle.model
    print(f"[load] done in {time.time() - t0:.0f}s", flush=True)

    image = Image.open(args.image).convert("RGB")
    request = VideoGenerationRequest(
        prompt=args.prompt,
        task_type="image_to_video",
        width=args.width,
        height=args.height,
        frame_count=args.frames,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    # ---- rollout (mirrors run_denoise_steps, denoise_mode=native) ----------
    with torch.no_grad():
        encoded = model.encode_prompt(
            [args.prompt] * args.num_samples,
            guidance_scale=args.guidance,
            reference_image=image,
        )
        state = model.prepare_sampling(request, encoded, reference_image=image)

    obs, act, old_lp, ts = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for step_idx in range(len(state.timesteps)):
            latents_ori = state.latents.clone()
            timestep = state.timesteps[step_idx]
            noise_pred = model.forward_step(state, step_idx)["noise_pred"]
            prev_latents = state.scheduler.step(
                noise_pred,
                timestep,
                state.latents,
                return_dict=False,
            )[0]
            sde = sde_step_with_logprob(
                state.scheduler,
                noise_pred.float(),
                timestep.unsqueeze(0),
                state.latents.float(),
                prev_sample=prev_latents.float(),
                noise_level=args.noise_level,
                sde_type=args.sde_type,
                step_index=step_idx,
            )
            state.latents = prev_latents
            obs.append(latents_ori.float().cpu())
            act.append(prev_latents.float().cpu())
            old_lp.append(sde.log_prob.float().cpu())
            ts.append(timestep.detach().float().cpu().expand(args.num_samples).clone())
            print(
                f"[rollout] step {step_idx + 1}/{len(state.timesteps)} "
                f"old_lp[0]={sde.log_prob.reshape(-1)[0].item():.4f} "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                flush=True,
            )
    rollout_s = time.time() - t0

    # ---- export -> CPU wire round-trip (the trajectory-store contract) ------
    replay_tensors = {
        key: (value.detach().float().cpu() if torch.is_tensor(value) else value)
        for key, value in model.export_replay_tensors(state).items()
    }
    replay_tensors["timesteps"] = torch.stack(ts, dim=1)  # [B, S]
    batch_context = model.export_batch_context(state)
    assert batch_context["conditioning"] == "reference_image"
    assert replay_tensors["condition"] is not None, "I2V condition lost in export"

    # ---- replay (mirrors replay_forward + DiffusionSDELogProbEvaluator) -----
    device = torch.device("cuda")
    replay_on_gpu = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in replay_tensors.items()
    }
    diffs, ratios = [], []
    t0 = time.time()
    with torch.no_grad():
        for step_idx in range(len(obs)):
            eval_state = model.restore_eval_state(
                replay_on_gpu,
                batch_context,
                obs[step_idx].to(device),
                step_idx,
            )
            fresh_noise_pred = model.forward(eval_state, 0)["noise_pred"]
            fresh = sde_step_with_logprob(
                state.scheduler,
                fresh_noise_pred.float(),
                replay_on_gpu["timesteps"][:, step_idx],
                obs[step_idx].to(device).float(),
                prev_sample=act[step_idx].to(device).float(),
                noise_level=args.noise_level,
                sde_type=args.sde_type,
            )
            diff = (fresh.log_prob.cpu() - old_lp[step_idx]).abs()
            ratio = torch.exp(fresh.log_prob.cpu() - old_lp[step_idx])
            diffs.append(diff)
            ratios.append(ratio)
            print(
                f"[replay] step {step_idx + 1}/{len(obs)} "
                f"|fresh-old| mean={diff.mean().item():.6f} max={diff.max().item():.6f}",
                flush=True,
            )
    replay_s = time.time() - t0

    all_diff = torch.stack(diffs)
    all_ratio = torch.stack(ratios)
    mean_diff = all_diff.mean().item()
    passed = mean_diff <= args.threshold
    report = {
        "model_path": args.model_path,
        "offload": args.offload,
        "num_samples": args.num_samples,
        "steps": args.steps,
        "shape": f"{args.width}x{args.height}x{args.frames}f",
        "noise_level": args.noise_level,
        "sde_type": args.sde_type,
        "mean_abs_diff": mean_diff,
        "max_abs_diff": all_diff.max().item(),
        "ratio_mean": all_ratio.mean().item(),
        "ratio_max": all_ratio.max().item(),
        "threshold": args.threshold,
        "passed": passed,
        "rollout_seconds": round(rollout_s, 1),
        "replay_seconds": round(replay_s, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (out_dir / "parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.decode_video:
        from diffusers.utils import export_to_video

        with torch.no_grad():
            frames = model.decode_latents(state.latents)  # [B, C, T, H, W]
        video = model.pipeline.video_processor.postprocess_video(frames, output_type="np")
        for i in range(video.shape[0]):
            path = out_dir / f"sample{i:02d}.mp4"
            export_to_video(list(video[i]), str(path), fps=16)
            print(f"[video] {path}", flush=True)

    print(json.dumps(report, indent=2), flush=True)
    print(
        f"[verdict] {'PASS' if passed else 'FAIL'}: mean |fresh-old| log-prob "
        f"{mean_diff:.6f} vs threshold {args.threshold}",
        flush=True,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
