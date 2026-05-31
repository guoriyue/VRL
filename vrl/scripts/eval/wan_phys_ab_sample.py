"""A/B sample 5 prompts with baseline vs LoRA-trained Wan 2.1 1.3B.

Two passes:
  pass=baseline -> no LoRA
  pass=lora     -> LoRA from --lora-path

Same prompts, same seed per prompt, same denoise config.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

PROMPTS = [
    "A whisk spins in the egg mixture, mixing it thoroughly.",
    "A spoon scoops creamy soup from a pot.",
    "Water dripping from a faucet into a sink.",
    "Frog leaping from one lilypad to another.",
    "Whipped cream pouring onto hot chocolate.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pass-name",
        required=True,
        help="Subfolder name under --out-dir (e.g. baseline, lora_50ep, best100)",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--num-frames", type=int, default=33)
    args = parser.parse_args()

    out = Path(args.out_dir) / args.pass_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{args.pass_name}] loading {args.model_id}", flush=True)
    pipe = WanPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    if args.lora_path:
        # The training pipeline saves via PEFT (adapter_config.json +
        # adapter_model.safetensors). Diffusers' load_lora_adapter expects
        # `pytorch_lora_weights.bin/.safetensors`, so we wrap with PeftModel
        # directly the same way training does (see wan_2_1/model.py:apply_lora).
        from peft import PeftModel

        print(f"[lora] loading PEFT adapter from {args.lora_path}", flush=True)
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            args.lora_path,
            is_trainable=False,
        )
        pipe.transformer.set_adapter("default")

    manifest = []
    for i, prompt in enumerate(PROMPTS):
        seed = i  # fixed seed per slot — baseline and lora share it
        gen = torch.Generator(device="cuda").manual_seed(seed)
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="",
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        )
        elapsed = time.time() - t0
        video_path = out / f"prompt{i:02d}_seed{seed}.mp4"
        export_to_video(result.frames[0], str(video_path), fps=8)
        print(
            f"[{args.pass_name}] prompt {i}: {prompt[:50]}... -> "
            f"{video_path.name} ({elapsed:.1f}s)",
            flush=True,
        )
        manifest.append({
            "index": i,
            "prompt": prompt,
            "seed": seed,
            "video": str(video_path),
            "elapsed_sec": elapsed,
        })

    with open(out / "manifest.json", "w") as f:
        json.dump(
            {
                "pass": args.pass_name,
                "model_id": args.model_id,
                "lora_path": args.lora_path,
                "num_steps": args.num_steps,
                "guidance_scale": args.guidance_scale,
                "size": [args.width, args.height, args.num_frames],
                "videos": manifest,
            },
            f,
            indent=2,
        )
    print(f"[{args.pass_name}] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
