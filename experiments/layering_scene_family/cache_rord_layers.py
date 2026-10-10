"""Cache RORD 2-layer samples for the multi-scene supervised trainer.

Frames: composite, background (real truth), foreground. Token weights: the
background frame's tokens under the foreground (``--occluder``: the coarse
polygon, or the foreground layer's own alpha for difference-matte samples)
carry a quarter of the loss (the occluded real background), the foreground
frame is weighted ``--foreground-weight`` (0.2 for coarse polygons), the rest 1;
mean one. Prompt embeds are reused from a scene-family cache.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vrl.scripts.generation.qwen_layered_rl_probe import load_upstream

TOKENS = 1600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--foreground-weight", type=float, default=0.2)
    parser.add_argument("--occluder", choices=("polygon", "matte"), default="polygon")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    upstream = load_upstream(Path(plan["upstream"]))
    prompt = torch.load(args.prompt_cache, map_location="cpu", weights_only=True)
    from diffusers import AutoencoderKLQwenImage

    vae = (
        AutoencoderKLQwenImage.from_pretrained(
            plan["base_path"], subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .cuda()
    )
    vae.requires_grad_(False)
    done = 0
    for d in sorted(
        p for split in ("train", "heldout") for p in (args.root / split).glob("*") if p.is_dir()
    ):
        if (d / "cache.pt").exists():
            continue

        def enc(image):
            x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
            with torch.no_grad():
                return upstream.encode_condition_image(vae, x[None].cuda()).cpu().float()

        source = Image.open(d / "source.png").convert("RGB")
        condition = enc(source).to(torch.bfloat16)
        frames = [enc(Image.open(d / "source.png").convert("RGBA"))] + [
            enc(Image.open(d / f"layer_{i}.png").convert("RGBA")) for i in range(2)
        ]
        target = torch.cat(frames, dim=1)
        if args.occluder == "matte":
            fg = torch.tensor(
                np.asarray(Image.open(d / "layer_1.png").convert("RGBA"))[..., 3] > 127,
                dtype=torch.float32,
            )[None, None]
        else:
            fg = torch.tensor(
                np.asarray(Image.open(d / "foreground_mask.png").convert("L")) > 127,
                dtype=torch.float32,
            )[None, None]
        coverage = F.interpolate(fg, size=(40, 40), mode="area")[0, 0].reshape(-1)
        weights = torch.ones(3, TOKENS)
        weights[2] = args.foreground_weight
        mass = float(coverage.sum())
        k = min(80.0, 0.25 * float(weights.sum()) / (0.75 * mass)) if mass else 0.0
        weights[1] += k * coverage
        weights = (weights / weights.mean()).reshape(1, 3 * TOKENS, 1)
        torch.save(
            {
                "condition": condition,
                "prompt": prompt["prompt"],
                "mask": prompt["mask"],
                "prompt_text": prompt["prompt_text"],
                "width": 640,
                "height": 640,
                "layers": 2,
                "target_latents": target,
                "weights_weighted": weights,
                "weights_control": weights,
                "weight_stats": {
                    "k": k,
                    "occluded_token_mass": mass,
                    "foreground_weight": args.foreground_weight,
                    "occluder": args.occluder,
                },
                "source_sha256": hashlib.sha256((d / "source.png").read_bytes()).hexdigest(),
                "base_revision": plan["base_revision"],
                "split": d.parent.name,
                "kind": "rord",
            },
            d / "cache.pt",
        )
        done += 1
        if done % 25 == 0:
            print(json.dumps({"cached": done, "last": d.name}), flush=True)
    print(json.dumps({"cached_total": done}), flush=True)


if __name__ == "__main__":
    main()
