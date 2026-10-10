"""Cache real room photographs for truth-free layered evaluation.

Each photo is center-cropped to a square, resized to 640x640 and VAE-encoded
into packed condition tokens; the prompt embeds are reused from a training
cache (same prompt text). No truth layers exist: the layer budget is fixed by
``--layers`` and scoring is truth-free (see evaluate_scene_family --no-truth).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vrl.scripts.generation.qwen_layered_rl_probe import load_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    upstream = load_upstream(Path(plan["upstream"]))
    prompt_cache = torch.load(args.prompt_cache, map_location="cpu", weights_only=True)
    assert prompt_cache["prompt_text"] == plan["prompt"]
    from diffusers import AutoencoderKLQwenImage

    vae = (
        AutoencoderKLQwenImage.from_pretrained(
            plan["base_path"], subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .cuda()
    )
    vae.requires_grad_(False)
    rows = []
    for index, photo in enumerate(args.photos):
        scene = args.out / "test" / f"scene_r{index:02d}"
        scene.mkdir(parents=True, exist_ok=True)
        image = Image.open(photo).convert("RGB")
        w, h = image.size
        s = min(w, h)
        image = image.crop(
            ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
        ).resize((640, 640), Image.LANCZOS)
        image.save(scene / "source.png")
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
        with torch.no_grad():
            condition = upstream.encode_condition_image(vae, pixels[None].cuda()).cpu()
        torch.save(
            {
                "condition": condition,
                "prompt": prompt_cache["prompt"],
                "mask": prompt_cache["mask"],
                "prompt_text": plan["prompt"],
                "width": 640,
                "height": 640,
                "layers": args.layers,
                "source_sha256": hashlib.sha256((scene / "source.png").read_bytes()).hexdigest(),
                "base_revision": plan["base_revision"],
                "split": "real",
                "photo": str(photo),
            },
            scene / "cache.pt",
        )
        rows.append({"scene": scene.name, "photo": str(photo), "original_size": [w, h]})
        print(json.dumps(rows[-1]), flush=True)
    (args.out / "real_rooms.json").write_text(
        json.dumps(
            {
                "layers": args.layers,
                "rows": rows,
                "source": "OmniEdit-Filtered-1.2M / GEdit-Bench photos (MIT), center-cropped to 640x640; never used in training",
                "scope": "Truth-free evaluation only.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
