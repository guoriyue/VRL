"""Encode one fully transparent 640x640 RGBA frame: the target for spare layers."""

import json
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage

from vrl.scripts.generation.qwen_layered_rl_probe import load_upstream

plan = json.loads(Path(sys.argv[1]).read_text())
upstream = load_upstream(Path(plan["upstream"]))
vae = (
    AutoencoderKLQwenImage.from_pretrained(
        plan["base_path"], subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True
    )
    .eval()
    .cuda()
)
transparent = torch.full((1, 4, 640, 640), -1.0)  # rgb 0 and alpha 0, in [-1, 1]
with torch.no_grad():
    latent = upstream.encode_condition_image(vae, transparent.cuda()).cpu().float()
    decoded = upstream.decode_layers(vae, torch.cat([latent, latent], dim=1).cuda(), 640, 640, 1)
alpha = (decoded[0, 0, 3].float().cpu() + 1) / 2
torch.save(
    {
        "latent": latent,
        "base_revision": plan["base_revision"],
        "decoded_alpha_max": float(alpha.max()),
    },
    Path(sys.argv[2]),
)
print(
    json.dumps(
        {
            "shape": list(latent.shape),
            "decoded_alpha_max": float(alpha.max()),
            "decoded_alpha_mean": float(alpha.mean()),
        }
    )
)
