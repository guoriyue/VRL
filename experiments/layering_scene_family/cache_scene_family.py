"""Cache conditioning, prompt embeds, target latents and token weights for every scene.

Per scene writes ``cache.pt``: packed condition tokens of the source image, the
shared prompt embeds, the target latents (source composite frame followed by
every truth layer, each VAE-encoded as its own one-frame clip), and two token
weight maps for the supervised arms: ``weights_weighted`` raises the tokens of
each layer's *inferable* hidden pixels so they carry a quarter of the loss
(unknowable pixels are never raised, so memorizing them is not rewarded);
``weights_control`` is all ones. Run with the layered_rl interpreter.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vrl.scripts.generation.qwen_layered_rl_probe import load_upstream

TOKENS_PER_FRAME = 40 * 40


def token_weights(
    dataset_scene: Path, layer_count: int, share: float = 0.25
) -> tuple[torch.Tensor, dict]:
    frames = layer_count + 1  # composite frame + every layer
    alphas = [
        np.asarray(Image.open(dataset_scene / f"layer_{i}.png").convert("RGBA"))[..., 3] > 127
        for i in range(layer_count)
    ]
    inferable = np.asarray(Image.open(dataset_scene / "inferable_mask.png").convert("L")) > 127
    coverage = torch.zeros(frames, TOKENS_PER_FRAME)
    for layer in range(layer_count):
        front = np.zeros_like(inferable)
        for above in alphas[layer + 1 :]:
            front |= above
        region = torch.tensor(alphas[layer] & front & inferable, dtype=torch.float32)[None, None]
        coverage[layer + 1] = F.interpolate(region, size=(40, 40), mode="area")[0, 0].reshape(-1)
    total = frames * TOKENS_PER_FRAME
    mass = float(coverage.sum())
    k = min(80.0, share * total / ((1.0 - share) * mass)) if mass > 0 else 0.0
    weights = 1.0 + k * coverage
    weights = weights / weights.mean()
    return weights.reshape(1, total, 1), {
        "k": k,
        "inferable_token_mass": mass,
        "inferable_share": float((k * coverage).sum() / (total + k * mass)) if mass else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, required=True, help="probe plan with base_path, upstream, prompt"
    )
    parser.add_argument("--splits", nargs="+", default=["train", "heldout"])
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    upstream = load_upstream(upstream_path)
    from diffusers import QwenImageLayeredPipeline

    pipe = QwenImageLayeredPipeline.from_pretrained(
        plan["base_path"], transformer=None, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.text_encoder.requires_grad_(False).eval().to("cuda")
    with torch.no_grad():
        prompt, mask = pipe.encode_prompt(plan["prompt"], device=torch.device("cuda"))
    prompt, mask = prompt.cpu(), None if mask is None else mask.cpu()
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    vae = pipe.vae.requires_grad_(False).eval().to("cuda")
    vae.enable_slicing()
    done = 0
    for split in args.splits:
        for scene in sorted((args.dataset / split).glob("scene_*")):
            if (scene / "cache.pt").exists():
                continue
            layer_count = len(list(scene.glob("layer_*.png")))
            source = Image.open(scene / "source.png").convert("RGB")
            assert source.size == (640, 640)
            pixels = torch.from_numpy(np.array(source)).permute(2, 0, 1).float() / 127.5 - 1
            images = [Image.open(scene / "source.png").convert("RGBA")] + [
                Image.open(scene / f"layer_{i}.png").convert("RGBA") for i in range(layer_count)
            ]
            with torch.no_grad():
                condition = upstream.encode_condition_image(vae, pixels[None].cuda()).cpu()
                frames = []
                for image in images:
                    rgba = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
                    frames.append(upstream.encode_condition_image(vae, rgba[None].cuda()).cpu())
                target = torch.cat(frames, dim=1).float()
            assert condition.shape == (1, TOKENS_PER_FRAME, 64) and target.shape == (
                1,
                (layer_count + 1) * TOKENS_PER_FRAME,
                64,
            )
            weighted, stats = token_weights(scene, layer_count)
            torch.save(
                {
                    "condition": condition,
                    "prompt": prompt,
                    "mask": mask,
                    "prompt_text": plan["prompt"],
                    "width": 640,
                    "height": 640,
                    "layers": layer_count,  # generated frames = layers + 1 (composite first)
                    "target_latents": target,
                    "weights_weighted": weighted,
                    "weights_control": torch.ones_like(weighted),
                    "weight_stats": stats,
                    "source_sha256": hashlib.sha256(
                        (scene / "source.png").read_bytes()
                    ).hexdigest(),
                    "base_revision": plan["base_revision"],
                    "split": split,
                },
                scene / "cache.pt",
            )
            done += 1
            print(
                json.dumps(
                    {
                        "cached": str(scene.name),
                        "split": split,
                        "layers": layer_count,
                        **{k: round(v, 4) for k, v in stats.items()},
                    }
                ),
                flush=True,
            )
    print(json.dumps({"cached_total": done}), flush=True)


if __name__ == "__main__":
    main()
