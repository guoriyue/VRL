"""Encode layered scenes into the ``cache.pt`` the SFT recipe and the evaluator read.

Each scene directory holds ``source.png`` and, when it carries truth, one
``layer_<i>.png`` per layer (back to front) plus an optional
``inferable_mask.png``. A scene without layers is truth-free: it caches only
its conditioning, with ``sampling.layers`` as its layer budget. The family
model of ``--config`` encodes the source into condition tokens and prompt
embeds (``encode_prompt``) and the frames into packed target latents
(``encode_video_to_latents``), so the cache is pinned to the model revision
the run trains with. Every root of ``data.manifest`` is encoded, every
``<split>/`` directory under it; scenes that already hold ``cache.pt`` are
skipped. The transparent spare-layer target ``empty_frame.pt`` is written
beside the splits. The contract is documented in
``vrl/trainers/data/layered_scenes.py``.

    python -m vrl.scripts.denoise.encode_layered_scenes \\
        --config experiment/qwen_image_layered/offline_sft_layered_scenes \\
        data.manifest=data/layered_scenes/dataset_v3 model.use_lora=false model.lora=null

On a 32 GB card add ``precision.rollout.quantization.format=fp8`` so the full
pipeline loads.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "a clean, well composed image"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Bundled config name or YAML path.")
    parser.add_argument("overrides", nargs="*", help="preset overlays and dotlist overrides")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_frames(scene: Path, layer_count: int) -> Any:
    """The source composite (opaque) and every layer as ``[1, 4, layers + 1, H, W]`` in [0, 1]."""

    import numpy as np
    import torch
    from PIL import Image

    source = np.array(Image.open(scene / "source.png").convert("RGBA"))
    source[..., 3] = 255
    frames = [source] + [
        np.array(Image.open(scene / f"layer_{i}.png").convert("RGBA")) for i in range(layer_count)
    ]
    stack = torch.from_numpy(np.stack(frames))  # [T, H, W, 4]
    return stack.permute(3, 0, 1, 2).float().div(255.0).unsqueeze(0)


def encode_scene(
    model: Any,
    scene: Path,
    *,
    prompt: str,
    height: int,
    width: int,
    layers: int,
) -> dict[str, Any]:
    """Cache one scene: conditioning for every scene, targets and weights when it has truth.

    ``layers`` is the budget recorded for a truth-free scene; a scene with
    ``layer_*.png`` records its own layer count.
    """

    import torch
    from PIL import Image

    from vrl.trainers.data.layered_scenes import (
        LAYERED_SCENE_SCHEMA_VERSION,
        token_weights,
    )

    source = Image.open(scene / "source.png")
    if source.size != (width, height):
        raise ValueError(
            f"{scene / 'source.png'} is {source.size}, not the sampling size {(width, height)}",
        )
    layer_paths = sorted(scene.glob("layer_*.png"))
    layer_count = len(layer_paths)
    with torch.no_grad():
        encoded = model.encode_prompt(
            prompt,
            None,
            reference_image=source.convert("RGBA"),
            height=height,
            width=width,
            input_mode="rgb",
        )
    cache: dict[str, Any] = {
        "schema": LAYERED_SCENE_SCHEMA_VERSION,
        "condition_latents": encoded["condition_latents"].detach().cpu(),
        "prompt_embeds": encoded["prompt_embeds"].detach().cpu(),
        "prompt_embeds_mask": (
            None
            if encoded.get("prompt_embeds_mask") is None
            else encoded["prompt_embeds_mask"].detach().cpu()
        ),
        "prompt": prompt,
        "height": height,
        "width": width,
        "layers": layer_count or layers,
        "source_sha256": _sha256(scene / "source.png"),
    }
    if layer_count:
        with torch.no_grad():
            target = model.encode_video_to_latents(
                scene_frames(scene, layer_count).to(model.device)
            )
        cache["target_latents"] = target.detach().float().cpu()
        if (scene / "inferable_mask.png").exists():
            weighted, stats = token_weights(scene, layer_count, height=height, width=width)
        else:
            weighted, stats = torch.ones(1, target.shape[1], 1), {}
        cache["weights_weighted"] = weighted
        cache["weights_control"] = torch.ones_like(weighted)
        cache["weight_stats"] = stats
    return cache


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    import torch

    from vrl.config.data import manifest_sources
    from vrl.config.loading import load_config
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry
    from vrl.trainers.data.layered_scenes import EMPTY_FRAME_FILE, LAYERED_SCENE_CACHE

    root = parse_config(load_config(args.config, overrides=args.overrides))
    if root.model is None or root.sampling is None or root.data is None:
        raise ValueError("scene encoding requires model, sampling and data configuration")
    entry = get_model_family_entry(str(root.model.family))
    if entry.family != "qwen_image_layered":
        raise ValueError(f"layered scenes encode through qwen_image_layered, not {entry.family!r}")
    height, width = int(root.sampling.height), int(root.sampling.width)
    layers = int(root.sampling.layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    build = entry.resolve_model_build(
        root,
        device,
        precision=PrecisionPolicy.from_section(root.precision),
        parameter_dtype_override=torch.float32 if device.type == "cpu" else None,
    )
    model = entry.build_rollout(build, expected_identity=None).model.eval()
    provenance = {
        "family": entry.family,
        "model_path": str(root.model.path or ""),
        "model_revision": str(root.model.revision or ""),
    }

    for source in manifest_sources(root.data.manifest):
        dataset_root = Path(source).expanduser()
        empty_path = dataset_root / EMPTY_FRAME_FILE
        if not empty_path.exists():
            with torch.no_grad():
                latent = model.encode_video_to_latents(
                    torch.zeros(1, 4, 1, height, width, device=device)
                )
            torch.save({"latent": latent.detach().float().cpu(), **provenance}, empty_path)
            logger.info("wrote the transparent spare-layer target to %s", empty_path)

        done = 0
        for split_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
            scenes = sorted(p for p in split_dir.iterdir() if (p / "source.png").exists())
            for scene in scenes:
                if (scene / LAYERED_SCENE_CACHE).exists():
                    continue
                cache = encode_scene(
                    model, scene, prompt=DEFAULT_PROMPT, height=height, width=width, layers=layers
                )
                torch.save(
                    {**cache, **provenance, "split": split_dir.name}, scene / LAYERED_SCENE_CACHE
                )
                done += 1
                logger.info(
                    "cached %s/%s layers=%d %s",
                    split_dir.name,
                    scene.name,
                    cache["layers"],
                    {k: round(v, 4) for k, v in cache.get("weight_stats", {}).items()},
                )
        logger.info("cached %d scenes under %s", done, dataset_root)


if __name__ == "__main__":
    main()
