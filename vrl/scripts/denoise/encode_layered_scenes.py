"""Encode layered scenes into the ``cache.pt`` the SFT recipe and the evaluator read.

Each scene directory holds ``source.png`` and, when it carries truth, one
``layer_<i>.png`` per layer (back to front) plus an optional
``inferable_mask.png``. The family model of ``--experiment`` encodes the source
into condition tokens and prompt embeds (``encode_prompt``) and the frames into
packed target latents (``encode_video_to_latents``), so the cache is pinned to
the model revision the run trains with. The transparent spare-layer target
``empty_frame.pt`` is written beside the splits. The contract is documented in
``vrl/trainers/data/layered_scenes.py``.

    python -m vrl.scripts.denoise.encode_layered_scenes \\
        --experiment qwen_image_layered/offline_sft_layered_scenes \\
        --root data/layered_scenes/dataset_v3 --splits train heldout model.use_lora=false

Truth-free photos become scenes first: ``--photos a.jpg b.jpg --out DIR --layers 6``
center-crops each to the sampling size under ``DIR/test/scene_r<NN>/source.png``
and caches it with that fixed layer budget. On a 32 GB card add
``precision.rollout.quantization.format=fp8`` so the full pipeline loads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "a clean, well composed image"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="experiment config name")
    parser.add_argument("--root", type=Path, help="dataset root holding <split>/<scene>/")
    parser.add_argument("--splits", nargs="+", default=["train", "heldout"])
    parser.add_argument("--photos", nargs="+", type=Path, help="photos to crop into --out/test")
    parser.add_argument("--out", type=Path, help="dataset root the photos are written to")
    parser.add_argument("--layers", type=int, help="layer budget cached for truth-free photos")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="the fixed decomposition prompt")
    parser.add_argument("--source-note", help="provenance and license of the photos")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: cuda if available)")
    parser.add_argument("--limit", type=int, default=None, help="encode only the first N scenes")
    parser.add_argument("overrides", nargs="*", help="preset overlays and dotlist overrides")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop_photo(photo: Path, scene: Path, *, height: int, width: int) -> dict[str, Any]:
    """Center-crop ``photo`` to the sampling aspect, resize it and write ``scene/source.png``."""

    from PIL import Image

    scene.mkdir(parents=True, exist_ok=True)
    image = Image.open(photo).convert("RGB")
    w, h = image.size
    scale = min(w / width, h / height)
    crop_w, crop_h = round(width * scale), round(height * scale)
    left, top = (w - crop_w) // 2, (h - crop_h) // 2
    image = image.crop((left, top, left + crop_w, top + crop_h)).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    image.save(scene / "source.png")
    return {"scene": scene.name, "photo": str(photo), "original_size": [w, h]}


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
    layers: int | None = None,
) -> dict[str, Any]:
    """Cache one scene: conditioning for every scene, targets and weights when it has truth."""

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
    if layer_count == 0 and layers is None:
        raise ValueError(f"{scene} has no layer_*.png; pass --layers for a truth-free photo")
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
        "layers": layer_count or int(layers),
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
    if (args.root is None) == (args.photos is None):
        raise ValueError("pass exactly one of --root or --photos")
    if args.photos is not None and (args.out is None or args.layers is None):
        raise ValueError("--photos needs --out and --layers")

    import torch

    from vrl.config.loading import load_config
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry
    from vrl.trainers.data.layered_scenes import EMPTY_FRAME_FILE, LAYERED_SCENE_CACHE

    cfg = load_config(f"experiment/{args.experiment}", overrides=args.overrides)
    root = parse_config(cfg)
    if root.model is None or root.sampling is None:
        raise ValueError("scene encoding requires model and sampling configuration")
    entry = get_model_family_entry(str(root.model.family))
    if entry.family != "qwen_image_layered":
        raise ValueError(f"layered scenes encode through qwen_image_layered, not {entry.family!r}")
    height, width = int(root.sampling.height), int(root.sampling.width)
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if args.photos is not None:
        rows = [
            crop_photo(
                photo, args.out / "test" / f"scene_r{index:02d}", height=height, width=width
            )
            for index, photo in enumerate(args.photos)
        ]
        card = {"layers": args.layers, "rows": rows, "scope": "Truth-free evaluation only."}
        if args.source_note:
            card["source"] = args.source_note
        (args.out / "photos.json").write_text(json.dumps(card, indent=2) + "\n")
        dataset_root, splits = args.out, ["test"]
    else:
        dataset_root, splits = args.root, args.splits

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

    empty_path = dataset_root / EMPTY_FRAME_FILE
    if not empty_path.exists():
        with torch.no_grad():
            latent = model.encode_video_to_latents(
                torch.zeros(1, 4, 1, height, width, device=device)
            )
        torch.save({"latent": latent.detach().float().cpu(), **provenance}, empty_path)
        logger.info("wrote the transparent spare-layer target to %s", empty_path)

    done = 0
    for split in splits:
        scenes = sorted(p for p in (dataset_root / split).glob("*") if (p / "source.png").exists())
        for scene in scenes[: args.limit]:
            if (scene / LAYERED_SCENE_CACHE).exists():
                continue
            cache = encode_scene(
                model, scene, prompt=args.prompt, height=height, width=width, layers=args.layers
            )
            torch.save({**cache, **provenance, "split": split}, scene / LAYERED_SCENE_CACHE)
            done += 1
            logger.info(
                "cached %s/%s layers=%d %s",
                split,
                scene.name,
                cache["layers"],
                {k: round(v, 4) for k, v in cache.get("weight_stats", {}).items()},
            )
    logger.info("cached %d scenes under %s", done, dataset_root)


if __name__ == "__main__":
    main()
