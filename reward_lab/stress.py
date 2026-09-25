"""Deterministic image perturbations for reward stress audits.

The perturbed manifest is scored like any other (``Evaluation.score``) and
``Analysis.stress`` pairs each variant with its baseline. Perturbations are
probes, not labels: a score that rises under blur is a case to review, not an
automatic verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from vrl.rewards.evaluation import load_media_manifest
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import canonical_json_sha256, write_json

BLUR_RADIUS = 3
NOISE_STD = 32


def build_stress_manifest(manifest: Path, output_dir: Path, *, seed: int = 42) -> Path:
    """Write one baseline plus six perturbed variants per source image.

    RGB perturbations keep the source alpha; only the two alpha probes change
    transparency. Each variant's noise seed derives from the source content, so
    row order does not change the images.
    """

    rows = load_media_manifest(manifest)
    if any("reward_stress" in row.metadata for row in rows):
        raise ValueError(
            "reward_stress metadata is reserved; nested stress inputs are unsupported"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    images = output_dir / "images"
    images.mkdir()
    records = []
    for row in rows:
        source_hash = sha256_file(row.path)
        with Image.open(row.path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("stress inputs must be single images")
            rgba = source.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")
        rng = np.random.default_rng(
            int(
                canonical_json_sha256([seed, row.sample_id, source_hash], allow_nan=False)[:16], 16
            )
        )
        pixels = np.asarray(rgb, dtype=np.float32)
        noisy = np.clip(pixels + rng.normal(0, NOISE_STD, pixels.shape), 0, 255)
        variants = {
            "baseline": rgba,
            "blur_rgb": rgb.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)),
            "noise_rgb": Image.fromarray(noisy.round().astype(np.uint8)),
            "checkerboard_rgb": Image.fromarray(
                np.repeat(
                    ((np.indices(pixels.shape[:2]).sum(axis=0) % 2) * 255)[..., None], 3, axis=2
                ).astype(np.uint8)
            ),
            "black_rgb": Image.new("RGB", rgba.size, "black"),
            "opaque_alpha": rgba.copy(),
            "empty_alpha": rgba.copy(),
        }
        variants["opaque_alpha"].putalpha(255)
        variants["empty_alpha"].putalpha(0)
        baseline_id = f"{row.sample_id}:baseline"
        for transform, candidate in variants.items():
            if candidate.mode != "RGBA":
                candidate.putalpha(alpha)
            sample_id = f"{row.sample_id}:{transform}"
            path = images / f"{canonical_json_sha256(sample_id, allow_nan=False)}.png"
            candidate.save(path)
            records.append(
                {
                    "sample_id": sample_id,
                    "prompt_id": row.prompt_id,
                    "prompt": row.prompt,
                    "path": str(path.resolve()),
                    "assets": {key: str(value) for key, value in row.assets.items()},
                    "metadata": {
                        **row.metadata,
                        "reward_stress": {
                            "source_sample_id": row.sample_id,
                            "source_sha256": source_hash,
                            "baseline_id": baseline_id,
                            "transform": transform,
                            "seed": seed,
                        },
                    },
                }
            )
    path = output_dir / "media.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records)
    )
    write_json(
        output_dir / "recipe.json",
        {"seed": seed, "blur_radius_pixels": BLUR_RADIUS, "noise_std_255": NOISE_STD},
    )
    return path


__all__ = ["build_stress_manifest"]
