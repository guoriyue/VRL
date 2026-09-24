"""Build deterministic image perturbations for independent reward stress audits.

Outputs are diagnostic candidates, not automatically labelled preferences.
The existing rescore_media transport scores them; analyze_scores stress reports
paired changes without deciding which semantic outputs are correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from vrl.rewards.evaluation import load_media_manifest
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import canonical_json_sha256, write_json


def build_stress_manifest(manifest: Path, output_dir: Path, *, seed: int = 42) -> Path:
    """Keep prompts/assets intact; use a source-specific seed independent of row order."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rows = load_media_manifest(manifest)
    if any("reward_stress" in row.metadata or "reward_stress" in row.assets for row in rows):
        raise ValueError(
            "reward_stress metadata is reserved; nested stress inputs are unsupported"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    images = output_dir / "images"
    images.mkdir()
    records, inputs = [], []
    for row in rows:
        source_hash = sha256_file(row.path)
        with Image.open(row.path) as source:
            if getattr(source, "n_frames", 1) != 1 or source.getexif().get(274, 1) != 1:
                raise ValueError("stress inputs must be single images with normalized orientation")
            rgba = source.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")
        rng = np.random.default_rng(
            int(
                canonical_json_sha256([seed, row.sample_id, source_hash], allow_nan=False)[:16], 16
            )
        )
        pixels = np.asarray(rgb, dtype=np.float32)
        noisy = np.clip(pixels + rng.normal(0, 32, pixels.shape), 0, 255).round().astype(np.uint8)
        # This small, explicit perturbation set is the data recipe of this tool,
        # not an implicit domain vocabulary in the training workflow.
        variants = {
            "baseline": rgba,
            "blur_rgb": rgb.filter(ImageFilter.GaussianBlur(3)),
            "noise_rgb": Image.fromarray(noisy),
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
        baseline_id = canonical_json_sha256([row.sample_id, "baseline"], allow_nan=False)
        for transform, candidate in variants.items():
            if candidate.mode != "RGBA":
                candidate.putalpha(alpha)
            sample_id = canonical_json_sha256([row.sample_id, transform], allow_nan=False)
            path = images / f"{sample_id}.png"
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
                            "schema": "vrl.reward-stress.v1",
                            "source_sample_id": row.sample_id,
                            "source_sha256": source_hash,
                            "baseline_id": baseline_id,
                            "transform": transform,
                            "seed": seed,
                        },
                    },
                }
            )
        inputs.append(
            {
                **row.model_dump(mode="json"),
                "sha256": source_hash,
                "asset_sha256": {key: sha256_file(value) for key, value in row.assets.items()},
            }
        )
    path = output_dir / "media.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records)
    )
    write_json(
        output_dir / "recipe.json",
        {
            "schema": "vrl.reward-stress.v1",
            "seed": seed,
            "blur_radius_pixels": 3,
            "noise_std_255": 32,
            "inputs": inputs,
            "manifest_sha256": sha256_file(path),
            "limitations": [
                "Perturbations are probes, not human preference labels or task correctness proofs.",
                "RGB corruption preserves source alpha; only alpha-labelled probes change transparency.",
                "RGB-only scorers may ignore alpha probes entirely.",
            ],
        },
    )
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(build_stress_manifest(args.manifest, args.output_dir, seed=args.seed))


if __name__ == "__main__":
    main()
