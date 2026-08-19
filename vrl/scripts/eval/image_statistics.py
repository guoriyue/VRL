"""Model-free image statistics: saturation, brightness, edge energy.

Standalone diagnostics for generated images — no ``vrl`` imports, mirroring the
other probe scripts in this directory. These statistics are reward-independent
evidence for eval analyses: SPRINT_anima_rl_5090 used within-group correlation
against saturation to show every shelf reward partly ranks style, and edge
energy (line/detail density) to detect the flattening that preference rewards
disagreed about.

Extracted from the deleted ``image_statistic`` probe reward. That wrapper let a
statistic drive GRPO as a controllable ground-truth signal ("is the pipeline
learning?" separated from "is the reward any good?"); the pipeline question was
settled by experiment D, so the reward-side wrapper is gone. To resurrect the
probe, wrap ``statistic_value`` in a reward model returning
``direction * value``.

Usage:
    python -m vrl.scripts.eval.image_statistics outputs/eval_a/images outputs/eval_b/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STATISTICS = ("saturation", "brightness", "edge_energy")

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def saturation(image: np.ndarray) -> float:
    """Mean per-pixel channel spread (max - min) of a float [0, 1] HWC image."""
    return float((image.max(axis=2) - image.min(axis=2)).mean())


def brightness(image: np.ndarray) -> float:
    """Mean gray level of a float [0, 1] HWC image."""
    return float(image.mean(axis=2).mean())


def edge_energy(image: np.ndarray) -> float:
    """Mean absolute first difference along both axes — a plain proxy for
    line/detail density."""
    gray = image.mean(axis=2)
    return float(np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean())


def statistic_value(image: np.ndarray, statistic: str) -> float:
    """Dispatch by statistic name; ``image`` is float [0, 1] HWC."""
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}; got {statistic!r}")
    return {"saturation": saturation, "brightness": brightness, "edge_energy": edge_energy}[
        statistic
    ](image)


def load_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _collect_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES),
            )
        else:
            images.append(path)
    return images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="Image files or directories")
    parser.add_argument("--statistic", choices=(*STATISTICS, "all"), default="all")
    parser.add_argument("--per-image", action="store_true", help="Print one row per image")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    names = STATISTICS if args.statistic == "all" else (args.statistic,)

    images = _collect_images(args.paths)
    if not images:
        raise SystemExit("no images found")

    values = {name: [] for name in names}
    for path in images:
        image = load_image(path)
        row = {name: statistic_value(image, name) for name in names}
        for name in names:
            values[name].append(row[name])
        if args.per_image:
            cells = "  ".join(f"{name}={row[name]:.4f}" for name in names)
            print(f"{path}  {cells}")

    print(f"n={len(images)}")
    for name in names:
        arr = np.asarray(values[name])
        print(f"{name}: mean={arr.mean():.4f} std={arr.std():.4f}")


if __name__ == "__main__":
    main()
