"""Expose complete layers and counterfactual edits for reward review.

Views reuse decoded model pixels. They do not repair masks, complete backgrounds
or alter the candidate before judging. Scalar reconstruction error is a diagnostic,
not a sufficient reward for editable decomposition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def composite(layers):
    result = Image.new("RGBA", layers[0].size, (0, 0, 0, 0))
    for layer in layers:
        result = Image.alpha_composite(result, layer)
    return result


def prepare_views(source, layers, directory):
    """Save faithful per-layer and removal/motion views plus scalar diagnostics."""
    directory.mkdir(parents=True, exist_ok=True)
    width, height = source.size
    assert all(layer.size == source.size for layer in layers)
    full = composite(layers)
    displays = [("Source", source), ("Recomposed", full)]
    displays.extend(
        (f"Layer {index} (back to front)", layer) for index, layer in enumerate(layers)
    )
    for index in range(len(layers)):
        remaining = [layer for i, layer in enumerate(layers) if i != index]
        removed = composite(remaining) if remaining else Image.new("RGBA", source.size)
        removed.save(directory / f"remove_{index}.png")
        shifted = Image.new("RGBA", source.size, (0, 0, 0, 0))
        shifted.paste(layers[index], (max(1, width // 12), 0))
        moved = composite([shifted if i == index else layer for i, layer in enumerate(layers)])
        moved.save(directory / f"move_{index}.png")
        displays.extend([(f"Hide layer {index}", removed), (f"Move layer {index} right", moved)])
    tile_width = 480
    tile_height = round(height * tile_width / width)
    columns = 3
    rows = (len(displays) + columns - 1) // columns
    contact = Image.new("RGB", (columns * tile_width, rows * (tile_height + 28)), "white")
    draw = ImageDraw.Draw(contact)
    yy, xx = np.indices((tile_height, tile_width))
    checker = np.where(((xx // 12 + yy // 12) % 2)[..., None], 220, 245)
    checker = Image.fromarray(np.repeat(checker, 3, axis=2).astype("uint8")).convert("RGBA")
    for index, (label, image) in enumerate(displays):
        x = index % columns * tile_width
        y = index // columns * (tile_height + 28)
        visual = Image.alpha_composite(
            checker, image.convert("RGBA").resize((tile_width, tile_height))
        )
        contact.paste(visual.convert("RGB"), (x, y + 28))
        draw.text((x + 5, y + 6), label, fill="black")
    contact.save(directory / "review.png")
    alpha = np.array(full)[..., 3]
    black = Image.new("RGBA", source.size, (0, 0, 0, 255))
    source_rgba = source.convert("RGBA")
    source_rgb = np.array(Image.alpha_composite(black, source_rgba).convert("RGB"), dtype=float)
    reconstructed = np.array(Image.alpha_composite(black, full).convert("RGB"), dtype=float)
    stats = {
        "rgba_layers": len(layers),
        "alpha_pixels_over_half": [int((np.array(layer)[..., 3] > 127).sum()) for layer in layers],
        "composite_under_opaque_pixels": int((alpha < 250).sum()),
        "rgb_mae_over_black": float(np.abs(reconstructed - source_rgb).mean()),
        "alpha_mae": float(np.abs(alpha.astype(float) - np.array(source_rgba)[..., 3]).mean()),
        "scope": "View construction and diagnostics only; no semantic correctness or hidden truth claim.",
    }
    (directory / "diagnostics.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "plan.json").read_text())
    rows = []
    for case in plan["cases"]:
        source = Image.open(args.root / "conditions" / case["name"] / "source.png").convert(
            case.get("input_mode", "RGB")
        )
        for candidate in sorted((args.root / "rollouts" / case["name"]).glob("candidate*")):
            paths = [candidate / f"layer_{i}.png" for i in range(plan["layers"])]
            if not all(path.is_file() for path in paths):
                continue
            layers = [Image.open(path).convert("RGBA") for path in paths]
            stats = prepare_views(source, layers, candidate / "review")
            rows.append({"case": case["name"], "candidate": candidate.name, **stats})
    (args.root / "candidate_diagnostics.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
