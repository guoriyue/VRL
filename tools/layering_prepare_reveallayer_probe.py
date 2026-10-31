"""Download a pinned, bounded benchmark sample and audit its layer references.

References are candidate annotations, not automatically trusted training truth.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

REVISION = "2183b789899cb0ccff58a504f446cecf19c77103"
REPO = "https://huggingface.co/datasets/qihoo360/RevealLayer-100K"
PREFIX = "Benchmark/RevealLayerBenchMark-200"


def square(image, size):
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def composite(layers):
    out = np.zeros_like(layers[0][..., :3])
    for layer in layers:
        out = layer[..., :3] * layer[..., 3:] + out * (1 - layer[..., 3:])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    metadata = json.loads((root / PREFIX / "metaData.json").read_text())
    # Fixed, balanced selection independent of model outputs. Keep 2-5 objects
    # for a tractable first probe and never repurpose benchmark scenes as training.
    selected = []
    for count in [2, 3, 4, 5]:
        selected.extend(
            sorted(
                [r for r in metadata if len(r["LayerInfoRaw"]) == count], key=lambda r: r["imgid"]
            )[:6]
        )
    paths = sorted(
        {
            str(PurePosixPath(PREFIX) / name)
            for row in selected
            for name in [row["full_image"], row["background"], *row["LayerInfoRaw"]]
        }
    )

    def download(name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(name)
        path = root / name
        if not path.exists():
            request = urllib.request.Request(
                f"{REPO}/resolve/{REVISION}/" + urllib.parse.quote(name)
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise ValueError("Unexpectedly large benchmark image")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(download, paths))
    reports = []
    for row in selected:
        source = square(Image.open(root / PREFIX / row["full_image"]).convert("RGB"), 640)
        background = square(Image.open(root / PREFIX / row["background"]).convert("RGBA"), 640)
        foreground = [
            square(Image.open(root / PREFIX / name).convert("RGBA"), 640)
            for name in row["LayerInfoRaw"]
        ]
        small = [
            np.asarray(im.resize((64, 64), Image.Resampling.BOX), dtype=np.float32) / 255
            for im in [background, *foreground]
        ]
        target = np.asarray(source.resize((64, 64), Image.Resampling.BOX), dtype=np.float32) / 255
        order = min(
            itertools.permutations(range(len(foreground))),
            key=lambda order: float(
                np.abs(composite([small[0], *[small[i + 1] for i in order]]) - target).mean()
            ),
        )
        out = root / "probe640" / row["imgid"]
        out.mkdir(parents=True, exist_ok=True)
        source.save(out / "source.png")
        background.save(out / "layer_0.png")
        for index, i in enumerate(order, 1):
            foreground[i].save(out / f"layer_{index}.png")
        full = np.stack(
            [
                np.asarray(im, dtype=np.float32) / 255
                for im in [background, *[foreground[i] for i in order]]
            ]
        )
        error = float(np.abs(composite(full) - np.asarray(source, dtype=np.float32) / 255).mean())
        visible = []
        covered = np.zeros((640, 640), bool)
        for layer in full[:0:-1]:
            solid = layer[..., 3] >= 0.5
            visible.append(int((solid & ~covered).sum()))
            covered |= solid
        reports.append(
            {
                "id": row["imgid"],
                "scene_dir": str(out.resolve()),
                "layers": len(full),
                "source_composite_mae": error,
                "visible_object_pixels": visible[::-1],
                "foreground_order": list(order),
                "candidate_for_visual_review": error < 0.025 and min(visible) >= 64,
                "complete_layer_truth": False,
            }
        )
        print(row["imgid"], round(error, 5), flush=True)
    report = {
        "repository": REPO,
        "revision": REVISION,
        "license": "Apache-2.0 according to author dataset card",
        "source_split": PREFIX,
        "metadata_rows": len(metadata),
        "selected_rows": len(selected),
        "selection": "First six IDs per foreground count 2, 3, 4, 5; no model output used.",
        "sha256": hashes,
        "scenes": reports,
        "limitation": "Data include generated/inpainted references; low recomposition error does not prove complete or semantically correct annotations. Not admitted to training.",
    }
    (root / "probe_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Probe prepared:", len(reports), "scenes; require visual review before evaluation")


if __name__ == "__main__":
    main()
