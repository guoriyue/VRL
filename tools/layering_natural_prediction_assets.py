"""Export fixed held-out predictions for interactive layer-removal inspection."""

import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--site", type=Path, required=True)
    args = p.parse_args()
    out = args.site / "natural" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    rows = {
        r["id"]: r
        for r in map(json.loads, (args.run / "eval_rows.jsonl").read_text().splitlines())
    }
    selected = {
        "natural_00014511_02": "白色鞋",
        "natural_00062920_02": "画架",
        "natural_00109515_02": "耳环",
        "coco_L6_r01": "真实厨房",
    }
    records = []
    for scene, label in selected.items():
        row = rows[scene]
        for arm in sorted((args.run / "eval").iterdir()):
            directory = arm / scene / "seed4401"
            paths = sorted(directory.glob("layer_*.png"), key=lambda x: int(x.stem.split("_")[-1]))
            if len(paths) != row["layers"]:
                continue
            urls = []
            for i, path in enumerate([Path(row["scene_dir"]) / "source.png", *paths]):
                filename = f"{scene}_{arm.name}_{i}.webp"
                im = Image.open(path).convert("RGBA")
                im.thumbnail((320, 320), Image.Resampling.LANCZOS)
                im.save(out / filename, quality=90, method=6)
                urls.append(f"natural/predictions/{filename}")
            records.append(
                {
                    "scene": scene,
                    "label": label,
                    "arm": arm.name,
                    "source": urls[0],
                    "layers": urls[1:],
                }
            )
    (out / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(records)} fixed scene/arm combinations")


if __name__ == "__main__":
    main()
