"""Export compact reference previews and measured results into the report checkout."""

import argparse
import datetime
import json
import shutil
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--site", type=Path, required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--note", required=True)
    args = p.parse_args()
    out = args.site / "natural"
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.run / "data/dataset_audit.json").read_text())
    labels = {
        "00002781": "棕色鞋",
        "00013620": "碗",
        "00047170": "灯泡",
        "00093378": "台灯",
        "00014511": "白色鞋",
        "00062920": "画架",
        "00109515": "耳环",
    }
    cases = []
    for row in audit["scenes"]:
        if not row["id"].endswith("_02"):
            continue
        scene = Path(row["scene_dir"])
        urls = []
        for name in ["source", "layer_0", "layer_1", "layer_2"]:
            image = Image.open(scene / f"{name}.png").convert("RGBA")
            image.thumbnail((320, 320), Image.Resampling.LANCZOS)
            filename = f"{row['id']}_{name}.webp"
            image.save(out / filename, format="WEBP", quality=90, method=6)
            urls.append(f"natural/{filename}")
        cases.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "label": labels[row["source_id"]],
                "split": row["split"],
                "hidden_fraction": row["actual_hidden_fraction"],
                "source": urls[0],
                "layers": urls[1:],
            }
        )
    (out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    Image.open(args.run / "data_review.png").save(
        out / "data_review.webp", format="WEBP", quality=92, method=6
    )
    shutil.copy2(args.run / "data/dataset_audit.json", out / "dataset_audit.json")
    shutil.copy2(args.run / "reward_audit.json", out / "reward_audit.json")
    arms = {}
    for path in [args.run / "baseline_report.json", *sorted(args.run.glob("comparison_*.json"))]:
        if path.exists():
            arms.update(json.loads(path.read_text())["arms"])
    rows = []
    for arm, data in arms.items():
        c, n = data["groups"]["coco"], data["groups"]["natural"]
        rows.append(
            {
                "arm": arm,
                "photo_complete": c["total_complete_pure_objects"],
                "photo_merges": round(c["means"]["merge_layers"] * c["scenes"]),
                "natural_complete": n["total_complete_pure_objects"],
                "natural_full": n["means"]["full_object_quality_diagnostic"],
                "natural_removal": n["means"]["single_object_removal_quality"],
            }
        )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "status": args.status,
                "note": args.note,
                "rows": rows,
                "updated": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        "Exported",
        len(cases),
        "reference previews;",
        sum(f.stat().st_size for f in out.iterdir()),
        "asset bytes",
    )


if __name__ == "__main__":
    main()
