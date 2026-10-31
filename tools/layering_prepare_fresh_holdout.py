"""Freeze a COCO train2017 photo screen outside this experiment's val2017 training pool."""

import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    source = Path("/mnt/nvme/layering_data/coco_panoptic")
    output = Path("/mnt/nvme/layering_fresh_holdout_20261025")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "panoptic_train2017.json").read_text())
    categories = {x["id"]: x for x in metadata["categories"]}
    images = {x["id"]: x for x in metadata["images"]}
    annotations = sorted(metadata["annotations"], key=lambda x: x["image_id"])
    random.Random(20261026).shuffle(annotations)
    train = [
        json.loads(line)
        for line in Path("data/external/coco/layered_val2017/train_mixed_v3.jsonl")
        .read_text()
        .splitlines()
    ]
    forbidden = {x["metadata"].get("coco_id") for x in train}
    existing = [
        json.loads(line)
        for line in Path("/mnt/nvme/layering_natural_20261025/eval_rows.jsonl")
        .read_text()
        .splitlines()
    ]
    forbidden.update(x["metadata"].get("coco_id") for x in existing)
    quotas = {"2": 0, "3-4": 0, "5-6": 0, "7-8": 0}
    selected = []
    with zipfile.ZipFile(source / "panoptic_train2017.zip") as archive:
        for annotation in annotations:
            cid = annotation["image_id"]
            if cid in forbidden:
                continue
            image = Image.open(
                io.BytesIO(archive.read("panoptic_train2017/" + annotation["file_name"]))
            ).convert("RGB")
            side = min(image.size)
            x, y = (image.width - side) // 2, (image.height - side) // 2
            rgb = np.array(
                image.crop((x, y, x + side, y + side)).resize(
                    (640, 640), Image.Resampling.NEAREST
                ),
                dtype=np.uint32,
            )
            labels = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
            objects = [
                (labels == item["id"], categories[item["category_id"]]["name"])
                for item in annotation["segments_info"]
                if categories[item["category_id"]]["isthing"] and not item["iscrowd"]
            ]
            objects = [(mask, name) for mask, name in objects if mask.sum() >= 256]
            count = len(objects)
            if not 2 <= count <= 8:
                continue
            bucket = "2" if count == 2 else "3-4" if count <= 4 else "5-6" if count <= 6 else "7-8"
            if quotas[bucket] == 8:
                continue
            scene_id = f"coco_fresh_{cid:012d}"
            directory = output / "inputs" / scene_id
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / "instances.npz",
                masks=np.stack([m for m, _ in objects]),
                names=np.array([n for _, n in objects]),
            )
            selected.append(
                {
                    "id": scene_id,
                    "scene_dir": str(directory),
                    "layers": count + 2,
                    "metadata": {
                        "coco_id": cid,
                        "source": "coco_train2017_frozen_test",
                        "instances": str(directory / "instances.npz"),
                        "objects": [n for _, n in objects],
                        "complete_layer_truth": False,
                    },
                    "source_record": images[cid],
                    "bucket": bucket,
                }
            )
            quotas[bucket] += 1
            if len(selected) == 32:
                break
    assert len(selected) == 32
    # Freeze selection before downloading or seeing any model outputs.
    selection = output / "selection.json"
    if selection.exists():
        assert json.loads(selection.read_text()) == selected
    else:
        selection.write_text(json.dumps(selected, indent=2) + "\n")
    downloads = []
    for row in selected:
        record = row["source_record"]
        directory = Path(row["scene_dir"])
        original = directory / "original.jpg"
        if not original.exists():
            with urllib.request.urlopen(record["coco_url"], timeout=60) as response:
                original.write_bytes(response.read())
        image = Image.open(original).convert("RGB")
        assert image.size == (record["width"], record["height"])
        side = min(image.size)
        x, y = (image.width - side) // 2, (image.height - side) // 2
        image.crop((x, y, x + side, y + side)).resize((640, 640), Image.Resampling.LANCZOS).save(
            directory / "source.png"
        )
        downloads.append(
            {
                "id": row["id"],
                "url": record["coco_url"],
                "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "license_id": record["license"],
            }
        )
        print(row["id"], flush=True)
    rows = [
        {k: v for k, v in row.items() if k not in ["source_record", "bucket"]} for row in selected
    ]
    (output / "eval_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "audit.json").write_text(
        json.dumps(
            {
                "scenes": len(rows),
                "buckets": quotas,
                "annotated_objects": sum(len(x["metadata"]["objects"]) for x in rows),
                "recorded_training_id_overlap": [],
                "downloads": downloads,
                "licenses": metadata["licenses"],
                "selection_seed": 20261026,
                "sampling_steps": 10,
                "generation_seed": 4401,
                "active_training_changed": False,
                "limitations": "Official train2017 images reserved for testing in this project, whose known COCO training uses val2017. Unknown base-model pretraining overlap cannot be excluded. COCO categories and visible masks are not exhaustive complete-object truth. Visual review and generation remain pending.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
