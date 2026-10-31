"""Generate review-only SAM 3 masks without changing any training annotations."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model_path = Path("data/models/sam3_modelscope")
    model = Sam3Model.from_pretrained(model_path, local_files_only=True).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for request in json.loads(args.requests.read_text()):
        started = time.monotonic()
        image = Image.open(request["reference_image"]).convert("RGB")
        original = np.asarray(image)
        existing = np.load(request["instances"])["masks"].astype(bool)
        inputs = processor(images=image, text=request["concept"], return_tensors="pt").to(
            args.device
        )
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        masks = result["masks"].cpu().numpy().astype(bool)
        scores = result["scores"].cpu().numpy()
        directory = args.output / request["id"]
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / "proposals.npz", masks=masks, scores=scores)
        objects = []
        for index, mask in enumerate(masks):
            intersection = (existing & mask).sum(axis=(1, 2))
            union = (existing | mask).sum(axis=(1, 2))
            overlay = original.copy()
            overlay[mask] = (0.5 * overlay[mask] + np.array([127.5, 0, 0])).astype(np.uint8)
            Image.fromarray(overlay).save(directory / f"proposal_{index}.png")
            objects.append(
                {
                    "index": index,
                    "confidence": float(scores[index]),
                    "pixels": int(mask.sum()),
                    "max_existing_iou": float((intersection / np.maximum(union, 1)).max()),
                    "fraction_overlapping_existing": float(
                        (mask & existing.any(axis=0)).sum() / max(int(mask.sum()), 1)
                    ),
                }
            )
        record = {
            **request,
            "proposals": objects,
            "seconds": time.monotonic() - started,
            "review_status": "pending",
            "admitted_to_training": False,
        }
        records.append(record)
        (directory / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            json.dumps(
                {"id": request["id"], "proposals": len(objects), "seconds": record["seconds"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
