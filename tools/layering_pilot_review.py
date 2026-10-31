"""Build a visual review sheet for the largest held-out changes in a pilot."""

import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", default="step_30")
    args = parser.parse_args()
    run = args.run.resolve()
    comparison = json.loads((run / "comparison.json").read_text())
    base_name = comparison["baseline"]
    baseline = {r["id"]: r for r in comparison["arms"][base_name]["rows"]}
    candidate = {r["id"]: r for r in comparison["arms"][args.candidate]["rows"]}
    scenes = {
        r["id"]: r for r in map(json.loads, (run / "eval_rows.jsonl").read_text().splitlines())
    }
    photos = [k for k in baseline if k.startswith("coco")]

    def change(k):
        return (
            candidate[k]["complete_pure_objects"] - baseline[k]["complete_pure_objects"],
            candidate[k]["grouping"] - baseline[k]["grouping"],
        )

    chosen = list(dict.fromkeys([max(photos, key=change), min(photos, key=change)]))
    rendered = [k for k in baseline if k.startswith("rendered")]
    chosen.append(
        min(
            rendered,
            key=lambda k: candidate[k]["layered_edit_holes"] - baseline[k]["layered_edit_holes"],
        )
    )
    markup = [
        """<!doctype html><meta charset="utf-8"><title>Pilot visual review</title>
<style>body{font:15px sans-serif;background:#eee;margin:12px}section{display:flex;gap:7px;margin:8px 0 22px}figure{margin:0;width:140px;flex:none}img{width:140px;height:140px;object-fit:contain;background:repeating-conic-gradient(#ddd 0% 25%,#fff 0% 50%) 0 0/20px 20px}h3{margin:0}figcaption{font-size:12px}</style>"""
    ]
    records = []
    for key in chosen:
        for arm, metrics in [(base_name, baseline[key]), (args.candidate, candidate[key])]:
            directory = run / "eval" / arm / key / "seed4401"
            images = [
                Path(scenes[key]["scene_dir"]) / "source.png",
                *sorted(directory.glob("layer_*.png"), key=lambda p: int(p.stem.split("_")[-1])),
            ]
            assert len(images) > 2
            title = f"{key} | {arm} | complete/pure {metrics['complete_pure_objects']} | background {metrics['object_background_share']:.3f} | merges {metrics['merge_layers']}"
            markup.append(f"<h3>{html.escape(title)}</h3><section>")
            for image in images:
                markup.append(
                    f'<figure><img src="{html.escape(image.as_uri())}"><figcaption>{html.escape(image.name)}</figcaption></figure>'
                )
            markup.append("</section>")
        records.append({"id": key, "baseline": baseline[key], "candidate": candidate[key]})
    (run / "pilot_visual_review.html").write_text("\n".join(markup))
    (run / "pilot_visual_review_selection.json").write_text(
        json.dumps(
            {
                "selection": "Largest gain and largest loss in complete/pure objects, breaking ties by grouping; rendered scene with largest edit-quality decrease. Selected after evaluation for failure inspection, not a representative quality sample.",
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
    print("Prepared review:", chosen)


if __name__ == "__main__":
    main()
