"""Export a chain run's image states as a media manifest for independent rescoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic.chains import Artifact
from vrl.utils.artifacts import atomic_file
from vrl.utils.json_files import canonical_json_sha256, write_json


def export_chain_media(trace: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write ``media.jsonl`` (source, then every step output), lineage and the run record."""

    if trace.get("status") != "success":
        raise ValueError("only a successful chain run exports media for rescoring")
    chain = trace["chain"]
    states = [Artifact(**chain["source"])] + [
        Artifact(**step["artifact"]) for step in trace["steps"]
    ]
    run_id = canonical_json_sha256(trace, allow_nan=False)
    rows, lineage = [], []
    for index, artifact in enumerate(states):
        sample_id = f"{chain['chain_id']}:state:{index:04d}"
        rows.append(
            {
                "sample_id": sample_id,
                "prompt_id": chain["chain_id"],
                "prompt": chain["instruction"],
                "path": artifact.path,
                "sha256": artifact.sha256,
                "assets": {
                    "reference_image": chain["source"]["path"],
                    **{name: asset["path"] for name, asset in chain["reward_assets"].items()},
                },
                "metadata": {"source_group": chain["source"]["sha256"], "run_id": run_id},
            }
        )
        lineage.append(
            {
                "sample_id": sample_id,
                "step": index - 1 if index else None,
                "parent_sample_id": rows[index - 1]["sample_id"] if index else None,
                "parent_sha256": states[index - 1].sha256 if index else None,
                "sha256": artifact.sha256,
            }
        )
    report = {
        "schema": "vrl.chain-media-export.v1",
        "run_id": run_id,
        "chain": chain,
        "editor_policy": trace["editor_policy"],
        "judge_revision": trace["judge_revision"],
        "sample_order": [row["sample_id"] for row in rows],
        "lineage": lineage,
    }
    report = {"export_id": canonical_json_sha256(report, allow_nan=False), **report}
    output_dir.mkdir(parents=True, exist_ok=False)
    with atomic_file(output_dir / "media.jsonl", overwrite=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_json(output_dir / "run.json", trace)
    write_json(output_dir / "provenance.json", report)
    return report
