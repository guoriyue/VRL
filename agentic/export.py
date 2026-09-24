"""Export an episode's image states as a media manifest for independent rescoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic.episode import Artifact, Task
from vrl.utils.artifacts import atomic_file
from vrl.utils.json_files import canonical_json_sha256, write_json


def export_episode_media(trace: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write ``media.jsonl`` (source, then every edit output) plus lineage and the trace."""

    if trace.get("status") != "success":
        raise ValueError("only a successful episode exports media for rescoring")
    task = Task.from_record(trace["task"])
    states: list[tuple[Artifact, int | None]] = [(task.source, None)] + [
        (Artifact(**step["tool_result"]["artifact"]), index)
        for index, step in enumerate(trace["steps"])
        if "tool_result" in step
    ]
    episode_id = canonical_json_sha256(trace, allow_nan=False)
    rows, lineage = [], []
    for index, (artifact, decision_index) in enumerate(states):
        sample_id = f"{task.task_id}:state:{index:04d}"
        rows.append(
            {
                "sample_id": sample_id,
                "prompt_id": task.task_id,
                "prompt": task.instruction,
                "path": artifact.path,
                "sha256": artifact.sha256,
                "assets": {
                    "reference_image": task.source.path,
                    **{name: asset.path for name, asset in task.reward_assets.items()},
                },
                "metadata": {"source_group": task.source.sha256, "episode_id": episode_id},
            }
        )
        lineage.append(
            {
                "sample_id": sample_id,
                "decision_index": decision_index,
                "parent_sample_id": rows[index - 1]["sample_id"] if index else None,
                "parent_sha256": states[index - 1][0].sha256 if index else None,
                "sha256": artifact.sha256,
            }
        )
    report = {
        "schema": "vrl.episode-media-export.v1",
        "episode_id": episode_id,
        "task": task.as_record(),
        "controller_policy": trace["controller_policy"],
        "tool_policy": trace["tool_policy"],
        "judge_revision": trace["judge_revision"],
        "sample_order": [row["sample_id"] for row in rows],
        "lineage": lineage,
    }
    report = {"export_id": canonical_json_sha256(report, allow_nan=False), **report}
    output_dir.mkdir(parents=True, exist_ok=False)
    with atomic_file(output_dir / "media.jsonl", overwrite=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_json(output_dir / "episode.json", trace)
    write_json(output_dir / "provenance.json", report)
    return report
