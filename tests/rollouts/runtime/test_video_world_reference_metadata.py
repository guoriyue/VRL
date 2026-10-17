from __future__ import annotations

from pathlib import Path

from vrl.models.families.registry import get_model_family_entry
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trainers.data.prompts import (
    load_prompt_dataset_index,
)
from vrl.utils.paths import RootedPaths


def _write_reference_manifest(root: Path) -> Path:
    reference = root / "video_world" / "references" / "ref.ppm"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    manifest = root / "video_world" / "v2w_train.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        (
            '{"prompt":"The robot arm moves toward the cup.",'
            '"reference_image":"video_world/references/ref.ppm",'
            '"metadata":{"source_episode":"episode_train"}}\n'
        ),
        encoding="utf-8",
    )
    return manifest


def test_resolved_reference_image_flows_to_collector_metadata(tmp_path: Path) -> None:
    """A manifest row's reference image, resolved against the data root, reaches both the request
    input and the collector metadata.
    """
    manifest = _write_reference_manifest(tmp_path)
    example = load_prompt_dataset_index(manifest)[0].with_resolved_references(
        RootedPaths.data_root(tmp_path)
    )
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("cosmos-predict2"),
        config=RolloutCollectorConfig(request_sampling={"num_steps": 1}),
    )

    collector_request = builder.build(
        [example.generation_input()],
        1,
        metadata=example.reward_metadata(),
    )

    assert collector_request.metadata["reference_images"][0].endswith("ref.ppm")
    assert collector_request.request.inputs[0].reference_images[0].endswith("ref.ppm")
