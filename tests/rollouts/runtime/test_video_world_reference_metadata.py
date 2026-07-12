from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from vrl.rollouts.collector.config import RolloutConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.scripts.common.online import _resolve_reference_artifacts
from vrl.scripts.diffusion.cosmos.train import _predict2_collector_kwargs
from vrl.trainers.data import load_prompt_manifest
from vrl.trainers.data.artifacts import (
    require_reference_images,
    resolve_prompt_example_artifacts,
)


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
    """Checks resolved reference image flows to collector metadata."""
    manifest = _write_reference_manifest(tmp_path)
    example = resolve_prompt_example_artifacts(
        load_prompt_manifest(manifest)[0],
        data_root=tmp_path,
    )
    builder = GenerationRequestBuilder(
        family="cosmos-predict2",
        task="v2w",
        request_prefix="cosmos",
        config=RolloutConfig(family="cosmos-predict2", values={"num_steps": 1}),
        return_artifacts=("trajectory",),
        default_task_type="video2world",
        metadata_key="rollout_metadata",
    )

    collector_request = builder.build(
        [example.prompt],
        1,
        {
            "reference_image": example.reference_image,
            "sample_metadata": example.metadata,
        },
    )

    assert collector_request.metadata["reference_image"].endswith("ref.ppm")
    assert collector_request.request.inputs[0].reference_image.endswith("ref.ppm")


def test_cosmos_per_sample_reference_uses_vrl_data_root(monkeypatch, tmp_path: Path) -> None:
    """Checks Cosmos per sample reference resolves at load and passes the hook."""
    manifest = _write_reference_manifest(tmp_path)
    monkeypatch.setenv("VRL_DATA_ROOT", str(tmp_path))
    examples = load_prompt_manifest(manifest)
    cfg = OmegaConf.create({"data": {"manifest": manifest.as_posix()}})

    # Production sequence: run_online_recipe resolves reference paths at load
    # time, then the family hook only validates.
    _resolve_reference_artifacts(examples, cfg)
    require_reference_images(
        examples,
        manifest_path=manifest,
    )

    assert examples[0].reference_image == str(
        (tmp_path / "video_world" / "references" / "ref.ppm").resolve(),
    )
    assert "reference_image" not in examples[0].metadata


def test_cosmos_per_sample_reference_uses_artifact_data_root(tmp_path: Path) -> None:
    """Checks cosmos per_sample references resolve under data.artifact_data_root."""
    manifest = _write_reference_manifest(tmp_path)
    examples = load_prompt_manifest(manifest)
    cfg = OmegaConf.create(
        {
            "data": {
                "manifest": manifest.as_posix(),
                "artifact_data_root": str(tmp_path),
            },
        },
    )

    _resolve_reference_artifacts(examples, cfg)
    require_reference_images(
        examples,
        manifest_path=manifest,
    )

    assert examples[0].reference_image == str(
        (tmp_path / "video_world" / "references" / "ref.ppm").resolve(),
    )
    assert "reference_image" not in examples[0].metadata


def test_cosmos_predict2_collector_uses_prompts_per_batch_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Checks Cosmos predict2 collector uses rollout batch size config."""
    manifest = _write_reference_manifest(tmp_path)
    monkeypatch.setenv("VRL_DATA_ROOT", str(tmp_path))
    examples = load_prompt_manifest(manifest)
    cfg = OmegaConf.create(
        {
            "data": {
                "manifest": manifest.as_posix(),
                "preprocessing": {"conditioning": "reference_image"},
            },
        },
    )

    _resolve_reference_artifacts(examples, cfg)
    kwargs = _predict2_collector_kwargs(cfg, examples)

    assert kwargs == {}
    assert examples[0].reference_image.endswith("ref.ppm")
