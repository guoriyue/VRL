"""Worker-side reward artifact materialization (program A: media never enters the driver)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vrl.generation.execution.reward_artifacts import materialize_reward_artifacts
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.types import GenerationRequest, RewardArtifactSpec
from vrl.rewards.artifacts import DiskRewardArtifactStore
from vrl.rewards.types import MaterializedArtifact, RewardSample


def _spec(tmp_path: Path, name: str = "hpsv3", **overrides) -> RewardArtifactSpec:
    values = {
        "name": name,
        "root": str(tmp_path / name),
        "media_type": "video",
        "artifact_format": "tensor",
    }
    values.update(overrides)
    return RewardArtifactSpec(**values)


def test_spec_validates_its_vocabulary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mp4 requires media_type=video"):
        _spec(tmp_path, media_type="image", artifact_format="mp4")
    with pytest.raises(ValueError, match="component name"):
        RewardArtifactSpec(name="", root="/x", media_type="image", artifact_format="tensor")
    with pytest.raises(ValueError, match="must be unique"):
        GenerationRequest(
            request_id="r",
            family="sd3_5",
            task="t2i",
            inputs=["p"],
            samples_per_prompt=1,
            reward_artifacts=[_spec(tmp_path), _spec(tmp_path)],
        )


def test_materialize_writes_one_tensor_file_per_sample_per_spec(tmp_path: Path) -> None:
    media = torch.randint(0, 255, (3, 3, 2, 4, 4), dtype=torch.uint8)
    specs = [_spec(tmp_path, "a"), _spec(tmp_path, "b")]
    out = materialize_reward_artifacts(media, specs)
    assert set(out) == {"a", "b"} and all(len(files) == 3 for files in out.values())
    for name, files in out.items():
        for index, ref in enumerate(files):
            path = Path(ref.path)
            assert path.parent == (tmp_path / name).resolve() and path.suffix == ".pt"
            assert ref.size_bytes == path.stat().st_size
            assert torch.equal(torch.load(path, weights_only=True), media[index])
    with pytest.raises(ValueError, match="expects image media"):
        materialize_reward_artifacts(media, [_spec(tmp_path, "img", media_type="image")])
    assert materialize_reward_artifacts(media, []) == {}


def test_materialize_encodes_mp4_when_the_store_wants_video_files(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    media = torch.rand(2, 3, 4, 16, 16)
    out = materialize_reward_artifacts(media, [_spec(tmp_path, artifact_format="mp4", fps=4.0)])
    files = out["hpsv3"]
    assert len(files) == 2 and all(Path(f.path).suffix == ".mp4" for f in files)
    assert all(f.size_bytes > 0 and len(f.sha256) == 64 for f in files)


def test_worker_hook_drops_media_from_the_wire_once_materialized(tmp_path: Path) -> None:
    request = GenerationRequest(
        request_id="r",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=1,
        reward_artifacts=[_spec(tmp_path, media_type="image")],
    )
    output = SimpleNamespace(video=torch.zeros(1, 3, 4, 4), artifacts={})
    GenerationWorkerCore._materialize_reward_artifacts(output, request)
    assert output.video is None
    assert list(output.artifacts) == ["hpsv3"] and len(output.artifacts["hpsv3"]) == 1
    # Without specs nothing changes (evaluation scripts keep receiving media).
    plain = SimpleNamespace(video=torch.zeros(1, 3, 4, 4), artifacts={})
    GenerationWorkerCore._materialize_reward_artifacts(
        plain, GenerationRequest("r", "sd3_5", "t2i", ["p"], 1)
    )
    assert plain.video is not None and plain.artifacts == {}
    with pytest.raises(TypeError, match="expose decoded media"):
        GenerationWorkerCore._materialize_reward_artifacts(SimpleNamespace(), request)


def test_disk_store_adopts_worker_files_and_still_writes_tensors(tmp_path: Path) -> None:
    store = DiskRewardArtifactStore(tmp_path / "ocr", media_type="image", name="ocr")
    assert store.spec() == {
        "name": "ocr",
        "root": str((tmp_path / "ocr").resolve()),
        "media_type": "image",
        "artifact_format": "tensor",
    }
    media = torch.rand(2, 3, 4, 4)
    delivered = materialize_reward_artifacts(media, [RewardArtifactSpec(**store.spec())])["ocr"]
    samples = [
        RewardSample(prompt="p0", output=None, sample_id="s0", artifacts={"ocr": delivered[0]}),
        RewardSample(prompt="p1", output=media[1], sample_id="s1"),
    ]
    artifacts = store.materialize(samples)
    assert artifacts[0].path == delivered[0].path
    assert artifacts[0].sha256 == delivered[0].sha256
    assert artifacts[0].artifact_id.startswith("s0:")
    assert Path(artifacts[1].path).exists() and artifacts[1].sample_id == "s1"
    store.release(artifacts)
    assert not Path(artifacts[0].path).exists() and not Path(artifacts[1].path).exists()

    foreign = tmp_path / "elsewhere.pt"
    torch.save(media[0], foreign)
    bad = RewardSample(
        prompt="p",
        output=None,
        sample_id="s2",
        artifacts={"ocr": MaterializedArtifact(str(foreign), foreign.stat().st_size, "0" * 64)},
    )
    with pytest.raises(ValueError, match="outside this store's root"):
        store.materialize([bad])
