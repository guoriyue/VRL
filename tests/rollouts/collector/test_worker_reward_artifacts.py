"""Worker-side reward artifact materialization (program A: media never enters the driver)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.execution.reward_artifacts import materialize_reward_artifacts
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.types import GenerationRequest
from vrl.rewards.artifacts import DiskRewardArtifactStore
from vrl.rewards.types import RewardSample
from vrl.utils.artifacts import MaterializedArtifact, RewardArtifactSpec


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
            # Reward models take unit-range floats; the wire's uint8 is restored
            # as k/255, the representation the driver used to hand over.
            saved = torch.load(path, weights_only=True)
            assert saved.dtype == torch.float32
            assert torch.equal(saved, media[index].float() / 255.0)
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


class _DiffusionShaped:
    """The diffusion result's media contract: ``video`` behind ``reward_media``."""

    def __init__(self, video: Any) -> None:
        self.video = video
        self.artifacts: dict[str, list[Any]] = {}

    @property
    def reward_media(self) -> Any:
        return self.video

    @reward_media.setter
    def reward_media(self, value: Any) -> None:
        self.video = value


class _ARShaped:
    """The token-AR result's media contract: decoded ``output`` behind ``reward_media``."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.artifacts: dict[str, list[Any]] = {}

    @property
    def reward_media(self) -> Any:
        return self.output

    @reward_media.setter
    def reward_media(self, value: Any) -> None:
        self.output = value


def _request(tmp_path: Path, *, media_off_wire: bool) -> GenerationRequest:
    return GenerationRequest(
        request_id="r",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=1,
        reward_artifacts=[_spec(tmp_path, media_type="image")],
        media_off_wire=media_off_wire,
    )


def test_worker_hook_drops_media_only_when_every_reward_reads_files(tmp_path: Path) -> None:
    off_wire = _DiffusionShaped(torch.zeros(1, 3, 4, 4))
    GenerationWorkerCore._materialize_reward_artifacts(
        off_wire, _request(tmp_path, media_off_wire=True), primary=True
    )
    assert off_wire.video is None
    assert list(off_wire.artifacts) == ["hpsv3"] and len(off_wire.artifacts["hpsv3"]) == 1

    # An in-memory reward in the same run: files are written AND the media stays.
    mixed = _DiffusionShaped(torch.zeros(1, 3, 4, 4))
    GenerationWorkerCore._materialize_reward_artifacts(
        mixed, _request(tmp_path, media_off_wire=False), primary=True
    )
    assert mixed.video is not None
    assert len(mixed.artifacts["hpsv3"]) == 1

    # Without specs nothing changes (evaluation scripts keep receiving media).
    plain = _DiffusionShaped(torch.zeros(1, 3, 4, 4))
    GenerationWorkerCore._materialize_reward_artifacts(
        plain, GenerationRequest("r", "sd3_5", "t2i", ["p"], 1), primary=True
    )
    assert plain.video is not None and plain.artifacts == {}
    with pytest.raises(TypeError, match="reward_media"):
        GenerationWorkerCore._materialize_reward_artifacts(
            SimpleNamespace(video=torch.zeros(1, 3, 4, 4)),
            _request(tmp_path, media_off_wire=True),
            primary=True,
        )


def test_worker_hook_materializes_token_ar_outputs_by_the_same_contract(tmp_path: Path) -> None:
    """An AR family's decoded images are its reward media; the hook must not
    assume the diffusion field name."""

    result = _ARShaped(torch.zeros(1, 3, 4, 4))
    GenerationWorkerCore._materialize_reward_artifacts(
        result, _request(tmp_path, media_off_wire=True), primary=True
    )
    assert result.output is None
    assert len(result.artifacts["hpsv3"]) == 1
    assert Path(result.artifacts["hpsv3"][0].path).is_file()


def test_only_the_primary_rank_of_an_engine_writes_reward_files(tmp_path: Path) -> None:
    """Every rank of a multi-rank engine runs the batch; the driver keeps rank 0's
    result, so a file written by another rank would have no owner to release it."""

    secondary = _DiffusionShaped(torch.zeros(1, 3, 4, 4))
    GenerationWorkerCore._materialize_reward_artifacts(
        secondary, _request(tmp_path, media_off_wire=True), primary=False
    )
    assert secondary.artifacts == {}
    assert secondary.video is not None
    assert not any(tmp_path.rglob("*.pt"))


def test_media_off_wire_requires_artifact_specs() -> None:
    with pytest.raises(ValueError, match="media_off_wire requires reward_artifacts"):
        GenerationRequest("r", "sd3_5", "t2i", ["p"], 1, media_off_wire=True)


def test_gather_reward_artifacts_concatenates_in_batch_order(tmp_path: Path) -> None:
    from vrl.generation.execution.reward_artifacts import gather_reward_artifacts
    from vrl.utils.artifacts import MaterializedArtifact

    def files(n: int) -> list[MaterializedArtifact]:
        return [
            MaterializedArtifact(path=f"/x/{i}.pt", size_bytes=1, sha256="a" * 64)
            for i in range(n)
        ]

    first = SimpleNamespace(
        batch=SimpleNamespace(sample_count=2, batch_key="b0"), artifacts={"hpsv3": files(2)}
    )
    second = SimpleNamespace(
        batch=SimpleNamespace(sample_count=1, batch_key="b1"), artifacts={"hpsv3": files(1)}
    )
    assert gather_reward_artifacts([first, second]) == {"hpsv3": files(2) + files(1)}
    assert gather_reward_artifacts([SimpleNamespace(batch=first.batch, artifacts={})]) is None
    with pytest.raises(ValueError, match="missing or misaligned"):
        gather_reward_artifacts([first, SimpleNamespace(batch=second.batch, artifacts={})])


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
