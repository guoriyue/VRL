"""File-only model compatibility is local to the real inference scorer."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import pytest
import torch

from vrl.rewards.artifacts import DiskRewardArtifactStore
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.media import decode_artifact_frames
from vrl.rewards.runtime import InProcessRewardScorer
from vrl.rewards.types import RewardSample


class _FileConsumer:
    input_artifact_format: ClassVar[Literal["mp4", "tensor"]] = "mp4"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.paths: list[Path] = []
        self.frames: list[torch.Tensor] = []

    def __call__(self, artifact):
        path = Path(artifact.as_path())
        assert path.is_file()
        assert artifact.media is None
        self.paths.append(path)
        frames = decode_artifact_frames(artifact)
        self.frames.append(frames)
        if self.fail:
            raise RuntimeError("scoring failed after opening local media")
        return {"mean": float(frames.mean())}


class _TensorFileConsumer(_FileConsumer):
    input_artifact_format: ClassVar[Literal["mp4", "tensor"]] = "tensor"


def _sample(metadata: dict | None = None) -> RewardSample:
    # Nontrivial, deterministic pixels pin real encoding behavior rather than
    # comparing a mocked file writer or a uniform frame.
    video = torch.arange(3 * 4 * 16 * 16).remainder(256).to(torch.uint8).reshape(3, 4, 16, 16)
    return RewardSample(
        prompt="a moving pattern",
        output=video,
        sample_id="pattern",
        metadata=metadata or {},
    )


def _inline(sample: RewardSample) -> RewardInferenceArtifact:
    return RewardInferenceArtifact(
        artifact_id="inline-pattern",
        sample_id=sample.sample_id,
        path="",
        prompt=sample.prompt,
        media=sample.output,
        metadata=dict(sample.metadata),
    )


@pytest.mark.parametrize("consumer_type", [_FileConsumer, _TensorFileConsumer])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.asyncio
async def test_scorer_owns_only_temporary_files_and_cleans_them_on_error(
    consumer_type, fail, tmp_path
) -> None:
    sample = _sample()
    store = DiskRewardArtifactStore(
        tmp_path / "borrowed", artifact_format=consumer_type.input_artifact_format
    )
    borrowed = store.materialize([sample])[0]
    borrowed_path = Path(borrowed.path)
    original_bytes = borrowed_path.read_bytes()
    temporary_root = tmp_path / "scorer-temporary"
    temporary_root.mkdir()
    consumer = consumer_type(fail=fail)
    scorer = InProcessRewardScorer(model=consumer, media_temp_dir=str(temporary_root))
    request = RewardInferenceRequest("request", (_inline(sample), borrowed))

    if fail:
        with pytest.raises(RuntimeError, match="scoring failed"):
            await scorer.score_batch(request)
    else:
        scores = await scorer.score_batch(request)
        assert scores[0].scores == scores[1].scores
        assert consumer.paths[1] == borrowed_path

    local_path = consumer.paths[0]
    assert local_path.is_relative_to(temporary_root)
    assert not local_path.exists()
    assert not list(temporary_root.iterdir())
    assert borrowed_path.read_bytes() == original_bytes
    assert request.artifacts[0].path == ""
    assert request.artifacts[0].media is sample.output


@pytest.mark.parametrize(
    "metadata", [{}, {"fps": 12.0}, {"video_fps": 6.0, "fps": 12.0}, {"video_fps": None}]
)
@pytest.mark.asyncio
async def test_scorer_local_mp4_matches_legacy_disk_encoding_exactly(metadata, tmp_path) -> None:
    sample = _sample(metadata)
    store = DiskRewardArtifactStore(tmp_path / "legacy", artifact_format="mp4")
    legacy = store.materialize([sample])[0]
    consumer = _FileConsumer()
    scorer = InProcessRewardScorer(model=consumer, media_temp_dir=str(tmp_path))

    baseline = await scorer.score_batch(RewardInferenceRequest("baseline", (legacy,)))
    actual = await scorer.score_batch(RewardInferenceRequest("local", (_inline(sample),)))

    torch.testing.assert_close(consumer.frames[0], consumer.frames[1], rtol=0, atol=0)
    assert actual[0].scores == baseline[0].scores
    assert Path(legacy.path).exists()
    assert not consumer.paths[1].exists()


@pytest.mark.asyncio
async def test_memory_model_receives_tensor_without_any_local_file(tmp_path) -> None:
    sample = _sample()
    artifact = _inline(sample)

    def score(incoming):
        assert incoming.path == ""
        assert incoming.as_media() is sample.output
        assert not list(tmp_path.iterdir())
        return {"mean": float(incoming.as_media().float().mean())}

    scorer = InProcessRewardScorer(model=score, media_temp_dir=str(tmp_path))
    result = await scorer.score_batch(RewardInferenceRequest("memory", (artifact,)))

    assert result[0].scores["mean"] == float(sample.output.float().mean())
    assert not list(tmp_path.iterdir())
