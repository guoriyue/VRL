"""The shared Qwen-VL judge prelude on a tiny real generative Qwen2-VL.

``QwenVLVideoJudge`` owns processor + model loading, the chat turn, the
processor call, generation, and decoding for VideoScore2 and the
reasoner. A ~35K-parameter ``Qwen2VLForConditionalGeneration`` written with
``save_pretrained`` (``tests/rewards/fixtures.py``) drives the real path on CPU;
only the judge's own text is substituted where a score line is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.rewards.fixtures import build_tiny_qwen_vl_judge_repo
from vrl.utils.media import write_mp4

pytest.importorskip("qwen_vl_utils")

_WORKER_CONFIG = {
    "device": "cpu",
    "dtype": "float32",
    "local_files_only": True,
    "max_new_tokens": 8,
    # Both bounds: qwen-vl-utils' default video minimum is far above this tiny
    # budget and it asserts max >= min.
    "max_frame_pixels": 28 * 28 * 4,
    "min_frame_pixels": 28 * 28 * 4,
}


@pytest.fixture(scope="module")
def judge_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_qwen_vl_judge_repo(tmp_path_factory.mktemp("tiny-qwen-vl-judge"))


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "clip.mp4"
    write_mp4(torch.rand(3, 8, 32, 32), path, fps=4.0)
    return path


def _videoscore2(judge_repo: Path, **overrides):
    from vrl.rewards.models.videoscore2 import VideoScore2Model

    return VideoScore2Model({**_WORKER_CONFIG, "model_path": str(judge_repo), **overrides})


def test_judge_loads_offline_and_runs_the_real_video_pipeline(
    judge_repo: Path, clip: Path
) -> None:
    """Load, chat template, video decode, processor, and generation all run for real;
    the tiny judge's babble then fails the parser with the judge's own message."""

    judge = _videoscore2(judge_repo)

    inputs, messages = judge._judge_inputs(str(clip), "a red fox")
    assert messages[0]["role"] == "system"
    assert messages[1]["content"][0]["max_pixels"] == 28 * 28 * 4
    assert "a red fox" in messages[1]["content"][1]["text"]
    assert inputs["pixel_values_videos"].ndim == 2
    assert inputs["input_ids"].device.type == judge.model.device.type == "cpu"

    with pytest.raises(ValueError, match="no parseable score line"):
        judge._score_video(str(clip), "a red fox")


def test_videoscore2_parses_hard_scores_from_the_generated_text(
    judge_repo: Path,
    clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the generation substituted, the shared decode + parse path yields scores."""

    judge = _videoscore2(judge_repo, soft_scores=False)
    answer = judge.tokenizer.encode(
        "visual quality: 4; text-to-video alignment: 3, physical/common-sense consistency: 5",
        add_special_tokens=False,
    )
    seen: dict[str, object] = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        prompt_ids = kwargs["input_ids"]
        return SimpleNamespace(
            sequences=torch.cat([prompt_ids, torch.tensor([answer])], dim=1),
            scores=None,
        )

    monkeypatch.setattr(judge.model, "generate", fake_generate)

    scores = judge._score_video(str(clip), "a red fox")

    assert scores == {
        "visual_quality": 4.0,
        "text_alignment": 3.0,
        "physical_common_sense": 5.0,
        "overall": 4.0,
    }
    # VideoScore2's shipped recipe, not the greedy loop that degenerates on real video.
    assert seen["do_sample"] is True
    assert seen["temperature"] == pytest.approx(1e-6)
    assert seen["max_new_tokens"] == 8
