"""Region-bound transcripts reject misplaced/extra text and preserve raw evidence."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.text_regions import TextRegionsRewardModel
from vrl.rewards.types import RewardSample


@pytest.mark.asyncio
async def test_region_location_and_extra_words_cannot_get_substring_full_credit(tmp_path):
    class Engine:
        def predict(self, frame):
            transcripts = {40: "Wait here.", 80: "Meet me at the bridge.", 120: "Wait here. EXTRA"}
            return [{"rec_texts": [transcripts[int(frame[0, 0, 0])]], "rec_scores": [0.9]}]

    page = np.full((40, 80, 3), 40, dtype=np.uint8)
    page[20:] = 80
    layout = {
        "width": 80,
        "height": 40,
        "regions": [
            {"region_id": "panel-1", "text": "Wait here.", "box": [0, 0, 80, 20]},
            {
                "region_id": "panel-2",
                "text": "Meet me at the bridge.",
                "box": [0, 20, 80, 40],
            },
        ],
    }
    swapped = page[::-1].copy()
    extra = page.copy()
    extra[:20] = 120
    reward = MultiReward.from_dict(
        {"text_regions": 1.0},
        device="cuda",
        reward_kwargs={"text_regions": {"worker_config": {"engine": Engine()}}},
        inference_configs={"text_regions": RewardInferenceConfig(kind="in_process")},
    )
    try:
        output = await reward.score_batch(
            [
                RewardSample(
                    prompt="Keep dialogue in its assigned panel",
                    output=torch.from_numpy(pixels).permute(2, 0, 1).float() / 255,
                    sample_id=str(index),
                    metadata={"text_layout": layout},
                )
                for index, pixels in enumerate((page, swapped, extra))
            ]
        )
    finally:
        await reward.shutdown()
    assert output.scores[0] == 1
    assert all(score < 1 for score in output.scores[1:])
    assert output.components["text_regions/text_exact_fraction"] == (1, 0, 0.5)
    assert output.components["text_regions/text_worst_region"][2] < output.scores[2]
    model = TextRegionsRewardModel({"engine": Engine()})
    result = model.score_results(
        [
            RewardInferenceArtifact(
                "a", "a", "", media=Image.fromarray(extra), metadata={"text_layout": layout}
            )
        ]
    )[0]
    assert result.diagnostics["regions"][0]["recognized"] == "Wait here. EXTRA"
    assert result.diagnostics["regions"][1]["exact"]
    path = tmp_path / "extra.png"
    Image.fromarray(extra).save(path)
    file_result = model.score_results(
        [RewardInferenceArtifact("a", "a", str(path), metadata={"text_layout": layout})]
    )[0]
    assert file_result == result


def test_missing_regions_and_changed_canvas_fail_before_ocr():
    class MustNotRun:
        def predict(self, frame):
            raise AssertionError("invalid task must fail before OCR")

    model = TextRegionsRewardModel({"engine": MustNotRun()})
    artifact = RewardInferenceArtifact("a", "a", "", media=Image.new("RGB", (40, 20)))
    with pytest.raises(ValueError):
        model(artifact)
    layout = {
        "width": 40,
        "height": 20,
        "regions": [{"region_id": "a", "text": "hello", "box": [0, 0, 41, 20]}],
    }
    artifact.metadata["text_layout"] = layout
    with pytest.raises(ValueError, match="inside the canvas"):
        model(artifact)
    layout["regions"][0]["box"] = [0, 0, 40, 20]
    layout["height"] = 21
    with pytest.raises(ValueError, match="declared canvas"):
        model(artifact)
    layout["height"] = 20
    layout["regions"].append(copy.deepcopy(layout["regions"][0]))
    with pytest.raises(ValueError, match="unique"):
        model(artifact)


def test_empty_recognition_counts_and_whitespace_does_not_erase_case_or_punctuation():
    class Engine:
        def predict(self, frame):
            return (
                {"rec_texts": ["Hello", "world!"], "rec_scores": [0.9, 0.8]}
                if frame[0, 0, 0]
                else None
            )

    layout = {
        "width": 20,
        "height": 20,
        "regions": [{"region_id": "a", "text": "Hello\n world!", "box": [0, 0, 20, 20]}],
    }
    artifact = RewardInferenceArtifact(
        "a", "a", "", media=Image.new("RGB", (20, 20), "white"), metadata={"text_layout": layout}
    )
    model = TextRegionsRewardModel({"engine": Engine()})
    assert model(artifact)["text_similarity"] == 1
    artifact.metadata["text_layout"]["regions"][0]["text"] = "hello world"
    assert model(artifact)["text_exact_fraction"] == 0
    artifact = replace(artifact, media=Image.new("RGB", (20, 20), "black"))
    result = model(artifact)
    assert result["text_similarity"] == 0
    assert result["text_empty_fraction"] == 1


def test_ignoring_punctuation_forgives_line_end_periods_but_not_words():
    class Engine:
        def predict(self, frame):
            return {
                "rec_texts": ["Wait here.", "until. the train", "arrives.."],
                "rec_scores": [0.9, 0.9, 0.9],
            }

    layout = {
        "width": 20,
        "height": 20,
        "regions": [
            {"region_id": "a", "text": "Wait here until the train arrives.", "box": [0, 0, 20, 20]}
        ],
    }
    artifact = RewardInferenceArtifact(
        "a", "a", "", media=Image.new("RGB", (20, 20), "white"), metadata={"text_layout": layout}
    )
    strict = TextRegionsRewardModel({"engine": Engine()})(artifact)
    lenient = TextRegionsRewardModel({"engine": Engine(), "punctuation": "ignore"})(artifact)
    assert strict["text_exact_fraction"] == 0 and strict["text_similarity"] < 1
    assert lenient["text_exact_fraction"] == 1 and lenient["text_similarity"] == 1
    layout["regions"][0]["text"] = "Wait here until the coach arrives."
    assert (
        TextRegionsRewardModel({"engine": Engine(), "punctuation": "ignore"})(artifact)[
            "text_exact_fraction"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="keep or ignore"):
        TextRegionsRewardModel({"engine": Engine(), "punctuation": "loose"})


def test_text_focus_requires_the_new_word_and_the_old_word_gone():
    class Engine:
        def __init__(self, text):
            self.text = text

        def predict(self, frame):
            return {"rec_texts": [self.text], "rec_scores": [0.9]}

    layout = {
        "width": 20,
        "height": 20,
        "regions": [{"region_id": "a", "text": "Follow the purple signs.", "box": [0, 0, 20, 20]}],
    }
    focus = {"region_id": "a", "must_contain": "purple", "must_not_contain": "yellow"}

    def run(text):
        artifact = RewardInferenceArtifact(
            "a",
            "a",
            "",
            media=Image.new("RGB", (20, 20), "white"),
            metadata={"text_layout": layout, "text_focus": focus},
        )
        return TextRegionsRewardModel({"engine": Engine(text), "punctuation": "ignore"})(artifact)

    assert run("Follow the purple signs..")["text_focus_done"] == 1
    assert run("Follow the yellow signs.")["text_focus_done"] == 0
    assert run("Follow the purlple signs.")["text_focus_done"] == 0
    assert run("Follow the purple yellow signs.")["text_focus_done"] == 0
    with pytest.raises(ValueError, match="unknown region"):
        TextRegionsRewardModel({"engine": Engine("x")})(
            RewardInferenceArtifact(
                "a",
                "a",
                "",
                media=Image.new("RGB", (20, 20), "white"),
                metadata={"text_layout": layout, "text_focus": {**focus, "region_id": "zz"}},
            )
        )


def test_cumulative_focus_charges_reverted_earlier_words_in_long_transcripts():
    class Engine:
        def predict(self, frame):
            text = {
                40: "The cap is here beside the door. Wait for me before leaving.",
                80: "The map is here beside the door. Wait for me before leaving.",
                120: "Go south.",
            }[int(frame[0, 0, 0])]
            return [{"rec_texts": [text], "rec_scores": [0.99]}]

    layout = {
        "width": 20,
        "height": 40,
        "regions": [
            {
                "region_id": "a",
                "text": "The cap is here beside the door. Wait for me before leaving.",
                "box": [0, 0, 20, 20],
            },
            {"region_id": "b", "text": "Go south.", "box": [0, 20, 20, 40]},
        ],
    }
    previous = {"region_id": "a", "must_contain": "cap", "must_not_contain": "map"}
    current = {"region_id": "b", "must_contain": "south", "must_not_contain": "north"}
    page = np.full((40, 20, 3), 40, dtype=np.uint8)
    page[20:] = 120
    model = TextRegionsRewardModel({"engine": Engine(), "punctuation": "ignore"})
    artifact = RewardInferenceArtifact(
        "a",
        "a",
        "",
        media=Image.fromarray(page),
        metadata={"text_layout": layout, "text_focus": [previous, current]},
    )
    assert model(artifact)["text_focus_done"] == 1
    page[:20] = 80
    artifact = replace(artifact, media=Image.fromarray(page))
    reverted = model(artifact)
    assert reverted["text_worst_region"] > 0.98
    assert reverted["text_focus_done"] == 0.5
    # The old single-step specification still reports this current edit as done.
    artifact.metadata["text_focus"] = current
    assert model(artifact)["text_focus_done"] == 1
    artifact.metadata["text_focus"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        model(artifact)
