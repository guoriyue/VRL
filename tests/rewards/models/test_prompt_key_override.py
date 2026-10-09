"""A torch reward model scores against ``metadata[prompt_key]`` when configured."""

from __future__ import annotations

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.base import TorchRewardModel


class _EchoPrompt(TorchRewardModel):
    def _load_module(self):
        return object()

    def score_media(self, *, media, prompt):
        return {"seen": prompt}


def _artifact(metadata):
    return RewardInferenceArtifact(
        artifact_id="a",
        sample_id="s",
        path="",
        prompt="nude figure on a beach",
        metadata=metadata,
        media="pixels",
    )


def test_prompt_key_reads_the_metadata_text() -> None:
    model = _EchoPrompt({"device": "cpu", "prompt_key": "safe_prompt"})
    model.prepare_for_inference = lambda: None

    assert model(_artifact({"safe_prompt": "figure in a long dress on a beach"})) == {
        "seen": "figure in a long dress on a beach"
    }


def test_prompt_key_falls_back_to_the_prompt_when_missing_or_blank() -> None:
    model = _EchoPrompt({"device": "cpu", "prompt_key": "safe_prompt"})
    model.prepare_for_inference = lambda: None

    assert model(_artifact({}))["seen"] == "nude figure on a beach"
    assert model(_artifact({"safe_prompt": "  "}))["seen"] == "nude figure on a beach"


def test_without_prompt_key_the_prompt_is_used() -> None:
    model = _EchoPrompt({"device": "cpu"})
    model.prepare_for_inference = lambda: None

    assert model(_artifact({"safe_prompt": "ignored"}))["seen"] == "nude figure on a beach"
