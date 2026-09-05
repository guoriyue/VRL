from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.countgd import (
    COUNTGD_MODEL_VERSION,
    CountGDConfig,
    CountGDDetection,
    CountGDModel,
    CountGDResult,
    _expose_isolated_upstream,
)


def _artifact(expected_count: object, object_class: object = "person") -> RewardInferenceArtifact:
    return RewardInferenceArtifact(
        artifact_id="candidate",
        sample_id="sample",
        path="",
        media=object(),
        prompt="A scene unrelated to the explicit counting target.",
        metadata={"object_class": object_class, "expected_count": expected_count},
    )


def test_countgd_reward_uses_typed_target_instead_of_prompt_text() -> None:
    model = CountGDModel.__new__(CountGDModel)
    model.detect = lambda artifact: (object(),) * 4

    matched = model.evaluate(_artifact(4))
    missed = model.evaluate(_artifact(5))

    assert matched == CountGDResult(
        expected_count=4,
        observed_count=4,
    )
    assert matched.exact_match is True
    assert missed.exact_match is False
    assert model(_artifact(4)) == matched.to_scores() == {"countgd": 1.0}
    assert model(_artifact(5)) == missed.to_scores() == {"countgd": 0.0}
    model.detect = lambda artifact: ()
    assert model(_artifact(0, "cat")) == {"countgd": 1.0}


@pytest.mark.parametrize("value", [True, -1, 4.0, "4", None])
def test_countgd_reward_rejects_invalid_count_target(value: object) -> None:
    model = CountGDModel.__new__(CountGDModel)

    with pytest.raises(ValueError, match="non-negative integer"):
        model(_artifact(value))


@pytest.mark.parametrize("value", [None, "", "  ", ".", 3, "cat . dog", "cat? dog"])
def test_countgd_requires_explicit_class_before_loading_model(value: object) -> None:
    model = CountGDModel.__new__(CountGDModel)
    with pytest.raises(ValueError, match="object_class"):
        model(_artifact(2, value))


def test_countgd_retains_typed_detections_and_derives_the_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_captions: list[str] = []

    class FakeTransform:
        def __call__(self, image: object, target: dict[str, torch.Tensor]):
            assert image is sentinel_image
            assert target["exemplars"].shape == (0,)
            return torch.zeros((3, 2, 2)), target

    class FakeModel:
        def __call__(
            self,
            image: torch.Tensor,
            exemplars: list[torch.Tensor],
            point_indices: list[torch.Tensor],
            *,
            captions: list[str],
        ) -> dict[str, torch.Tensor]:
            assert image.shape == (1, 3, 2, 2)
            assert exemplars[0].shape == (0,)
            assert point_indices[0].tolist() == [0]
            seen_captions.extend(captions)
            return {
                "pred_logits": torch.tensor([[[-2.0], [0.0], [2.0]]]),
                "pred_boxes": torch.tensor(
                    [
                        [
                            [0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.2, 0.3],
                            [0.7, 0.8, 0.1, 0.2],
                        ]
                    ],
                ),
            }

    sentinel_image = object()
    monkeypatch.setattr(
        "vrl.rewards.models.countgd.decode_artifact_frames",
        lambda artifact, num_frames: torch.zeros((1, 2, 2, 3)),
    )
    monkeypatch.setattr(
        "vrl.rewards.models.countgd.to_pil_image",
        lambda frame: sentinel_image,
    )
    model = CountGDModel.__new__(CountGDModel)
    model.config = SimpleNamespace(device="cpu")
    model._runtime = SimpleNamespace(
        model=FakeModel(),
        transform=FakeTransform(),
        torch=torch,
    )

    detections = model.detect(_artifact(2))

    assert all(isinstance(detection, CountGDDetection) for detection in detections)
    assert detections[0].bbox_cxcywh == pytest.approx((0.5, 0.6, 0.2, 0.3))
    assert detections[1].bbox_cxcywh == pytest.approx((0.7, 0.8, 0.1, 0.2))
    assert [detection.confidence for detection in detections] == pytest.approx(
        [0.5, torch.sigmoid(torch.tensor(2.0)).item()],
    )
    assert len(detections) == 2
    assert model.score_batch([_artifact(2, " CAT. "), _artifact(3, "red apple")]) == [
        {"countgd": 1.0},
        {"countgd": 0.0},
    ]
    assert seen_captions == ["person .", "cat .", "red apple ."]


def test_countgd_model_version_binds_service_and_client_configs() -> None:
    root = Path(__file__).resolve().parents[3]
    service = OmegaConf.load(root / "vrl/config/reward_service/countgd.yaml")
    reward = OmegaConf.load(root / "vrl/config/presets/reward/countgd_http.yaml")

    assert service.model_version == COUNTGD_MODEL_VERSION
    assert service.worker_config.reward_model_version == COUNTGD_MODEL_VERSION
    assert reward.reward.inference.countgd.expected_model_version == (COUNTGD_MODEL_VERSION)
    assert (
        CountGDConfig.from_mapping(
            {"reward_model_version": COUNTGD_MODEL_VERSION},
        ).device
        == "cpu"
    )
    for advertised_version in ("", "CountGD@stale"):
        with pytest.raises(ValueError, match="does not match the executable protocol"):
            CountGDConfig.from_mapping(
                {"reward_model_version": advertised_version},
            )


def test_countgd_source_is_first_even_when_already_on_sys_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    earlier_dir = tmp_path / "earlier"
    source_dir.mkdir()
    earlier_dir.mkdir()
    for module_name in ("datasets_inference", "groundingdino", "models", "util"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(sys, "path", [str(earlier_dir), str(source_dir), *sys.path])

    _expose_isolated_upstream(source_dir)

    assert sys.path[0] == str(source_dir)
    assert sys.path.count(str(source_dir)) == 1


def test_countgd_rejects_an_already_loaded_foreign_top_level_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    foreign_module = ModuleType("models")
    foreign_module.__file__ = str(tmp_path / "foreign/models/__init__.py")
    monkeypatch.setitem(sys.modules, "models", foreign_module)

    with pytest.raises(RuntimeError, match="isolated service process"):
        _expose_isolated_upstream(source_dir)
