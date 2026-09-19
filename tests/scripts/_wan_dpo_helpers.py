"""Real tiny Wan DPO runs for the ``test_wan_dpo_*`` family.

Everything here is real code on a KB-scale snapshot; the only double is the
Pick-a-Pic Hub download, which is a network boundary: ``install_local_pickapic``
hands ``from_hub`` an in-memory ``datasets.Dataset`` of genuine JPEG pairs, so
``PickAPicPreferenceDataset.__init__`` / ``__getitem__`` / ``collate`` still run.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

# The local Pick-a-Pic double is an HF ``Dataset``; that package ships with the
# ``data`` extra, which the model-only test environments do not install. The
# repository's own ``datasets/`` asset directory imports as an empty namespace
# package from the repo root, so the module import alone cannot tell.
if not hasattr(pytest.importorskip("datasets"), "Dataset"):
    pytest.skip("HF datasets is not installed (uv extra: data)", allow_module_level=True)

from tests.scripts.eval.fixtures import write_tiny_wan_snapshot
from vrl.config.loading import load_config
from vrl.trainers.data import preferences


def tiny_dpo_run(tmp_path: Path, *, family: str = "wan_2_1", overrides: list[str] = ()) -> Any:
    """The Pick-a-Pic Wan DPO preset pointed at a tiny snapshot, one step, fp32 on CPU."""

    snapshot = tmp_path / "wan-snapshot"
    if not snapshot.exists():
        write_tiny_wan_snapshot(snapshot)
    return load_config(
        "experiment/wan_2_1/offline_dpo_pickapic",
        overrides=[
            f"model.family={family}",
            f"model.path={snapshot}",
            "model.revision=null",
            "model.use_lora=false",
            "precision.training.dtype=fp32",
            "sampling.height=32",
            "sampling.width=32",
            "trainer.max_train_steps=1",
            "trainer.checkpointing_steps=1",
            "trainer.log_interval=1",
            f"trainer.output_dir={tmp_path / 'run'}",
            "data.sampler.dataloader_num_workers=0",
            *overrides,
        ],
    )


def install_local_pickapic(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the Hub download with one real red-vs-blue preference pair.

    Returns the list of ``from_hub`` keyword calls so a test can assert the run
    reached (or, for a guard, never reached) dataset loading.
    """

    def jpeg(color: tuple[int, int, int]) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (40, 40), color).save(buffer, format="JPEG")
        return buffer.getvalue()

    rows = [
        {
            "jpg_0": jpeg((200, 10, 10)),
            "jpg_1": jpeg((10, 10, 200)),
            "label_0": 1,
            "caption": "a cat",
        }
    ]
    calls: list[dict[str, Any]] = []

    def from_hub(cls, **kwargs):
        from datasets import Dataset

        calls.append(kwargs)
        return cls(
            Dataset.from_list(rows),
            resolution=kwargs["resolution"],
            random_crop=kwargs["random_crop"],
            no_hflip=kwargs["no_hflip"],
        )

    monkeypatch.setattr(preferences.PickAPicPreferenceDataset, "from_hub", classmethod(from_hub))
    return calls


def spy_wan_from_build(monkeypatch: pytest.MonkeyPatch, on_built=None) -> list[Any]:
    """Record every real ``WanT2VDiffusersModel.from_build`` result (a spy, not a fake)."""

    from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel

    real = WanT2VDiffusersModel.from_build.__func__
    built: list[Any] = []

    def from_build(cls, build):
        model = real(cls, build)
        built.append(model)
        if on_built is not None:
            on_built(model)
        return model

    monkeypatch.setattr(WanT2VDiffusersModel, "from_build", classmethod(from_build))
    return built
