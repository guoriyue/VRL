from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import torch
from omegaconf import OmegaConf

from vrl.scripts.eval import cosmos_predict25_frame_prefix_gate as gate


def test_prepare_prefix_uses_real_tail_frames_and_pads_last(monkeypatch, tmp_path: Path) -> None:
    source = torch.zeros(7, 2, 3, 3)
    source[-5:] = torch.linspace(0.1, 0.5, 5).reshape(5, 1, 1, 1)
    monkeypatch.setattr(gate, "read_video_frames", lambda _path: source)

    video = gate._prepare_prefix_video(
        tmp_path / "prefix.mp4",
        prefix_frames=5,
        num_frames=9,
        height=4,
        width=6,
    )

    assert video.shape == (1, 3, 9, 4, 6)
    assert torch.allclose(video[:, :, 5:], video[:, :, 4:5].expand(-1, -1, 4, -1, -1))
    assert video.amin() >= -1.0
    assert video.amax() <= 1.0


def test_parser_requires_a_real_prefix_video() -> None:
    args = gate.build_parser().parse_args(["--prefix-video", "prefix.mp4"])

    assert args.prefix_frames == 5
    assert args.prefix_video == Path("prefix.mp4")


def test_run_gate_loads_through_production_resolve_and_materialize(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "prefix.mp4"
    prefix.write_bytes(b"video")
    calls: list[str] = []

    class FakePipeline:
        vae = SimpleNamespace(dtype=torch.float32)
        transformer = SimpleNamespace(config=SimpleNamespace(in_channels=17))

        def prepare_latents(self, **kwargs):
            calls.append("prepare_latents")
            assert kwargs["num_frames_in"] == 5
            assert kwargs["video"].shape == (1, 3, 9, 8, 8)
            latents = torch.zeros(1, 16, 3, 1, 1)
            condition = torch.ones_like(latents)
            mask = torch.zeros(1, 1, 3, 1, 1)
            mask[:, :, :2] = 1
            return latents, condition, mask, mask.clone()

    class FakeResolved:
        identity: ClassVar[dict[str, str]] = {"schema": "test-model/v1"}

        def materialize(self, *, context: str):
            calls.append("materialize")
            assert "frame-prefix" in context
            model = SimpleNamespace(pipeline=FakePipeline(), eval=lambda: model)
            return SimpleNamespace(model=model)

    entry = SimpleNamespace(family="cosmos-predict2.5")
    root = SimpleNamespace(
        model=SimpleNamespace(family="cosmos-predict2.5"),
        precision=object(),
        sampling=SimpleNamespace(height=8, width=8, num_frames=9),
    )
    monkeypatch.setattr(
        gate,
        "load_config",
        lambda *_args, **_kwargs: OmegaConf.create(
            {"sampling": {"height": 8, "width": 8, "num_frames": 9}},
        ),
    )
    monkeypatch.setattr(gate, "parse_config", lambda _cfg: root)
    monkeypatch.setattr(
        gate.PrecisionPolicy,
        "from_section",
        classmethod(lambda _cls, _section: object()),
    )
    monkeypatch.setattr(gate, "get_model_family_entry", lambda _family: entry)
    monkeypatch.setattr(gate, "resolve_eval_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(
        gate.run,
        "resolve_model",
        lambda *_args, **_kwargs: calls.append("resolve_model") or FakeResolved(),
    )
    monkeypatch.setattr(
        gate,
        "_prepare_prefix_video",
        lambda *_args, **_kwargs: torch.zeros(1, 3, 9, 8, 8),
    )
    monkeypatch.setattr(gate, "release_cuda_memory", lambda: None)
    args = gate.build_parser().parse_args(["--prefix-video", str(prefix), "--device", "cpu"])

    report = gate.run_gate(args)

    assert calls == ["resolve_model", "materialize", "prepare_latents"]
    assert report["schema"] == gate.REPORT_SCHEMA
    assert report["status"] == "passed"
    assert "pipeline constructs" in report["claim"]
    assert "wrapper/forward integration" in report["non_goal"]
