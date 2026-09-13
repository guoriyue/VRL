from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from tests.scripts.eval.fixtures import cosmos25_eval_config, write_tiny_cosmos25_snapshot
from vrl.scripts.eval import cosmos_predict25_frame_prefix_gate as gate
from vrl.utils.media import write_mp4


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


def test_run_gate_loads_through_production_resolve_and_materialize(tmp_path: Path) -> None:
    """The gate runs unpatched: real config, real tiny Cosmos-2.5 snapshot, real prefix mp4.

    ``resolve_model`` / ``materialize`` build the real LoRA bundle from the
    on-disk snapshot, ``_prepare_prefix_video`` decodes a real libx264 mp4, and
    ``pipe.prepare_latents`` runs the real conditioning math; the report is the
    script's own verdict on those tensors.
    """

    snapshot = write_tiny_cosmos25_snapshot(tmp_path / "cosmos-snapshot")
    config_path = tmp_path / "resolved_config.yaml"
    OmegaConf.save(cosmos25_eval_config(snapshot, num_frames=9), config_path)
    prefix = tmp_path / "prefix.mp4"
    write_mp4(torch.rand(3, 9, 32, 32), prefix, fps=4.0)

    args = gate.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--prefix-video",
            str(prefix),
            "--prefix-frames",
            "5",
            "--device",
            "cpu",
        ],
    )
    report = gate.run_gate(args)

    assert report["schema"] == gate.REPORT_SCHEMA
    assert report["status"] == "passed"
    assert report["prefix_video"]["conditioned_pixel_frames"] == 5
    assert report["sampling"] == {"height": 32, "width": 32, "num_frames": 9}
    assert report["tensors"]["latents"] == report["tensors"]["condition"]
    assert "pipeline constructs" in report["claim"]
    assert "wrapper/forward integration" in report["non_goal"]
