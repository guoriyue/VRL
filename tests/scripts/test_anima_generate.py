from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from vrl.scripts.diffusion.cosmos.anima import generate


def test_generate_disables_empty_training_lora_for_inference() -> None:
    """Checks generate disables empty training LoRA for inference."""
    cfg = OmegaConf.create({"model": {"use_lora": True, "lora": {"path": ""}}})

    generate._configure_lora_for_inference(
        cfg,
        lora_path="",
        use_config_lora=False,
    )

    assert cfg.model.use_lora is False


def test_generate_accepts_checkpoint_dir_as_lora_path(tmp_path) -> None:
    """Checks generate accepts checkpoint dir as LoRA path."""
    checkpoint = tmp_path / "checkpoint-final"
    exported = checkpoint / "lora_weights"
    exported.mkdir(parents=True)

    assert generate._resolve_lora_path(str(checkpoint)) == exported


def test_generate_image_conversion_accepts_chw_float() -> None:
    """Checks generate image conversion accepts chw float."""
    from vrl.utils.media import image_to_uint8_hwc

    image = np.zeros((3, 2, 2), dtype=np.float32)
    image[0] = 1.0

    out = image_to_uint8_hwc(image)

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    assert out[..., 0].max() == 255


def test_generate_sampling_defaults_follow_config() -> None:
    """Checks generate sampling defaults follow config."""
    cfg = OmegaConf.create(
        {
            "sampling": {
                "width": 768,
                "height": 512,
                "num_steps": 12,
                "guidance_scale": 4.0,
                "max_sequence_length": 256,
            },
        },
    )
    args = generate.build_parser().parse_args(["--prompt", "adult anime portrait"])

    sampling = generate._resolve_sampling(args, cfg)

    assert sampling["width"] == 768
    assert sampling["height"] == 512
    assert sampling["num_steps"] == 12
    assert sampling["guidance_scale"] == 4.0
    assert sampling["max_sequence_length"] == 256


def test_generate_sampling_preserves_explicit_zero_guidance() -> None:
    """An explicit zero disables CFG instead of falling back to the config."""
    cfg = OmegaConf.create({"sampling": {"guidance_scale": 4.0}})
    args = generate.build_parser().parse_args(
        ["--prompt", "adult anime portrait", "--guidance-scale", "0"],
    )

    sampling = generate._resolve_sampling(args, cfg)

    assert sampling["guidance_scale"] == 0.0


def test_generate_records_the_batch_seed_for_every_sample(monkeypatch, tmp_path) -> None:
    """Rows share the one generator seed used for their prompt batch."""
    cfg = OmegaConf.create(
        {
            "model": {
                "path": "unused",
                "use_lora": False,
                "lora": {"path": ""},
            },
            "sampling": {
                "width": 2,
                "height": 2,
                "num_steps": 1,
                "guidance_scale": 1.0,
                "max_sequence_length": 8,
            },
        },
    )

    class _Model:
        def eval(self):
            return self

    monkeypatch.setattr(generate, "load_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(generate, "extract_family_runtime_spec", lambda *_args: object())
    monkeypatch.setattr(
        generate,
        "build_family_runtime_bundle",
        lambda _spec: SimpleNamespace(model=_Model()),
    )
    monkeypatch.setattr(
        generate,
        "_generate_images",
        lambda *_args, **_kwargs: [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
    )

    output_dir = tmp_path / "generated"
    generate.main(
        [
            "--prompt",
            "adult anime portrait",
            "--samples-per-prompt",
            "2",
            "--seed",
            "37",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--output-dir",
            str(output_dir),
        ],
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["seed"] for row in rows] == [37, 37]


def test_generate_cuda_device_fails_fast_when_unavailable() -> None:
    """Checks generate cuda device fails fast when unavailable."""

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

        @staticmethod
        def device(value: str) -> object:
            return type("Device", (), {"type": value.split(":", 1)[0]})()

    with pytest.raises(RuntimeError, match="CUDA device was requested"):
        generate._resolve_device("cuda:0", _Torch)
