"""Real objects shared by the SANA eval-script tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from vrl.scripts.eval.sana_inference import SANA_EVAL_SCHEDULER_CONFIG


def build_official_sana_scheduler(**overrides: Any) -> Any:
    """The real ``DPMSolverMultistepScheduler`` at SANA's official protocol.

    Config-init, no download. An override is how the drift tests produce a
    scheduler that must be REJECTED, and because the object is genuine an
    upstream rename of any protocol key turns the accept case red instead of
    letting a double echo the table back.
    """

    from diffusers import DPMSolverMultistepScheduler

    kwargs = {
        key: value for key, value in SANA_EVAL_SCHEDULER_CONFIG.items() if key != "class_name"
    }
    kwargs.update(overrides)
    return DPMSolverMultistepScheduler(**kwargs)


class _TinyTextEncoder(torch.nn.Module):
    """A frozen encoder module that, like an HF model, reports its ``dtype``."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    @property
    def dtype(self) -> torch.dtype:
        return self.proj.weight.dtype


class TinySanaPipeline:
    """A ``SanaPipeline`` stand-in over REAL tiny modules.

    Everything the shared loader and the eval scripts touch is genuine -- a
    ``SanaTransformer2DModel``, an ``AutoencoderKL``, a frozen text-encoder
    module, the official DPM-Solver++ scheduler -- so ``SanaModel.from_build``,
    the rollout bundle, checkpoint save/restore and the local identity hash all
    run for real. Only ``__call__`` (diffusers' denoising loop, which needs the
    Gemma encoder) is a double: it records the call and paints a solid image
    whose colour says which weights the transformer held at that moment, so
    "base before restore, current after" is witnessed by the weights themselves.
    """

    BASE_COLOR = (10, 20, 200)
    RESTORED_COLOR = (200, 10, 20)

    def __init__(self) -> None:

        from tests.models.steps.denoise.fixtures import (
            build_tiny_autoencoder_kl,
            build_tiny_sana_transformer,
        )

        self.transformer = build_tiny_sana_transformer()
        self.vae = build_tiny_autoencoder_kl()
        self.text_encoder = _TinyTextEncoder()
        self.scheduler = build_official_sana_scheduler()
        self.calls: list[dict[str, Any]] = []
        self.loads = 0
        self._labels: dict[float, str] = {self.fingerprint(): "base"}

    def fingerprint(self) -> float:
        return float(sum(p.detach().double().sum().item() for p in self.transformer.parameters()))

    def label_weights(self, label: str) -> None:
        """Name the transformer's current weights so a later call can report them."""

        self._labels[self.fingerprint()] = label

    def holds(self) -> str:
        return self._labels.get(self.fingerprint(), "restored")

    def __call__(self, **kwargs: Any) -> Any:
        import torch
        from PIL import Image

        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert "complex_human_instruction" not in kwargs
        held = self.holds()
        self.calls.append(
            {
                **kwargs,
                "scheduler": self.scheduler,
                "generator_seed": kwargs["generator"].initial_seed(),
                "weights": held,
            },
        )
        color = self.BASE_COLOR if held == "base" else self.RESTORED_COLOR
        count = int(kwargs.get("num_images_per_prompt", 1))
        images = [
            Image.new("RGB", (kwargs["width"], kwargs["height"]), color=color)
            for _ in range(count)
        ]
        return SimpleNamespace(images=images)

    def install(self, monkeypatch: Any, snapshot: Path, *, on_load: Any = None) -> None:
        """Serve this pipeline for ``SanaPipeline.from_pretrained(snapshot)``.

        The HF load is the one external boundary; ``on_load`` lets a test act
        while the model is being constructed (e.g. mutate the snapshot tree).
        """

        from diffusers import SanaPipeline

        def from_pretrained(path: Any, **_kwargs: Any) -> TinySanaPipeline:
            assert Path(str(path)).resolve() == snapshot.resolve(), path
            self.loads += 1
            if on_load is not None:
                on_load()
            return self

        monkeypatch.setattr(SanaPipeline, "from_pretrained", staticmethod(from_pretrained))


def write_tiny_sana_snapshot(path: Path) -> Path:
    """A local model directory: the official scheduler config plus a weights marker.

    ``load_official_scheduler`` reads ``scheduler/`` from it for real
    (``local_files_only``), and the checkpoint identity is the hash of this tree.
    """

    path.mkdir(parents=True, exist_ok=True)
    build_official_sana_scheduler().save_pretrained(path / "scheduler")
    (path / "model_index.json").write_text('{"_class_name": "SanaPipeline"}\n', encoding="utf-8")
    return path


__all__ = ["TinySanaPipeline", "build_official_sana_scheduler", "write_tiny_sana_snapshot"]
