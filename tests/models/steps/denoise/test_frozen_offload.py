"""DiffusionModelBase frozen-component offload (SPRINT_frozen_component_preservation).

nn.Module.to moves only registered submodules — for diffusion families that is
just the transformer; the diffusers pipeline (with its frozen VAE / text
encoders) is attached unregistered, so it would stay GPU-resident across a
driver-model offload. ``move_frozen_components`` parks those frozen components
alongside the transformer, derived from the pipeline so the set never rots.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from vrl.models.steps.denoise.base import DiffusionModelBase


class _RecordingModule(nn.Module):
    """A tiny nn.Module that records the device passed to ``.to``."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.to_devices: list[Any] = []

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        if args:
            self.to_devices.append(args[0])
        return super().to(*args, **kwargs)


class _FakePipeline:
    """Mimics a diffusers pipeline: a ``components`` map + an unregistered hold."""

    def __init__(self, transformer: nn.Module, frozen: dict[str, Any]) -> None:
        self.transformer = transformer
        self._frozen = frozen

    @property
    def components(self) -> dict[str, Any]:
        # diffusers returns transformer + frozen nn.Modules + non-module entries
        # (schedulers/tokenizers); the last must be ignored by the derivation.
        return {"transformer": self.transformer, "scheduler": object(), **self._frozen}


class _TinyDiffusionModel(DiffusionModelBase):
    """Minimal concrete family: registers only the transformer, like SD3.5."""

    def __init__(self, pipeline: _FakePipeline) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    # abstractmethod stubs — unused by the offload path
    def encode_prompt(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_sampling(self, *a: Any, **k: Any) -> Any:
        raise NotImplementedError

    def forward_step(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def decode_latents(self, *a: Any, **k: Any) -> Any:
        raise NotImplementedError


def _model() -> tuple[_TinyDiffusionModel, _RecordingModule, dict[str, _RecordingModule]]:
    transformer = _RecordingModule()
    frozen = {"vae": _RecordingModule(), "text_encoder": _RecordingModule()}
    model = _TinyDiffusionModel(_FakePipeline(transformer, frozen))
    return model, transformer, frozen


def test_move_frozen_components_moves_only_frozen() -> None:
    model, transformer, frozen = _model()
    # pipeline-derived: the frozen VAE + text encoder move, and ONLY those — the
    # transformer (rides nn.Module.to / weight-sync) and the non-module scheduler
    # are excluded (the scheduler has no .to, so a wrong selection would raise).
    model.move_frozen_components("cpu")
    assert frozen["vae"].to_devices == ["cpu"]
    assert frozen["text_encoder"].to_devices == ["cpu"]
    assert transformer.to_devices == []


def test_no_pipeline_is_a_safe_no_op() -> None:
    """Replay models / single-file checkpoints expose no pipeline -> empty set."""

    class _NoPipeline(_TinyDiffusionModel):
        @property
        def pipeline(self) -> Any:
            raise RuntimeError("replay model has no pipeline")

    model, _, _ = _model()
    no_pipe = _NoPipeline(model._pipeline)
    object.__setattr__(no_pipe, "_pipeline", None)
    no_pipe.move_frozen_components("cpu")  # must not raise (moves nothing)
