from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import RuntimeBundle, register_checkpoint_owned_state
from vrl.models.steps.denoise.base import DiffusionModelBase
from vrl.models.steps.token.base import ARModelBase
from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    TrainingCheckpoint,
)
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.online.ema import EMAModuleWrapper

UNIT_IDENTITY = {"schema": "unit-model/v1"}


_EXPORT_PRECISION = RolePrecision(
    dtype="fp32",
    float32_precision="ieee",
    outer_autocast=False,
)


class _PublishableModule(nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1, bias=False)

    def save_pretrained(self, *_args, **_kwargs):  # pragma: no cover - never called
        raise AssertionError("build_adapter_exports must not write anything")


class _DenoisePolicy(DiffusionModelBase):
    """Minimal real diffusion policy: only ``trainable_modules`` is family data."""

    def __init__(self, roots: dict[str, nn.Module]) -> None:
        super().__init__()
        self._roots = roots

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        return dict(self._roots)

    def encode_prompt(self, prompt, negative_prompt=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def prepare_sampling(self, request, encoded, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def forward_step(self, state, step_idx):  # pragma: no cover
        raise NotImplementedError

    def decode_latents(self, latents):  # pragma: no cover
        raise NotImplementedError


class _TokenPolicy(ARModelBase):
    """Minimal real AR policy: LoRA lives on ``language_model``, one hop in."""

    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        # What ``build_token_family_bundle`` registers: the whole wrapper.
        return {"model": self}


def _export_bundle(model) -> RuntimeBundle:
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=None,
        raw_handle=None,
        precision=_EXPORT_PRECISION,
        loads_full_generation_modules=False,
        adapter_roots=model.adapter_roots,
    )


def _context(*, rank: int = 0, world_size: int = 1) -> DistributedTrainingContext:
    """The real context the checkpoint reads ``is_primary`` / ``world_size`` off.

    ``is_primary`` is derived (``rank == 0``), so the two can no longer be written
    independently: a hand-rolled namespace let a test claim "not primary, world
    size 1", a state ``DistributedTrainingContext.from_root`` cannot produce. ``strategy``
    follows the same rule the resolver enforces — ``single_process`` is world 1.

    The context is real; the strategies that carry it stay recording doubles,
    because a ``world_size=2`` rank-agreement *sequence* cannot be produced by one
    process. Those sequences run for real on gloo in
    ``tests/trainers/test_fsdp_gather_distributed.py`` (default lane).
    """

    return DistributedTrainingContext(
        strategy="single_process" if world_size == 1 else "fsdp",
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
    )


def _ema_holding(
    module: nn.Module,
    *,
    average: float,
    live: float,
    stepped: bool = True,
) -> EMAModuleWrapper:
    """A real EMA whose stored average differs from the module's live weights.

    ``has_updates`` is derived from ``num_updates`` in the real wrapper, so a
    ``step()`` is what makes it True — the hand-written doubles this replaces could
    simply declare it, and they had drifted apart on whether they even recorded
    the parameters they were handed. The step runs while the weights still equal
    ``average`` so the running average stays exactly there, which is what makes
    the EMA artifact distinguishable from the raw one on disk.
    """

    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    with torch.no_grad():
        for parameter in trainable:
            parameter.fill_(average)
    ema = EMAModuleWrapper(trainable, decay=0.9, device=torch.device("cpu"))
    if stepped:
        ema.step(trainable, 0)
    with torch.no_grad():
        for parameter in trainable:
            parameter.fill_(live)
    return ema


def _exported_adapter_weight(path: Path) -> torch.Tensor:
    """Read an exported adapter back through safetensors.

    ``write_text("stub")`` produced a five-byte file that no safetensors reader can
    open, so ``.exists()`` was the strongest claim available. Loading it proves the
    artifact is the real container with the real tensor in it.
    """

    return load_file(path)["weight"]


class _Trainer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {"step": 2, "global_step": 5}

    def load_state_dict(self, state, *, strict=True):
        del strict
        self.loaded = dict(state)


class _Bundle:
    def __init__(self, module=None) -> None:
        import torch.nn as nn

        self.module = module or nn.Linear(1, 1, bias=False)
        self.model = self.module
        self.trainable_modules = {"module": self.module}


class _OwnedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.frozen_base = nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.register_buffer("previous", torch.tensor([3.0]))
        register_checkpoint_owned_state(self, ["previous"])


class _OwnedBundle:
    def __init__(self) -> None:
        self.module = _OwnedModule()
        self.trainable_modules = {"module": self.module}


def _training_checkpoint(tmp_path, payload) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload=payload,
        meta={},
    )


def _v1_payload(
    state: dict[str, torch.Tensor],
    *,
    identity: dict | None = None,
) -> dict:
    model = {"trainable_modules": {"module": state}}
    if identity is not None:
        model["identity"] = identity
    return {
        "schema_version": 1,
        "family": "unit",
        "trainer": {"step": 2, "global_step": 5},
        "model": model,
        "progress": {"next_epoch": 2},
        "rng": {},
    }
