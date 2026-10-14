from __future__ import annotations

import torch
from torch import nn

from vrl.trainers.online.ema import EMAWeights

UNIT_IDENTITY = {"schema": "unit-model/v1"}


def _ema_holding(
    module: nn.Module,
    *,
    average: float,
    live: float,
    stepped: bool = True,
) -> EMAWeights:
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
    ema = EMAWeights(trainable, decay=0.9, device=torch.device("cpu"))
    if stepped:
        ema.step(trainable, 0)
    with torch.no_grad():
        for parameter in trainable:
            parameter.fill_(live)
    return ema


class _Trainer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {"step": 2, "global_step": 5}

    def load_state_dict(self, state):
        self.loaded = dict(state)


class _Bundle:
    def __init__(self, module=None) -> None:
        import torch.nn as nn

        self.module = module or nn.Linear(1, 1, bias=False)
        self.model = self.module
        self.trainable_modules = {"module": self.module}
