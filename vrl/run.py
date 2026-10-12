"""Run-level trainer policy every training entrypoint shares.

An entrypoint reads its run from ``BuiltConfigs.from_cfg`` and the resource
plan ``ResourcePlan.from_root`` once, then hands each role what it needs: the
trainer its device off the plan, the reward its
``RewardRuntimeConfig.build_function``, the rollout its
``RayGenerationLaunchInputs.from_run``. The model is a ``ModelBuild`` from
``ModelFamilyEntry.resolve_model_build`` plus the checkpoint identity it
resolves to; the entry's ``build_replay``/``build_rollout`` hold the built
bundle to that identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.config.schema import RootConfig
from vrl.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class OnlineRunConfig:
    """Controller-owned epoch, checkpoint cadence, and trainer RNG policy."""

    total_epochs: int
    save_freq: int = 50
    seed: int = 0
    deterministic: bool = False

    def __post_init__(self) -> None:
        # RootConfig types these fields as StrictInt/StrictBool; only the
        # ranges the schema does not express are checked here.
        if self.total_epochs < 0:
            raise ValueError("trainer.total_epochs must be >= 0")
        if self.save_freq < 0:
            raise ValueError("trainer.save_freq must be >= 0")
        if not -(2**63) <= self.seed < 2**64:
            raise ValueError("trainer.seed must fit torch's signed/unsigned 64-bit seed range")

    @classmethod
    def from_root(cls, root: RootConfig) -> OnlineRunConfig:
        trainer = root.trainer
        total_epochs = None if trainer is None else trainer.total_epochs
        if total_epochs is None:
            raise ValueError("config missing required key: trainer.total_epochs")
        values: dict[str, Any] = {"total_epochs": total_epochs}
        if trainer.save_freq is not None:
            values["save_freq"] = trainer.save_freq
        if trainer.seed is not None:
            values["seed"] = trainer.seed
        if trainer.deterministic is not None:
            values["deterministic"] = trainer.deterministic
        return cls(**values)

    def initialize_process_rng(self, *, rank: int = 0) -> None:
        """Seed model construction identically, then rank-local training streams.

        This config owns trainer RNG policy. It does not seed remote rollout or
        external reward processes. Resume restores checkpoint RNG after this call.
        """
        import os
        import random

        import numpy as np

        require_int(rank, path="training rank", minimum=0)
        if self.deterministic:
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace is None:
                if torch.cuda.is_initialized():
                    raise RuntimeError(
                        "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching deterministic training"
                    )
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            elif workspace not in {":4096:8", ":16:8"}:
                raise ValueError(
                    "deterministic training requires a supported cuBLAS workspace config"
                )
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seed = (self.seed + rank) % 2**64
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)


__all__ = ["OnlineRunConfig"]
