"""Reward model contract and shared base for in-process torch reward models.

``RewardModel`` is the scoring contract: given one already-materialized
artifact, return named scores. ``TorchRewardModel``
implements it and absorbs the device/dtype/lazy-load boilerplate that every
torch-nn reward used to hand-roll. Subclasses implement ``_load_module``
(build the model once) and ``score_media`` (score one media payload +
prompt). Media is pulled via ``artifact.as_media()``, so the same model
scores an in-memory tensor or a materialized disk artifact (``.pt`` /
``.mp4`` path).

CANONICAL LAZY-MODULE PARKING CONTRACT (this module is the reference):

- ``self._module: Any | None`` starts ``None``. Keeping the unbuilt state
  observable is deliberate: ``functools.cached_property`` would build the
  module on the very access that asks whether it was built, and the parking
  invariant has to be assertable.
- ``_module_for_inference()`` memoizes; it is the hot-path accessor called
  per artifact (twice per artifact in ``target_dino_similarity``).
- ``prepare_for_inference()`` is the public protocol hook. The reward runtime
  probes it by name and calls it inside the CuMem build frame, so a lazy
  module cannot defer its allocations past that frame and then survive every
  park. A reward that omits it loads outside the frame and is caught only by
  ``park_memory``'s residual gate — after a full load and a scored batch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

from vrl.models.dtypes import resolve_torch_dtype
from vrl.rewards.inference import RewardInferenceArtifact


class RewardModel(Protocol):
    """A reward model the runtime loads and runs.

    The runtime is model-agnostic: any model that implements this protocol and
    is named by ``worker_config.model_factory`` can be loaded. Concrete models
    live under ``vrl.rewards.models``.
    """

    def __call__(self, artifact: RewardInferenceArtifact) -> Mapping[str, float]: ...


class LazyTorchModule(ABC):
    """Opt-in lifecycle for modules built inside the inference memory pool."""

    def __init__(self) -> None:
        self._module: Any | None = None

    @abstractmethod
    def _load_module(self) -> Any:
        """Build this reward's complete device state."""

        raise NotImplementedError

    def _module_for_inference(self) -> Any:
        if self._module is None:
            self._module = self._load_module()
        return self._module

    def prepare_for_inference(self) -> None:
        """Materialize lazy model state inside the runtime's owning memory pool."""

        self._module_for_inference()


class TorchRewardModel(LazyTorchModule):
    """Base class for torch reward models loaded in-process."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(worker_config)
        self.worker_config = cfg
        self.device = str(cfg.get("device", "cuda"))
        self.dtype_str = str(cfg.get("dtype", "float32"))

    @property
    def dtype(self) -> Any:
        return resolve_torch_dtype(self.dtype_str)

    @abstractmethod
    def score_media(self, *, media: Any, prompt: str) -> Mapping[str, float]:
        """Return named scores for one artifact's media payload."""

        raise NotImplementedError

    def __call__(self, artifact: RewardInferenceArtifact) -> Mapping[str, float]:
        self.prepare_for_inference()
        return self.score_media(
            media=artifact.as_media(),
            prompt=artifact.prompt,
        )


__all__ = [
    "LazyTorchModule",
    "RewardModel",
    "TorchRewardModel",
]
