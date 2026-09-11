"""Latent-change-based TeaCache approximation for diffusion rollout.

Accumulate relative-L1 changes between consecutive latent signals. Reuse the
last computed noise prediction while the accumulated change stays below the
configured threshold. Warmup, the final step and a missing cache force a model
forward. This implementation uses latent signals rather than family-specific
timestep-modulated features or rescaling polynomials.

Adapted from the forward-interception approach in vLLM-Omni's
``vllm_omni/diffusion/cache/teacache`` without adopting its scheduler.

Skipped forwards change the rollout prediction relative to exact trainer replay.
The config drift checks require an explicit correction policy when matching
precision labels would otherwise leave that approximation unguarded. Caching is
disabled by default. Skip counters measure reuse; training speed and drift must
be evaluated on the actual workload.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from vrl.utils.config import require_exact_int


def relative_l1_change(cur: torch.Tensor, prev: torch.Tensor) -> float:
    """Relative-L1 change between consecutive denoise signals.

    THE TeaCache skip metric. Offline analysis tools must call this rather than
    reimplement it: a drifting private copy would silently measure something
    other than the signal the runtime actually skips on.

    Reduced to a scalar in fp32. The ``.item()`` requires the host to wait for
    the reduction, so this metric introduces synchronization on CUDA.
    """

    cur = cur.float()
    prev = prev.float()
    denom = prev.abs().sum()
    if float(denom) <= 0.0:
        return float("inf")  # degenerate prev -> never skip
    return float((cur - prev).abs().sum().div(denom).item())


@dataclass(frozen=True, slots=True)
class TeaCacheConfig:
    """Parsed ``sampling.teacache`` block for the diffusion rollout."""

    # Tune the skip budget against measured rollout/replay drift.
    threshold: float = 0.15
    # Prime the cache with real forwards before permitting skips.
    warmup_steps: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or self.threshold <= 0
        ):
            raise ValueError(f"teacache.threshold must be finite and > 0; got {self.threshold!r}")
        require_exact_int(self.warmup_steps, path="teacache.warmup_steps", minimum=0)

    @classmethod
    def from_sampling(cls, value: Any) -> TeaCacheConfig | None:
        """Build a config from a ``sampling.teacache`` value, or ``None`` when off.

        Accepts ``teacache: true`` (defaults), ``teacache: {threshold: .., ...}``,
        or absent/``false`` -> ``None`` (the executor then runs the unchanged
        full-forward path, so the baseline is untouched).
        """

        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError(
                f"sampling.teacache must be a bool or mapping; got {type(value).__name__}",
            )
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("teacache.enabled must be a bool")
        if not enabled:
            return None
        return cls(**{name: item for name, item in value.items() if name != "enabled"})


class TeaCacheState:
    """Per-batch TeaCache decision machine driving the denoise loop.

    Call :meth:`should_run` once per step with the step's input signal; when it
    returns ``False`` reuse :attr:`cached_noise_pred`; when it returns ``True``
    run the real forward and feed the result back via :meth:`cache_noise_pred`.
    """

    def __init__(self, config: TeaCacheConfig, num_steps: int) -> None:
        self._cfg = config
        self._num_steps = int(num_steps)
        self._prev_signal: torch.Tensor | None = None
        self._accumulated_change = 0.0
        self._cached_noise_pred: torch.Tensor | None = None
        self.runs = 0
        self.skips = 0

    def should_run(self, signal: torch.Tensor, step_idx: int) -> bool:
        """Decide whether step ``step_idx`` must run the real transformer forward.

        Accumulates the relative-L1 change of ``signal`` vs the last step; once
        the accumulated change crosses the threshold it forces a real forward and
        resets the accumulator. Warmup steps, the final step, and the first
        forward (no cache yet) always run. Updates ``prev_signal`` every step so
        the accumulated distance tracks the true step-to-step trajectory.
        """

        cfg = self._cfg
        is_last = step_idx >= self._num_steps - 1
        forced = (
            step_idx < cfg.warmup_steps
            or is_last
            or self._cached_noise_pred is None
            or self._prev_signal is None
        )
        if forced:
            run = True
        else:
            self._accumulated_change += relative_l1_change(signal, self._prev_signal)
            run = self._accumulated_change >= cfg.threshold
            if run:
                self._accumulated_change = 0.0
        self._prev_signal = signal.detach()
        if run:
            self.runs += 1
        else:
            self.skips += 1
        return run

    def cache_noise_pred(self, noise_pred: torch.Tensor) -> None:
        """Store the latest real forward output for reuse on skipped steps."""

        self._cached_noise_pred = noise_pred.detach()

    @property
    def cached_noise_pred(self) -> torch.Tensor | None:
        return self._cached_noise_pred

    @property
    def skip_ratio(self) -> float:
        total = self.runs + self.skips
        return self.skips / total if total else 0.0

    def counters(self) -> dict[str, Any]:
        """Engine counters so a profiled run can see how aggressive the cache was."""

        return {
            "teacache_runs": self.runs,
            "teacache_skips": self.skips,
            "teacache_skip_ratio": self.skip_ratio,
            "teacache_threshold": self._cfg.threshold,
        }


__all__ = ["TeaCacheConfig", "TeaCacheState", "relative_l1_change"]
