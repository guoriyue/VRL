"""TeaCache for the diffusion rollout denoise loop.

TeaCache (Timestep-Embedding Aware Cache) skips the transformer forward on
denoise steps whose input barely changed since the last real forward, reusing
the previous step's ``noise_pred``. Diffusion denoise is compute-bound (the
transformer forward dominates each step), so skipping a fraction of the forwards
is a direct wall-clock win — typically the single biggest rollout speedup
available for a dense DiT, larger than fp8 (see SPRINT_rollout_vllm_migration.md).

Ported from vLLM-Omni's hook-based TeaCache
(``vllm_omni/diffusion/cache/teacache``) as a *forward-intercepting* technique —
it is an algorithm, not an engine feature, so it drops into our own denoise loop
without adopting vLLM's scheduler. The v1 signal is the relative-L1 change of the
input latents between consecutive steps (model-agnostic, cheap); per-family
"timestep-modulated input" extractors + the rescale polynomial are a follow-up
refinement that improves the skip/accuracy trade-off.

RL caveat — this is NOT pure inference: TeaCache makes the rollout ``noise_pred``
approximate on skipped steps, so the collection-time log-prob diverges from the
trainer's exact replay forward. That is the SAME rollout-vs-replay drift fp8
introduces, and it rides the SAME correction machinery (precision drift guard /
truncated importance sampling). A run that enables TeaCache must therefore arm
the drift guard and treat the skip ratio as a tunable that trades speed for drift.
Default is OFF so the GRPO baseline stays bit-for-bit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

# Moderate default skip budget. The original TeaCache sweeps ~0.1 (conservative,
# few skips) .. ~0.25 (aggressive). 0.15 is a safe starting point; a run tunes it
# against the measured rollout-vs-replay drift.
_DEFAULT_THRESHOLD = 0.15
# First steps always run: they prime the cache and carry the largest step-to-step
# change (early denoise), where skipping costs the most accuracy.
_DEFAULT_WARMUP_STEPS = 2


@dataclass(frozen=True, slots=True)
class TeaCacheConfig:
    """Parsed ``sampling.teacache`` block for the diffusion rollout."""

    threshold: float
    warmup_steps: int

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError(f"teacache.threshold must be > 0; got {self.threshold}")
        if self.warmup_steps < 0:
            raise ValueError(f"teacache.warmup_steps must be >= 0; got {self.warmup_steps}")

    @staticmethod
    def from_sampling(value: Any) -> TeaCacheConfig | None:
        """Build a config from a ``sampling.teacache`` value, or ``None`` when off.

        Accepts ``teacache: true`` (defaults), ``teacache: {threshold: .., ...}``,
        or absent/``false`` -> ``None`` (the executor then runs the unchanged
        full-forward path, so the baseline is untouched).
        """

        if value is None or value is False:
            return None
        if value is True:
            return TeaCacheConfig(
                threshold=_DEFAULT_THRESHOLD,
                warmup_steps=_DEFAULT_WARMUP_STEPS,
            )
        if not isinstance(value, Mapping):
            raise TypeError(
                f"sampling.teacache must be a bool or mapping; got {type(value).__name__}",
            )
        if not bool(value.get("enabled", True)):
            return None
        return TeaCacheConfig(
            threshold=float(value.get("threshold", _DEFAULT_THRESHOLD)),
            warmup_steps=int(value.get("warmup_steps", _DEFAULT_WARMUP_STEPS)),
        )


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
        self._acc = 0.0
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
            self._acc += self._rel_l1(signal, self._prev_signal)
            run = self._acc >= cfg.threshold
            if run:
                self._acc = 0.0
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

    @staticmethod
    def _rel_l1(cur: torch.Tensor, prev: torch.Tensor) -> float:
        # Relative-L1 in fp32, reduced to a scalar. The .item() forces a tiny
        # device sync per step; it is negligible against the transformer forward
        # the skip avoids (a per-family on-device accumulator is a follow-up).
        denom = prev.abs().sum()
        if float(denom) <= 0.0:
            return float("inf")  # degenerate prev -> never skip
        return float((cur - prev).abs().sum().div(denom).item())


__all__ = ["TeaCacheConfig", "TeaCacheState"]
