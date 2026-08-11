"""Lightweight protocol shared by diffusion engine benchmark CLIs."""

from __future__ import annotations

# Both engines must encode identical text for their results to be comparable. Keep
# this protocol constant free of torch/model imports so the vLLM-Omni environment
# can import it without loading VRL's native runtime stack.
DIFFUSION_BENCHMARK_PROMPT = "a physical scene, high quality"

__all__ = ["DIFFUSION_BENCHMARK_PROMPT"]
