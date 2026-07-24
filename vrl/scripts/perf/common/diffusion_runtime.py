"""Runtime-model builders shared by real diffusion perf probes."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from vrl.generation.steps.denoise.teacache import TeaCacheState, teacache_signal
from vrl.generation.types import VideoGenerationRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.interfaces import RuntimeBundle
from vrl.utils.config import cfg_path

_PROMPT = "a physical scene, high quality"

if TYPE_CHECKING:
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import RootConfig


def build_runtime(
    root: RootConfig,
    device,
    *,
    precision: PrecisionPolicy,
) -> RuntimeBundle:
    """Build a registered diffusion rollout runtime from its resolved config."""

    from vrl.families.registry import (
        get_model_family_entry,
    )

    if root.model is None:
        raise ValueError("diffusion performance probe requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    build = entry.resolve_model_build(root, device, precision=precision)
    return entry.build_rollout(build)


def prepare_sampling_state(model, cfg):
    """Encode the shared prompt and prepare the model's sampling state."""

    sampling = cfg.sampling
    max_sequence_length = cfg_path(cfg, "sampling.max_sequence_length")
    if max_sequence_length is None:
        max_sequence_length = cfg_path(cfg, "model.executor.max_sequence_length")
    if max_sequence_length is not None:
        max_sequence_length = int(max_sequence_length)
    encode_kwargs = {
        "guidance_scale": float(sampling.guidance_scale),
    }
    request_extra = {}
    if max_sequence_length is not None:
        encode_kwargs["max_sequence_length"] = max_sequence_length
        request_extra["max_sequence_length"] = max_sequence_length
    request = VideoGenerationRequest(
        prompt=_PROMPT,
        negative_prompt=None,
        width=int(sampling.width),
        height=int(sampling.height),
        frame_count=int(sampling.get("num_frames", sampling.get("frame_count", 1))),
        num_steps=int(sampling.num_steps),
        guidance_scale=float(sampling.guidance_scale),
        seed=0,
        extra=request_extra,
    )
    prompt = model.encode_prompt(
        [_PROMPT],
        None,
        **encode_kwargs,
    )
    return model.prepare_sampling(request, prompt)


def make_step_fn(runtime: RuntimeBundle, cfg, teacache=None):
    """Return a closure for one denoise step plus the optional TeaCache state."""

    model = runtime.model
    sampling = cfg.sampling
    state = prepare_sampling_state(model, cfg)
    move_frozen = getattr(model, "move_frozen_components", None)
    if callable(move_frozen):
        move_frozen(torch.device("cpu"))
        torch.cuda.empty_cache()
    cache_state = (
        TeaCacheState(teacache, int(sampling.num_steps)) if teacache is not None else None
    )

    def one_step(idx: int):
        step_idx = idx % int(sampling.num_steps)
        with torch.no_grad():
            if cache_state is not None and not cache_state.should_run(
                teacache_signal(state.latents, teacache.signal),
                step_idx,
            ):
                noise_pred = cache_state.cached_noise_pred
            else:
                noise_pred = model.forward_step(state, step_idx)["noise_pred"]
                if cache_state is not None:
                    cache_state.cache_noise_pred(noise_pred)
            result = sde_step_with_logprob(
                state.scheduler,
                noise_pred.float(),
                state.timesteps[step_idx].unsqueeze(0),
                state.latents.float(),
                generator=None,
                deterministic=True,
                sde_type="cps",
            )
            state.latents = result.prev_sample

    return one_step, cache_state


def _e2e_once(runtime: RuntimeBundle, cfg):
    """One full image: encode -> prepare -> N denoise steps -> VAE decode."""

    model = runtime.model
    sampling = cfg.sampling
    state = prepare_sampling_state(model, cfg)
    with torch.no_grad():
        for step_idx in range(int(sampling.num_steps)):
            noise_pred = model.forward_step(state, step_idx)["noise_pred"]
            result = sde_step_with_logprob(
                state.scheduler,
                noise_pred.float(),
                state.timesteps[step_idx].unsqueeze(0),
                state.latents.float(),
                generator=None,
                deterministic=True,
                sde_type="cps",
            )
            state.latents = result.prev_sample
        return model.decode_latents(state.latents)


def run_e2e(runtime: RuntimeBundle, cfg, device):
    """Time full end-to-end image latency (encode+denoise+decode)."""

    iters = 3
    warmup = 2
    sampling = cfg.sampling
    for _ in range(warmup):
        _e2e_once(runtime, cfg)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.time()
        _e2e_once(runtime, cfg)
        torch.cuda.synchronize(device)
        times.append((time.time() - t0) * 1000.0)
    times.sort()
    peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(
        f"\n=== E2E one image (encode+{int(sampling.num_steps)} denoise+decode): "
        f"{times[len(times) // 2]:.0f} ms/img (median of {iters}), peak {peak:.0f} MiB ===",
        flush=True,
    )
