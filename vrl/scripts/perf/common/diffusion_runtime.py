"""Runtime-model builders shared by real diffusion perf probes."""

from __future__ import annotations

import time

import torch

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.diffusion.teacache import TeaCacheState, teacache_signal
from vrl.math.diffusion.flow_matching import sde_step_with_logprob
from vrl.models.interfaces import RuntimeBundle

_PROMPT = "a physical scene, high quality"


def build_runtime(cfg, device) -> RuntimeBundle:
    """Build a registered diffusion rollout runtime from its resolved config."""

    from vrl.families.registry import (
        get_model_family_entry,
    )

    cfg.model.use_lora = True
    entry = get_model_family_entry(str(cfg.model.family))
    build = entry.resolve_model_build(cfg, device)
    return entry.build_rollout(build)


def build_model(cfg, device, dtype):
    """Compatibility facade for the recorded TeaCache drift probe.

    That one-shot probe owns its historical BF16 context locally. Refuse a
    config/CLI mismatch instead of silently letting it diverge from the
    resolved rollout role.
    """

    from vrl.models.dtypes import dtype_to_precision_token

    runtime = build_runtime(cfg, device)
    token = dtype_to_precision_token(dtype)
    if runtime.precision.dtype != token:
        raise ValueError(
            "TeaCache probe dtype does not match resolved rollout precision: "
            f"requested {token!r}, resolved dtype={runtime.precision.dtype!r}, "
            f"outer_autocast={runtime.outer_autocast!r}",
        )
    return runtime.model


def prepare_sampling_state(model, cfg):
    """Encode the shared prompt and prepare the model's sampling state."""

    sampling = cfg.sampling
    request = VideoGenerationRequest(
        prompt=_PROMPT,
        negative_prompt=None,
        width=int(sampling.width),
        height=int(sampling.height),
        frame_count=int(sampling.get("num_frames", sampling.get("frame_count", 1))),
        num_steps=int(sampling.num_steps),
        guidance_scale=float(sampling.guidance_scale),
        seed=0,
        extra={"max_sequence_length": int(sampling.max_sequence_length)},
    )
    prompt = model.encode_prompt(
        [_PROMPT],
        None,
        guidance_scale=float(sampling.guidance_scale),
        max_sequence_length=int(sampling.max_sequence_length),
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


def run_e2e(runtime: RuntimeBundle, cfg, device, iters=3, warmup=2):
    """Time full end-to-end image latency (encode+denoise+decode)."""

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
