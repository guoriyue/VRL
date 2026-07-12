"""Runtime-model builders shared by real diffusion perf probes."""

from __future__ import annotations

import time

import torch

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.diffusion.teacache import TeaCacheState, teacache_signal
from vrl.math.diffusion.flow_matching import sde_step_with_logprob

_PROMPT = "a physical scene, high quality"


def build_model(cfg, device, dtype):
    """Build any registered diffusion family's rollout model from its config."""

    from vrl.rollouts.families import (
        get_rollout_family_entry,
        normalize_rollout_family,
    )
    from vrl.utils.config import import_from_path

    cfg.model.use_lora = True
    entry = get_rollout_family_entry(normalize_rollout_family(cfg.model.family))
    resolve_build = import_from_path(entry.model_build_resolver)
    build_bundle = import_from_path(entry.runtime_builder)
    build = resolve_build(cfg, device, parameter_dtype_override=dtype)
    return build_bundle(build).model


def _sampling_request(cfg) -> VideoGenerationRequest:
    sampling = cfg.sampling
    num_frames = int(sampling.get("num_frames", sampling.get("frame_count", 1)))
    return VideoGenerationRequest(
        prompt=_PROMPT,
        negative_prompt=None,
        width=int(sampling.width),
        height=int(sampling.height),
        frame_count=num_frames,
        num_steps=int(sampling.num_steps),
        guidance_scale=float(sampling.guidance_scale),
        seed=0,
        extra={"max_sequence_length": int(sampling.max_sequence_length)},
    )


def _encode_prompt(model, cfg):
    sampling = cfg.sampling
    return model.encode_prompt(
        [_PROMPT],
        None,
        guidance_scale=float(sampling.guidance_scale),
        max_sequence_length=int(sampling.max_sequence_length),
    )


def prepare_sampling_state(model, cfg):
    """Encode the shared prompt and prepare the model's sampling state."""

    return model.prepare_sampling(_sampling_request(cfg), _encode_prompt(model, cfg))


def park_frozen_components(model) -> None:
    """Move frozen encoders / VAE to CPU when the probe only times denoise."""

    move_frozen = getattr(model, "move_frozen_components", None)
    if callable(move_frozen):
        move_frozen(torch.device("cpu"))
        torch.cuda.empty_cache()


def make_step_fn(model, cfg, device, dtype, teacache=None):
    """Return a closure for one denoise step plus the optional TeaCache state."""

    sampling = cfg.sampling
    state = prepare_sampling_state(model, cfg)
    park_frozen_components(model)
    cache_state = (
        TeaCacheState(teacache, int(sampling.num_steps)) if teacache is not None else None
    )

    def one_step(idx: int):
        step_idx = idx % int(sampling.num_steps)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
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


def _e2e_once(model, cfg, dtype):
    """One full image: encode -> prepare -> N denoise steps -> VAE decode."""

    sampling = cfg.sampling
    state = prepare_sampling_state(model, cfg)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
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


def run_e2e(model, cfg, device, dtype, iters=3, warmup=2):
    """Time full end-to-end image latency (encode+denoise+decode)."""

    sampling = cfg.sampling
    for _ in range(warmup):
        _e2e_once(model, cfg, dtype)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.time()
        _e2e_once(model, cfg, dtype)
        torch.cuda.synchronize(device)
        times.append((time.time() - t0) * 1000.0)
    times.sort()
    peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(
        f"\n=== E2E one image (encode+{int(sampling.num_steps)} denoise+decode): "
        f"{times[len(times) // 2]:.0f} ms/img (median of {iters}), peak {peak:.0f} MiB ===",
        flush=True,
    )
