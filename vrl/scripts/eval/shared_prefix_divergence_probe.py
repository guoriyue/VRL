"""Does sharing the early denoise prefix collapse within-group sample diversity?

WHY (signal_paged P0 + lossless-research open question #1): shared-prefix / tree
rollout is the report's ONLY truly-untested LOSSLESS group-level compute lever for
diffusion RL — a GRPO group (same prompt, G samples) shares denoise steps 0..k once,
then branches with independent SDE noise for k..T, cutting forwards from G*T to
k + G*(T-k). It is lossless to the policy gradient ONLY if it does not destroy the
within-group reward variance GRPO learns from. Diffusion sets global structure early
and detail late, so sharing too many early steps may lock the layout and flatten
diversity. This probe measures the trade DIRECTLY, with the repo's real SDE step:

  retention(k) = pairwise-final-latent-distance(shared prefix k) / (fully independent)
  forward_savings(k) = 1 - (k + G*(T-k)) / (G*T)

A useful k needs BOTH high forward savings AND high diversity retention. The
prefix_steps -> retention curve is exactly what signal_paged P0 asks for (minus the
reward model; latent diversity is the variance proxy).

Faithful: uses vrl.math.denoise.flow_matching.sde_step_with_logprob (the same SDE
step the rollout/replay path uses), not a toy noise injection.

Run: python -m vrl.scripts.eval.shared_prefix_divergence_probe --group 6 --steps 28 --side 768
"""

from __future__ import annotations

import argparse

import torch

from vrl.math.denoise.flow_matching import sde_step_with_logprob


def _pairwise_mean_l2(x: torch.Tensor) -> float:
    """Mean L2 distance over all unordered sample pairs of [G, ...] (fp32)."""
    g = x.shape[0]
    flat = x.reshape(g, -1).float()
    d = torch.cdist(flat, flat)  # [G, G]
    iu = torch.triu_indices(g, g, offset=1)
    return float(d[iu[0], iu[1]].mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--group", type=int, default=6, help="GRPO group size G")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--side", type=int, default=768)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--noise-level", type=float, default=1.0, help="SDE stochasticity (Flow-GRPO)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(
        "cuda"
    )
    pipe.set_progress_bar_config(disable=True)
    tf = pipe.transformer
    device, dtype = tf.device, tf.dtype
    sched = pipe.scheduler
    G, T = args.group, args.steps

    embeds, neg, pooled, neg_pooled = pipe.encode_prompt(
        prompt="a photorealistic astronaut riding a horse on mars",
        prompt_2=None,
        prompt_3=None,
        do_classifier_free_guidance=True,
        device=device,
    )
    h, w = args.side // 8, args.side // 8
    ch = tf.config.in_channels

    def _embeds(n: int):
        e = torch.cat([neg, embeds]).repeat_interleave(n, dim=0)
        pl = torch.cat([neg_pooled, pooled]).repeat_interleave(n, dim=0)
        return e, pl

    def _vel(lat: torch.Tensor, t, emb, pl) -> torch.Tensor:
        """CFG velocity (transformer output) for a [n, ...] latent batch."""
        n = lat.shape[0]
        model_in = torch.cat([lat] * 2)
        out = tf(
            hidden_states=model_in,
            timestep=t.expand(2 * n),
            encoder_hidden_states=emb,
            pooled_projections=pl,
            return_dict=False,
        )[0]
        uncond, cond = out.chunk(2)
        return uncond + args.guidance * (cond - uncond)

    @torch.no_grad()
    def _denoise(lat: torch.Tensor, start: int, gen: torch.Generator) -> torch.Tensor:
        """Run SDE denoise steps [start, T) on a [n, ...] latent with one generator."""
        n = lat.shape[0]
        emb, pl = _embeds(n)
        for i in range(start, T):
            t = sched.timesteps[i]
            vel = _vel(lat, t, emb, pl)
            res = sde_step_with_logprob(
                sched,
                vel,
                t.expand(n),
                lat,
                generator=gen,
                noise_level=args.noise_level,
                step_index=i,
            )
            lat = res.prev_sample.to(dtype)
        return lat

    def _init_latents(n: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(n, ch, h, w, device=device, dtype=dtype, generator=g)

    print(
        f"model SD3.5-medium  G={G}  T={T}  side={args.side}^2  guidance={args.guidance}  "
        f"noise_level={args.noise_level}"
    )

    # Baseline: G fully-independent trajectories (own initial noise, own SDE noise).
    sched.set_timesteps(T, device=device)
    base_lat = _init_latents(G, args.seed)
    gen_base = torch.Generator(device=device).manual_seed(args.seed + 100)
    base_final = _denoise(base_lat, 0, gen_base)
    d0 = _pairwise_mean_l2(base_final)
    print(f"\nbaseline (independent) pairwise final-latent L2 = {d0:.3f}")
    print(f"\n{'k':>3} | {'fwd_saved':>9} | {'pairwise L2':>11} | {'retention':>9} | verdict")

    ks = sorted(set([0, T // 7, 2 * T // 7, T // 2, 5 * T // 7, 6 * T // 7, T]))
    for k in ks:
        sched.set_timesteps(T, device=device)
        if k == 0:
            # No shared prefix == fully independent (retention sanity check ~1.0).
            final = _denoise(
                _init_latents(G, args.seed + 1),
                0,
                torch.Generator(device=device).manual_seed(args.seed + 101),
            )
        else:
            # Shared prefix: ONE trajectory for steps 0..k-1, then broadcast and branch.
            lat_k = _run_prefix(
                sched, _init_latents(1, args.seed + 1), k, args, _embeds, _vel, dtype, device
            )
            branch = lat_k.repeat(G, *([1] * (lat_k.ndim - 1)))
            if k >= T:
                final = branch  # fully shared -> all identical (retention ~0 sanity check)
            else:
                gen_br = torch.Generator(device=device).manual_seed(args.seed + 202)
                final = _denoise(branch, k, gen_br)
        d = _pairwise_mean_l2(final)
        ret = d / d0 if d0 > 0 else 0.0
        saved = 1.0 - (k + G * (T - k)) / (G * T)
        verdict = "DEAD (diversity gone)" if ret < 0.3 else "viable" if ret >= 0.7 else "marginal"
        print(f"{k:>3} | {saved * 100:>8.1f}% | {d:>11.3f} | {ret * 100:>8.0f}% | {verdict}")

    print("\nread: want a k with BOTH high fwd_saved AND retention>=70%. If retention drops")
    print("      below ~30% while fwd_saved is still small, the early steps lock global")
    print(
        "      structure -> shared-prefix collapses GRPO reward variance -> lossless lever is dead."
    )


@torch.no_grad()
def _run_prefix(sched, lat, k, args, _embeds, _vel, dtype, device):
    """SDE steps [0, k) on a single [1, ...] latent (the shared prefix realization)."""
    gen = torch.Generator(device=device).manual_seed(args.seed + 101)
    emb, pl = _embeds(1)
    for i in range(0, k):
        t = sched.timesteps[i]
        vel = _vel(lat, t, emb, pl)
        res = sde_step_with_logprob(
            sched,
            vel,
            t.expand(1),
            lat,
            generator=gen,
            noise_level=args.noise_level,
            step_index=i,
        )
        lat = res.prev_sample.to(dtype)
    return lat


if __name__ == "__main__":
    main()
