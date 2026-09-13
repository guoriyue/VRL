# Shared denoise construction and replay residency

## Current change

Cosmos3ReplayModel now uses the common transformer-only replay base. Its dedicated
loader reads transformer weights, scheduler state and the VAE configuration's
`scale_factor_temporal`. It never constructs a VAE, tokenizer or full pipeline.
The pinned Cosmos3-Nano VAE configuration explicitly contains this field.

Replay calls the pinned upstream Cosmos3 segment-building methods with its own
transformer/configuration and uses the upstream velocity mask. These methods
need neither VAE weights nor tokenization: token IDs and latents are already in
the trajectory. No packing equations or token-position tables were copied.

The former `loads_full_generation_modules` assembler parameter and RuntimeBundle
field are removed, along with the boolean-based colocated warning and its
`VRL_STRICT_REPLAY_MEMORY_GUARD` switch. This supersedes the earlier audit fix
that reported Cosmos3's retained full pipeline through that flag. Actual CUDA
ownership checks, parking and worker physical-memory checks remain in place.

## Retain and why

- Rollout and replay have distinct loaders because they need different modules.
  Full rollout loading, VAE decoding and generation offload behavior are unchanged.
- Cosmos3's dedicated loader owns its configuration-only VAE dependency; generic
  replay assembly no longer carries an exception parameter for it.
- Shared replay assembly still applies LoRA/full-finetuning and compile settings.
- Upstream segment builders remain a framework adapter boundary. Their numerical
  implementations are shared, not replaced with a local approximation.
- No new component registry, base-class flag or family-name table was introduced.

## Verification and limits

A real tiny checkpoint test writes only transformer weights, scheduler and VAE
configuration, with no VAE weights, tokenizer or pipeline index. The actual
replay loader succeeds, reproduces rollout velocity and supports backward.
Existing Cosmos3, minimal replay wiring, runtime bundle, Ray configuration and
checkpoint tests pass: 226 passed, 3 skipped. No full pretrained GPU run or
physical-memory benchmark was performed.
