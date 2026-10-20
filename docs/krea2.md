# Krea 2

The `krea2` family trains Krea AI's Krea 2 Raw (released 2026-06-22): a 12B
single-stream MMDiT in the Qwen-Image f8 latent space, conditioned on a stack of
hidden states tapped from twelve layers of a frozen Qwen3-VL-4B. It runs through
the generic full-sequence denoise executor and the shared replay interfaces with
diffusers' own `Krea2Transformer2DModel` and `Krea2Pipeline`.

Use `model/krea2/raw` (`krea/Krea-2-Raw`, pinned revision). The Hub repo is
gated under the Krea 2 Community License: accept it on the model page with the
account whose token the machine holds, or the first load fails with 403. Raw is
the undistilled, CFG-guided checkpoint Krea recommends for LoRA training; the
distilled Turbo checkpoint shares the architecture but is sampled without CFG at
a fixed shift and is not a target here.

## What the family does differently

- Latents are packed `[B, (H/16)*(W/16), 64]` tokens that stay packed through the
  denoise loop; `decode_latents` unpacks them to the video VAE's `[B, 16, 1, H, W]`
  and drops the frame. Height and width must be multiples of 16; the upstream
  pipeline rounds up silently, the family fails loud.
- Text conditioning is `[B, S, 12, 2560]`: the prompt is padded to a fixed
  `max_sequence_length` inside the Qwen-Image chat template and the twelve tapped
  hidden states are stacked per token, with a bool mask. Padded keys are masked
  everywhere, so a shorter block changes nothing for prompts that fit and bounds
  the per-sample trajectory storage (16 MB per sample at 256 tokens in bf16).
- The rotary `position_ids` grid is batch-shared and deterministic from the
  prompt length and the latent grid; replay rebuilds it instead of storing it.
- Guidance: Krea writes CFG as `cond + g * (cond - uncond)`, on when `g > 0`.
  VRL's `sampling.guidance_scale` is the standard scale `1 + g`, the same
  velocity, so Krea's Raw default `g=3.5` is the shared `*_cfg_4_5` preset.
- The scheduler is the dynamic-shift FlowMatch schedule with Krea's
  `max_image_seq_len=6400`; rollout sets it from the packed token count and
  `prepare_replay` rebuilds the same grid from the run's resolution.

## Bundled experiments

- `experiment/krea2/online_grpo_pickscore_smoke_single_gpu`: the smallest
  end-to-end probe (256 px, 4 SDE steps, PickScore reward, rank-16 LoRA) on one
  32 GB card. The 26 GB bf16 DiT with the encoder on the host leaves little room;
  record the peak rather than assume a pass.

## Tests

CPU tests in `tests/models/families/krea2/` run the real tiny diffusers DiT through
`forward_step`, the replay restore and grid rebuild, and the packed prepare /
decode path through a real `Krea2Pipeline` shell over a real tiny Qwen-Image VAE;
the registry matrices, the scheduler log-prob parity fixture and the config sweep
cover the family automatically.
