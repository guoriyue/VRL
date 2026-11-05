# MiniMax-H3 (Hailuo 3.0)

The `minimax_h3` family trains MiniMax's Hailuo 3.0: a 33B
flow-matching transformer that denoises a video latent and its soundtrack's
latent jointly in one packed sequence, conditioned by a frozen Qwen3-VL-32B.
It runs through the generic full-sequence denoise executor and the shared
replay interfaces with the `MiniMaxH3Transformer3DModel`, the two H3 VAEs and the
`MiniMaxH3ModularPipeline` blocks of diffusers >= 0.40. Task token `t2v`: a
prompt in, a 24 fps clip out; the soundtrack is generated but not decoded.

Use `model/minimax_h3/h3` (`MiniMaxAI/MiniMax-H3`, pinned revision; the repo is
not gated). The weights are under the MiniMax-H3 Community License, which carries
geographic and use restrictions; check them before deploying. The repo holds
three 33B partitions, and this family loads exactly one of them: `transformer`
(the `t2va` / `fl2va` checkpoint). `transformer_ref` (the `ref2va` checkpoint)
and the repo's `FL2VA/` and `Ref2VA/` trees (144 GB each) stay on disk; a
download that skips them is 144 GB instead of 500.

## What the family does differently

- One packed sequence `[text | audio rows | video rows]`, strictly batch=1 per
  prompt: the row layout is a function of the prompt length and the canvas, so the
  executor pins `samples_per_generation_batch=1` and replay micro-batches must
  share one prompt length. The layout and the per-row timestep plan come from the
  diffusers block statics, not a re-derivation.
- The video latent is the RL action; the audio rows are a deterministic side
  stream stepped by the checkpoint's own Euler update one step behind the loop.
  Every step's audio input is recorded for replay, so audio never enters the
  policy ratio. A loop mode that skips a forward is refused.
- Velocity sign: H3 predicts a data-ward velocity on a `t = 1 - sigma` clock.
  `forward_step` returns `-v` as the shared `noise_pred`, and
  `MiniMaxH3FlowScheduler` flips it back inside `step`, so `denoise_mode: native`
  reproduces the reference sampler.
- Guidance-distilled: `sampling.guidance_scale` must be 1.0, one forward per step.
- Geometry is refused, not rounded: frame counts off the `17n + 5` grid or
  canvases off the 32 multiple raise. 124 frames is the shortest count at or
  above 5 s; the native canvas is 1344x768 at 24 fps.
- Two frozen fp32 fidelity boundaries, the video VAE (10.4 GB) and the audio
  VAE; the conditioner follows `rollout.prompt_encoder_dtype`.

## Conditioning workflows of the checkpoint

What the diffusers pipeline offers, against what the family wraps. This is the
reading an agentic "edit" loop needs before it decides what an H3 environment's
state is.

| Workflow | Input | Partition | How it conditions | Family |
| --- | --- | --- | --- | --- |
| `t2va` | prompt | `transformer` | text rows only | wrapped |
| `fl2va` | prompt + `image` and/or `last_image` | `transformer` | one keyframe block per image, anchored at the first or last latent frame on the rotary clock, resized onto the target canvas, encoded by the video VAE, noised to `t = 0.999` and held there; the prompt gets a `<Picture i>: ` label plus a vision block per keyframe | not wrapped (layout built with no anchors) |
| `ref2va` | prompt + ordered `references` | `transformer_ref` | up to 9 image references (subject, style or scene; encoded at a 2048 short edge of their own, they never bind the generated geometry), up to 3 video references (motion and camera, with their own soundtrack, resampled onto 24 fps), up to 3 audio references (voice or music, never alone); one packed block per reference ahead of the generated rows, labelled `<Picture i>` / `<Video k>` / `<Audio j>` in packed order | not wrapped (needs the other 33B partition) |

Neither conditional workflow is a pixel-preserving edit of an input clip: `fl2va`
continues from a frame (the frame is kept, everything after it is generated) and
`ref2va` generates a new clip that borrows subject, motion or sound from its
references. An environment that wants "keep this region of the video" has to
express it through a keyframe or a reference, and measure the preservation
itself.

## One 32 GB card

The checkpoint is ~144 GB bf16 and a 5090 holds 32 GB next to 91 GB of host
RAM, so the single-card recipe
`experiment/minimax_h3/online_grpo_kling_video_reward_single_gpu` sets the
residency explicitly:

- `model.offload` / `actor.offload` stream the transformer's frozen blocks (the
  50 main blocks and the 2 token-refiner blocks) from the host per call, with
  `pin_host: false` so the loader's file-backed weights stay evictable instead of
  becoming 66 GB of locked RAM.
- `model.memory.cpu_resident: [text_encoder]` keeps the Qwen3-VL-32B on the host
  and encodes there, one 32B CPU forward per prompt.
- 640x384 instead of 768p, about 9.7k packed rows at 124 frames.

The two file-backed models together exceed the host RAM, so every run goes
under `systemd-run --user --scope -p MemoryMax=80G`: the page cache then evicts
instead of the OOM killer rebooting the box (which it did once, 2026-10-14).

## Integration check

Pending. The full-sequence probe below is the check; the family stays
"Integrated" in the README until it passes on the released weights. The
`--no-module-hooks` flag is required: the per-module row-0 recorder would hold
~210 MB per hooked module of a 9.7k x 5376 sequence, several hundred of them.

```bash
CUDA_VISIBLE_DEVICES=0 systemd-run --user --scope -p MemoryMax=80G \
  python -m vrl.scripts.generation.full_sequence_probe \
  --config experiment/minimax_h3/online_grpo_kling_video_reward_single_gpu \
  sampling.num_steps=6 --no-module-hooks \
  --prompt "a red fox runs across fresh snow toward the camera, then stops and looks back" \
  --artifact-dir outputs/probes/minimax_h3
```

Expected cost at 640x384, from the measurements of the local inference script
(`~/h3_local_run/README.md`) and the DPO probe (`docs/sprints/planned/SPRINT_h3_first_frame_physics_dpo.md`):
about 7 s of compute per step plus the 66 GB host-to-device copy at pageable
bandwidth, for the rollout and again for the replay, plus the CPU prompt
encode and the fp32 VAE decode.

## Bundled experiments

- `experiment/minimax_h3/online_grpo_kling_video_reward`: the deployment recipe,
  768p, resident policy; needs a multi-GPU FSDP deployment that holds the 33B
  policy and the 32B conditioner.
- `experiment/minimax_h3/online_grpo_kling_video_reward_single_gpu`: the same
  recipe at 640x384 with the single-card residency above. The probe target; no
  training step has run on it.

## Tests

CPU tests in `tests/models/families/minimax_h3/` run the tiny real transformer,
both tiny VAEs and a tiny Qwen3-VL stack through the family: the reference
layout and row-timestep plan, the negated velocity, the audio side stream, the
bit-exact replay restore, the decode normalizations, the host-streamed forward
against the resident one, the two loaders (`t2va` workflow only; both VAEs fp32;
`cpu_resident` placement) and the block-partitioned multi-GPU placement. The
registry matrices and the config sweep cover the family automatically.
