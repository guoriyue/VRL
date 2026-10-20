# MOSS-Image

The `moss_image` family trains OpenMOSS's MOSS-Image (released 2026-10-16): a 5B
single-stream DiT in the FLUX.2 latent space with a frozen Qwen3.5-9B text
encoder, built for legible Chinese and English text rendering. It runs through
the generic full-sequence denoise executor and the shared replay interfaces.

Use `model/moss_image/sft` (`OpenMOSS-Team/MOSS-Image-SFT`, Apache-2.0,
pinned revision). The Base checkpoint shares the architecture; point
`model.path` and `model.revision` at `OpenMOSS-Team/MOSS-Image-Base` to train
it. The preset keeps the 18 GB encoder on the host for a 32 GB card.

## What the family does differently

- The checkpoint ships its DiT, scheduler and pipeline as remote code
  (`pipeline_moss_image.py`), not as diffusers classes. That file is vendored
  verbatim as `vrl/models/families/moss_image/pipeline_moss_image.py`, and the registry
  names its classes by import path (`module:Class`), which the shared component
  loaders resolve. Nothing is fetched at run time beyond the weights.
- Latents are unpacked `[B, 128, H/16, W/16]` tensors (32 VAE channels with the
  2x2 patch folded into channels); height and width must be multiples of 16.
- Prompts are encoded one at a time without padding through the Qwen3.5 chat
  template, then padded to the batch's longest row with a bool mask, as the
  released pipeline does.
- Classifier-free guidance runs the negative branch with DiT block 22 skipped
  (the released `uncond_skip_layers=(22,)`), so the two branches are separate
  forwards combined in fp32. The skip indices travel in the trajectory context
  and replay restores them.
- The scheduler is the released `MOSSImageScheduler`: shift 5 snapped onto the
  1000-point training grid. Fewer than the requested steps would collapse onto
  one grid sigma only at very high step counts; the family fails loud instead of
  recording a shorter trajectory.

## Bundled experiments

- `experiment/moss_image/online_grpo_ocr`: text-to-image GRPO on the OCR reward
  (`reward/ocr`, `dataset/ocr`), 512 px, 10 SDE steps at CFG 3, rank-16 LoRA on
  the attention projections, one 32 GB card. Batch shapes are the sd3_5 OCR
  recipe's; the card's real ceiling for this family is not measured yet.

## Tests

CPU tests in `tests/models/families/moss_image/` run the real tiny vendored DiT
through `forward_step`, the replay restore, a real tiny FLUX.2 VAE decode and
the import-path scheduler load; the registry matrices, the scheduler log-prob
parity fixture and the config sweep cover the family automatically.
