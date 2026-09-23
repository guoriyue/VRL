# Qwen Image 2.1

The `qwen_image_21` family supports text generation, ordered reference-image
editing, and RGB/RGBA output through the existing denoise executor and replay
interfaces. It is separate from the older `qwen_image` family.

Use `model/qwen_image_21/base`. The frozen dependency lock includes the upstream
Qwen 2.1 implementation; sync with the cosmos extra before running it. The base
preset keeps the frozen text encoder on CPU for a 32 GB GPU. Sampling uses
`reference_resolution` (default 1024) and `output_mode` (`rgb` or `rgba`).

A prompt manifest may supply either `reference_image` or an ordered
`reference_images` list of up to ten paths. Paths resolve against the existing
artifact data root. Reference latents are fixed conditioning: replay stores them,
but only generated target latents enter the SDE transition likelihood.

PNG output preserves alpha. Ordinary RGB reward models view RGBA composited over
white; transparent layers need their own alpha-aware checks. A successful RGBA
output alone does not establish correct layer decomposition.

## Local integration probe

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m vrl.scripts.generation.qwen_image_21_edit_probe \
  --source /path/to/bookshop.png --reference /path/to/leaf.png \
  --out outputs/qwen_image_21/probe \
  --resolution 512 --reference-resolution 512 --steps 20 \
  --compare-reference --check-lora-backward
```

The fixed probe prompts expect a bookshop sign and a leaf. Native Euler comparison
shares the initial latent and VAE preprocessing precision with the upstream
pipeline. The separate SDE probe checks likelihood replay, finite gradients, and
one LoRA optimizer update. It does not train on task reward or demonstrate an RL
improvement. Full online training and agentic control require their own validation.

See `docs/reports/visual_rl_engine_20260922/PROGRESS.md` for actual measurements.
