# SD3.5 OCR GRPO Training Example

This note records one qualitative SD3.5 OCR GRPO result. It is a concrete
training example, not a benchmark claim or paper reproduction.

## Training Recipe

Base command for this example:

```bash
python -m vrl.scripts.train \
  --config experiment/sd3_5/online_grpo_ocr \
  trainer.total_epochs=200 \
  trainer.output_dir=outputs/sd3_5_ocr_grpo_200ep_20260507_0011
```

Core recipe:

- Model: `stabilityai/stable-diffusion-3.5-medium`
- Trainable method: LoRA on SD3.5 transformer attention projections
- Algorithm: GRPO with flow-matching SDE replay
- Reward: OCR only, via `reward.components.ocr=1.0`
- Training data: `datasets/ocr/train.txt`
- Eval data: `datasets/ocr/test.txt`
- Sampling: 512x512, 10 train denoise steps, CFG 4.5
- Eval sampling: 40 denoise steps, EMA weights enabled
- Rollout shape: 8 samples per prompt, 8 prompt groups per outer epoch
- Runtime: single-GPU Ray rollout local debug preset

## Result

![Image #1: SD3.5 OCR GRPO qualitative comparison](qualitative_ocr_comparison.jpg)

**[Image #1]** Qualitative OCR comparison for SD3.5 GRPO. The right-side
samples are visibly better on several text-heavy prompts, including `Lost City
Near`, `Fearless`, and `Gas Next Exit 2 Miles`. The run still has clear failure
modes: `Upgrades Available`, `Private Property No Entry`, and `Trespassers Will
Be Jousted` remain misspelled or partially corrupted.

## Interpretation

The useful signal here is qualitative: the model produces visibly more legible
text for some selected prompt families. This page is intentionally not a metric
report; use the run's `outputs/` directory when exact logs, checkpoints, or
aggregate eval numbers matter.

Do not present this as a general OCR benchmark win without a locked eval
protocol and the matching metric report.

## Reuse Checklist

For another training example, keep the same shape:

- Put the run note in `docs/training_examples/<experiment_name>/README.md`.
- Put one or two curated images next to the README with descriptive names.
- Include the exact training command or the run-specific overrides.
- Keep detailed logs, checkpoints, and metrics in `outputs/`, not in this note.
- State clearly whether the result is qualitative, metric-backed, or both.
