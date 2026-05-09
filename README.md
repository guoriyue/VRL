# visual-rl

RL-style post-training infrastructure for visual generative models.

This matrix tracks repository integration status, not model quality or benchmark
claims. It lists canonical training recipes only; reward swaps, model-size
swaps, and short-lived ablations should be passed as config overrides instead
of becoming new `configs/experiment/*.yaml` files.

## Model / Algorithm Matrix

Legend:

- ✅ Active: recipe, training entrypoint, runtime path, and structural tests
  exist.
- 🟡 Wired: code path exists, but it still needs a real-checkpoint smoke run.
- ⬜ Not wired.
- — Not a target pairing for this model family.

| Model | Modality | GRPO | TokenGRPO | Diffusion DPO | Current progress |
| --- | --- | --- | --- | --- | --- |
| SD3.5 | text-to-image diffusion | ✅ `sd3_5_ocr_grpo` | — | ⬜ | Canonical active recipe: OCR GRPO. |
| Wan 2.1 1.3B | text-to-video diffusion | ✅ `wan_2_1_1_3b_ocr_grpo` | — | ✅ `wan_2_1_1_3b_dpo` | Canonical active recipes: OCR GRPO and offline DPO. |
| Cosmos Predict2 2B | video-to-world diffusion | ✅ `cosmos_predict2_2b_grpo` | — | ⬜ | Canonical Predict2 GRPO wiring. |
| Janus-Pro 1B | autoregressive image | — | ✅ `janus_pro_1b_ocr_grpo` | — | Canonical TokenGRPO recipe: OCR. |
| NextStep-1 1.1 | continuous-token autoregressive image | — | 🟡 `nextstep_1_ocr_grpo` | — | Wired, but still marked pre-smoke for real checkpoint binding. |

## Algorithm Kinds

| Algorithm kind | Used by | Config base |
| --- | --- | --- |
| `grpo` | SD3.5, Wan 2.1, Cosmos Predict2 | `configs/base/algorithm/grpo.yaml` |
| `token_grpo` | Janus-Pro, NextStep-1 | `configs/base/algorithm/token_grpo.yaml` |
| `diffusion_dpo` | Wan 2.1 offline DPO | `configs/base/algorithm/dpo.yaml` |

Run any active experiment with:

```bash
python -m vrl.scripts.train --config experiment/<config_name>
```

Example:

```bash
python -m vrl.scripts.train --config experiment/sd3_5_ocr_grpo
```

## SD3.5 OCR GRPO Recipe

`experiment/sd3_5_ocr_grpo` is the canonical SD3.5 OCR training recipe. It is
configured for the `stabilityai/stable-diffusion-3.5-medium` checkpoint with
LoRA training and a Ray-backed single-GPU rollout worker.

Run the recipe:

```bash
python -m vrl.scripts.train --config experiment/sd3_5_ocr_grpo
```

The recipe composes these reusable config layers:

- `configs/model/sd3_5/sd3_5.yaml`: SD3.5 Medium checkpoint, LoRA target
  modules, and compile settings.
- `configs/sampling/image_512.yaml`: 512x512 image sampling, 10 training
  denoise steps, CFG 4.5.
- `configs/base/rollout/flow_matching_sde.yaml`: diffusion rollout and SDE
  trajectory settings.
- `configs/base/distributed/ray_rollout_single_gpu.yaml`: one Ray rollout
  worker on one visible GPU.

Important defaults in `configs/experiment/sd3_5_ocr_grpo.yaml`:

- OCR-only reward: `reward.components.ocr=1.0`.
- Flow-GRPO parity rollout shape: `rollout.n=8`,
  `rollout.rollout_batch_size=8`, and `rollout.sample_batch_size=8`.
- Flow-GRPO parity optimizer rhythm:
  `actor.gradient_accumulation_steps=4`, which gives two optimizer updates per
  outer rollout epoch with eight prompt groups.
- Fixed evaluation is enabled every 60 epochs on `datasets/ocr/test.txt` with
  `eval.num_steps=40`, `eval.max_prompts=16`, `eval.seed=20260504`, and
  `eval.use_ema=true`.
- Training outputs go to `outputs/sd3_5_ocr_grpo` by default.

Training writes:

- `metrics.csv`: on-policy training rollout metrics.
- `eval_metrics.csv`: fixed OCR eval metrics.
- `eval_epoch_*/contact_sheet.png`: fixed eval contact sheets for visual
  inspection.
- `checkpoint-*` and `checkpoint-final`: resumable trainer checkpoints plus
  exported LoRA artifacts.

Use a fresh output directory for a new run:

```bash
python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  trainer.output_dir=outputs/sd3_5_ocr_grpo_run_001
```

Resume from a checkpoint:

```bash
python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  trainer.resume_from=outputs/sd3_5_ocr_grpo/checkpoint-60
```

Use overrides for one-off reward/model/data changes:

```bash
python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  reward.components.ocr=0.0 \
  reward.components.aesthetic=1.0 \
  reward.kwargs.aesthetic.model_name=openai/clip-vit-large-patch14 \
  data.manifest=datasets/drawbench/train_192.txt \
  trainer.output_dir=outputs/sd3_5_aesthetic_ablation
```
