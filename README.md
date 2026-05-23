# visual-rl

RL-style post-training infrastructure for visual generative models.

This README promotes only training recipes that have enough real-run validation
to be treated as current canonical entries. The current promoted recipe is:

- `experiment/diffusion/sd3_5/online_grpo_ocr`

Do not add Cosmos-Predict2.5 README recipe entries or gap docs until a real
DiffusionNFT training run proves optimizer updates, non-flat rewards, generated
artifacts, and changed LoRA weights.

## Current Canonical Recipe

Legend:

- ✅ Active: recipe, training entrypoint, runtime path, and structural tests
  exist.
- — Not a target pairing for this canonical recipe.

| Model | Modality | Algorithm | Config | Current progress |
| --- | --- | --- | --- | --- |
| SD3.5 | text-to-image diffusion | GRPO | ✅ `experiment/diffusion/sd3_5/online_grpo_ocr` | Canonical active recipe: OCR GRPO. |

## Algorithm Kinds

| Algorithm kind | Used by | Config base |
| --- | --- | --- |
| `grpo` | SD3.5 OCR | `configs/base/algorithm/grpo.yaml` |

Run the current canonical experiment with:

```bash
python -m vrl.scripts.train --config experiment/diffusion/sd3_5/online_grpo_ocr
```

## SD3.5 OCR GRPO Recipe

`experiment/diffusion/sd3_5/online_grpo_ocr` is the canonical SD3.5 OCR training recipe. It is
configured for the `stabilityai/stable-diffusion-3.5-medium` checkpoint with
LoRA training and a Ray-backed single-GPU rollout worker.

Run the recipe:

```bash
python -m vrl.scripts.train --config experiment/diffusion/sd3_5/online_grpo_ocr
```

The recipe composes these reusable config layers:

- `configs/model/diffusion/sd3_5/medium.yaml`: SD3.5 Medium checkpoint, LoRA target
  modules, and compile settings.
- `configs/sampling/image/512.yaml`: shared 512x512 image sampling shape.
- `configs/sampling/denoise/10_step_cfg_4_5.yaml`: 10 training denoise
  steps, CFG 4.5.
- `configs/reward/ocr.yaml`: OCR reward target and scorer kwargs.
- `configs/dataset/ocr.yaml`: OCR prompt dataset manifest.
- `configs/base/rollout/flow_matching_sde.yaml`: diffusion rollout and SDE
  trajectory settings.
- `configs/base/distributed/ray_rollout_colocated_single_gpu.yaml`: one Ray rollout
  worker on one visible GPU.

Important defaults in `configs/experiment/diffusion/sd3_5/online_grpo_ocr.yaml`:

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
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  trainer.output_dir=outputs/sd3_5_ocr_grpo_run_001
```

Resume from a checkpoint:

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  trainer.resume_from=outputs/sd3_5_ocr_grpo/checkpoint-60
```

Use overrides for one-off reward/model/data changes:

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  reward.components.ocr=0.0 \
  reward.components.aesthetic=1.0 \
  reward.kwargs.aesthetic.model_name=openai/clip-vit-large-patch14 \
  data.manifest=datasets/drawbench/train_192.txt \
  trainer.output_dir=outputs/sd3_5_aesthetic_ablation
```

Ray rollout resource presets use role-level allocation. Multi-GPU split should
declare trainer and rollout budgets, while single-GPU colocated Ray validation must use
the colocated preset so rollout actors are released before replay/backward:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  /base/distributed=ray_rollout \
  distributed.resources.trainer.num_gpus=1 \
  distributed.resources.rollout.num_gpus=auto

CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  /base/distributed=ray_rollout_colocated_single_gpu
```

Manual physical device pinning is an advanced override for debugging or mixed
jobs:

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  distributed.resources.visible_devices='[0,1,2,3]' \
  distributed.resources.trainer.devices='[0]' \
  distributed.resources.rollout.devices='[1,2,3]'
```

## Training Examples

Specific run notes and curated qualitative results live under
`docs/training_examples/`. Use these for concrete examples with visible output;
keep raw checkpoints and full generated artifacts under `outputs/`.

- `docs/training_examples/sd3_5_ocr_grpo/`: SD3.5 OCR GRPO qualitative result.



  1. Executor 调用 family model 的 generation 能力，生成图片/视频/token
  2. TrajectoryBatch 记录 rollout 过程
  3. Reward 给分
  4. Evaluator 调用 family model 的 replay_forward，重看旧 trajectory
  5. ReplayResult 给出当前模型 replay 的 raw output
  6. Evaluator 用 ReplayResult + old_log_prob/mask/ref 得到 SegmentSignal
  7. TrajectorySignalBatch 交给 Algorithm
  8. Algorithm 算 loss
  9. Trainer 更新模型
