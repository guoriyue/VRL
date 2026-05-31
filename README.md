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
- `configs/dataset/ocr.yaml`: OCR prompt loader, preprocessing, and sampler contract.
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

Runtime environment inputs stay outside experiment YAML. Use them for deployment
facts that change by machine, cluster, shell, or launcher:

| Input | How to set | Purpose |
| --- | --- | --- |
| `RAY_ADDRESS` | Environment variable | Connect VRL to an already-running Ray head for cross-node rollout. |
| `CUDA_VISIBLE_DEVICES` | Environment variable | Limit the trainer process to local CUDA devices before Python starts. |
| `VRL_DATA_ROOT` | Environment variable | Root for artifact-backed datasets stored outside git, defaulting to `data/external`. |
| `HF_HOME` / `HF_HUB_CACHE` | Environment variables | Hugging Face model/cache location shared by model loaders. |
| `RANK` / `WORLD_SIZE` / `LOCAL_RANK` | Launcher environment | Torch distributed rank metadata for distributed training launchers. |
| `data.manifest`, `data.eval_manifest` | YAML or dotlist override | Prompt manifest selection; this is experiment data, not deployment state. |
| `trainer.output_dir` | YAML or dotlist override | Run output location for metrics, checkpoints, evals, and reward artifacts. |

For cross-node Ray rollout, create the Ray cluster outside VRL and pass only the
head address at launch time:

```bash
RAY_ADDRESS=172.31.27.241:6379 CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  /base/distributed=ray_rollout_cross_node \
  distributed.resources.rollout.num_gpus=1 \
  distributed.resources.rollout.num_workers=1
```

VRL does not manage SSH hosts, Ray worker startup, cloud security groups, or
cluster lifecycle. Use manual `ray start`, Ray VM cluster launcher, KubeRay, or a
managed Ray platform for that layer.

## Memory Policy

VRL keeps memory behavior at the layer that owns it. `model.memory` carries two
structural sub-blocks: `vae_decode` (model-build VAE tiling/slicing, applied by
`vrl.models.diffusion.common.vae_decode_memory`) and `frozen_offload` (parking
frozen driver-only modules on CPU before Ray rollout, applied by
`vrl.trainers.frozen_module`). Training memory stays under `actor` or
distributed training config, including gradient checkpointing and FSDP CPU
offload. Rollout actor release stays in the Ray generation runtime, and replay
tensor storage stays in trajectory/replay helpers because it affects correctness
as well as memory.

Family model configs declare their defaults:

```yaml
model:
  memory:
    vae_decode:
      tiling: true
      slicing: true
    frozen_offload:
      enable: true
      modules: [text_encoder, vae]
```

Each sub-block is parsed and validated by its owner; unknown keys fail fast
against the typed dataclass fields. Execution lives with the owners, never in a
shared "memory" module.

## Training Examples

Specific run notes and curated qualitative results live under
`docs/training_examples/`. Use these for concrete examples with visible output;
keep raw checkpoints and full generated artifacts under `outputs/`.

- `docs/training_examples/sd3_5_ocr_grpo/`: SD3.5 OCR GRPO qualitative result.

## How training works (CEA loop)

The online trainer runs a Collector → Evaluator → Algorithm pipeline:
`collect → evaluate → advantage → loss → backward → step`.

1. The generation **Executor** drives a family model to produce images / video / tokens.
2. A **TrajectoryBatch** records the rollout.
3. The **Reward** scores it (see Reward layers below).
4. The **Evaluator** replays the old trajectory through the current model (`replay_forward`).
5. **ReplayResult** holds the current model's raw replay output.
6. The Evaluator combines ReplayResult + old log-probs / mask / ref into a **SegmentSignal**.
7. The **TrajectorySignalBatch** goes to the **Algorithm**.
8. The Algorithm computes the loss.
9. The **Trainer** updates the model; weights then sync back to the rollout worker.

## Reward layers

Reward scoring is a decoupled pipeline so inference can run in-process or on a Ray pool:

- **RewardRollout** (`vrl/rewards/types.py`) — the data being scored.
- **RewardScorer** (`vrl/rollouts/collector/rewards.py`) — collector-side adapter: engine
  output → `RewardRollout` → device tensor.
- **RewardFunction** (`vrl/rewards/base.py`) — the reward objective (name, `score_key`,
  artifact build); aesthetic / CLIP / PickScore / OCR / … subclass it.
- **RewardInferenceRuntime** (`vrl/rewards/inference.py`) — local vs Ray transport that runs
  the scoring **RewardModel**.

## Repository layout

```text
vrl/
  models/      diffusion (sd3_5, wan_2_1, cosmos) + ar (janus_pro, nextstep_1) families
  generation/  pipeline executors + Ray generation runtime
  rollouts/    collector, orchestration (schedule modes), family registry
  rewards/     reward objectives, models, local/ray transport
  algorithms/  GRPO, flow-matching, DPO, DiffusionNFT
  trainers/    online (CEA) + offline trainers, weight sync, checkpointing
  trajectory/  trajectory build / resolve / storage
  config/      OmegaConf loading + Pydantic typed schema (schema.py, validation.py)
  nn/ ray/ math/ utils/    shared kernels, Ray plumbing, helpers
  scripts/     train.py (vrl-train) + data/populate.py (dataset prep)
configs/    layered YAML: base / model / reward / dataset / experiment
datasets/   committed prompt datasets + per-dataset build scripts
docs/       architecture notes + training_examples/
```
