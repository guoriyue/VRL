<div align="center">

# visual-rl

**Easy, honest, single-process-friendly RL post-training for visual generative models.**

Diffusion (image / video) and autoregressive image generation — one layered-config
trainer, one Collector → Evaluator → Algorithm loop, real-run validation before any
recipe is promoted.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Built on](https://img.shields.io/badge/built%20on-PyTorch%20%2B%20Ray-orange)
![RL](https://img.shields.io/badge/RL-GRPO%20%2F%20DiffusionNFT%20%2F%20DPO-purple)

</div>

---

`visual-rl` is an RL training framework focused on **visual generative models**: text-to-image
and text/image-to-video diffusion, plus autoregressive image generation. It runs the same
online RL loop across model families through a layered YAML config system, a Ray-backed
rollout worker, and a decoupled reward-scoring pipeline — and it is built to run and be
debugged on a single GPU before it ever needs a cluster.

## Why visual-rl

- **One loop, many families.** A single Collector → Evaluator → Algorithm (CEA) trainer drives
  SD3.5, FLUX, Qwen-Image, Wan, Cosmos, Janus-Pro, and NextStep-1 from layered config — no
  per-model training script.
- **Single-GPU first.** Colocated Ray rollout releases actors before replay/backward, so a full
  GRPO recipe fits and is debuggable on one card; the same config scales to multi-GPU and
  cross-node Ray by swapping one `distributed` layer.
- **Decoupled reward serving.** Reward objectives (OCR, aesthetic, CLIP, PickScore, Kling
  VideoReward, physics) run in-process or on a Ray pool behind one interface.
- **Memory at its owner.** VAE tiling, frozen-module CPU offload, gradient checkpointing, and
  FSDP offload each live with the layer that owns them — no shared "memory" grab-bag.
- **Honest status.** A recipe is promoted to *canonical* only after a real run proves optimizer
  updates, non-flat rewards, generated artifacts, and changed weights. Everything else is
  labeled runnable-but-unvalidated, never advertised as working.

## Status & promotion policy

This README distinguishes what is **validated** from what is merely **wired up**. Recipes carry
config, a training entrypoint, a runtime path, and structural tests long before they have a
trustworthy learning curve — within-noise reward drift is not "learning."

| Status | Meaning |
| --- | --- |
| ✅ **Validated** | A real run proved optimizer updates, non-flat reward, artifacts, and changed weights. Promoted as canonical. |
| 🧪 **Runnable** | Config + entrypoint + runtime + structural tests exist; not yet validated by a trustworthy run. |
| 🚧 **Planned** | Targeted, not yet wired end-to-end. |

> Do not promote a 🧪 recipe (e.g. Cosmos-Predict2.5 DiffusionNFT) to ✅ or add gap docs for it
> until a real training run clears the bar above.

## Supported models

| Family | Modality | Algorithms (configs present) | Status |
| --- | --- | --- | --- |
| **SD3.5** | text → image diffusion | GRPO (OCR, GenEval, PickScore) | ✅ Validated: OCR GRPO |
| **FLUX** | text → image diffusion | GRPO-Guard, DanceGRPO, DiffusionNFT, Flow-DPPO | 🧪 Runnable |
| **Qwen-Image** | text → image diffusion | GRPO | 🧪 Runnable |
| **Wan2.1** | text/image → video diffusion | GRPO (physics, OCR, Kling), DPO | 🧪 Runnable |
| **Wan2.2** | image → video diffusion | GRPO (physics) | 🧪 Runnable |
| **Cosmos-Predict2** | video diffusion | GRPO (Kling VideoReward, V2W) | 🧪 Runnable |
| **Cosmos-Predict2.5** | video diffusion | DiffusionNFT (Kling, motion-physics) | 🧪 Runnable |
| **Cosmos-Anima** | video diffusion | GRPO (aesthetic, NSFW-safety) | 🧪 Runnable |
| **Janus-Pro** | autoregressive image | GRPO, R1-GRPO (OCR, aesthetic) | 🧪 Runnable |
| **NextStep-1** | autoregressive image | GRPO (OCR) | 🧪 Runnable |

## Supported algorithms

| Algorithm | Config base | Notes |
| --- | --- | --- |
| GRPO | `configs/base/algorithm/grpo.yaml` | Group-relative policy optimization (canonical). |
| GRPO-Guard | `configs/base/algorithm/grpo_guard.yaml` | Guarded GRPO variant. |
| DanceGRPO | `configs/base/algorithm/dance_grpo.yaml` | DanceGRPO-style objective. |
| DiffusionNFT | `configs/base/algorithm/diffusion_nft.yaml` | Likelihood-free diffusion RL. |
| Flow-DPPO | `configs/base/algorithm/flow_dppo.yaml` | Flow-matching DPPO. |
| Token-GRPO | `configs/base/algorithm/token_grpo{,_multisegment}.yaml` | Token-level GRPO for AR families. |
| DPO | `configs/base/algorithm/dpo.yaml` | Offline preference optimization. |

## Architecture — the CEA loop

The online trainer runs a **Collector → Evaluator → Algorithm** pipeline:
`collect → evaluate → advantage → loss → backward → step`.

1. The generation **Executor** drives a family model to produce images / video / tokens.
2. A **TrajectoryBatch** records the rollout.
3. The **Reward** scores it (see [Reward layers](#reward-layers)).
4. The **Evaluator** replays the old trajectory through the current model (`replay_forward`).
5. **ReplayResult** holds the current model's raw replay output.
6. The Evaluator combines ReplayResult + old log-probs / mask / ref into a **SegmentSignal**.
7. The **TrajectorySignalBatch** goes to the **Algorithm**.
8. The Algorithm computes the loss.
9. The **Trainer** updates the model; weights then sync back to the rollout worker.

## Quickstart

`pyproject.toml` is the single source of truth for dependencies. Install the base plus the
extras for the backends you will run (they combine):

| Extra | Adds | Needed for |
| --- | --- | --- |
| *(base)* | torch, ray, safetensors, pillow, imageio | every training run |
| `cosmos` | diffusers, transformers, accelerate, peft, torchvision | SD3.5 / FLUX / Qwen-Image / Wan / Cosmos diffusion |
| `reward` | qwen-vl-utils, peft, torchvision, transformers | Kling VideoReward / Qwen2-VL video scoring |
| `data` | datasets, pyarrow, requests, av | dataset prep under `vrl/scripts/data` |
| `ocr` | paddleocr, paddlepaddle, rapidocr, python-Levenshtein | OCR reward (canonical SD3.5 recipe) |
| `pose` / `pose-gpu` | onnxruntime(-gpu), opencv, huggingface-hub | pose-based rewards (CPU / GPU) |
| `ar-vllm` | vllm, pinned torch | vLLM-Omni rollout for AR families |
| `dev` | pytest, ruff, … | running the test suite |

```bash
# uv (recommended — fast, reproducible via uv.lock)
uv sync --extra cosmos --extra ocr --extra dev    # canonical SD3.5 OCR recipe

# or classic pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cosmos,ocr,dev]"
```

VideoCon-Physics scoring additionally needs its vendored model sources (a git submodule, not a
pip package): `git submodule update --init --recursive`.

Then run the canonical recipe:

```bash
python -m vrl.scripts.train --config experiment/diffusion/sd3_5/online_grpo_ocr
```

(`vrl-train` is the installed console-script alias for `python -m vrl.scripts.train`.)

## Canonical recipe: SD3.5 OCR GRPO

`experiment/diffusion/sd3_5/online_grpo_ocr` is the **only ✅ validated** recipe. It targets the
`stabilityai/stable-diffusion-3.5-medium` checkpoint with LoRA training and a Ray-backed
single-GPU rollout worker.

It composes these reusable config layers:

- `configs/model/diffusion/sd3_5/medium.yaml` — checkpoint, LoRA target modules, compile settings.
- `configs/sampling/image/512.yaml` — shared 512×512 sampling shape.
- `configs/sampling/denoise/10_step_cfg_4_5.yaml` — 10 training denoise steps, CFG 4.5.
- `configs/reward/ocr.yaml` — OCR reward target and scorer kwargs.
- `configs/dataset/ocr.yaml` — OCR prompt loader, preprocessing, sampler contract.
- `configs/base/rollout/flow_matching_sde.yaml` — diffusion rollout + SDE trajectory settings.
- `configs/base/distributed/ray_rollout_colocated_single_gpu.yaml` — one Ray rollout worker on one GPU.

Important defaults (`configs/experiment/diffusion/sd3_5/online_grpo_ocr.yaml`):

- OCR-only reward: `reward.components.ocr=1.0`.
- Scale-bump rollout shape: `rollout.n=16`, `rollout.rollout_batch_size=8`, `rollout.sample_batch_size=16`.
- Flow-GRPO parity rhythm: `actor.gradient_accumulation_steps=4` → two optimizer updates per outer epoch.
- Fixed eval every 60 epochs on `datasets/ocr/test.txt` (`eval.num_steps=40`, `eval.max_prompts=16`, `eval.seed=20260504`, `eval.use_ema=true`).
- Outputs go to `outputs/sd3_5_ocr_grpo`.

Training writes `metrics.csv`, `eval_metrics.csv`, `eval_epoch_*/contact_sheet.png`, and
`checkpoint-*` / `checkpoint-final` (resumable trainer state + exported LoRA artifacts).

```bash
# fresh run
python -m vrl.scripts.train --config experiment/diffusion/sd3_5/online_grpo_ocr \
  trainer.output_dir=outputs/sd3_5_ocr_grpo_run_001

# resume
python -m vrl.scripts.train --config experiment/diffusion/sd3_5/online_grpo_ocr \
  trainer.resume_from=outputs/sd3_5_ocr_grpo/checkpoint-60

# one-off reward/model/data overrides
python -m vrl.scripts.train --config experiment/diffusion/sd3_5/online_grpo_ocr \
  reward.components.ocr=0.0 reward.components.aesthetic=1.0 \
  reward.kwargs.aesthetic.model_name=openai/clip-vit-large-patch14 \
  data.manifest=datasets/drawbench/train_192.txt \
  trainer.output_dir=outputs/sd3_5_aesthetic_ablation
```

### Scaling out

Ray rollout presets use role-level allocation. Multi-GPU split declares trainer and rollout
budgets; single-GPU colocated validation must use the colocated preset so rollout actors are
released before replay/backward:

```bash
# multi-GPU split
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  /base/distributed=ray_rollout \
  distributed.resources.trainer.num_gpus=1 \
  distributed.resources.rollout.num_gpus=auto

# single-GPU colocated
CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  /base/distributed=ray_rollout_colocated_single_gpu

# cross-node: create the Ray cluster outside VRL, pass only the head address
RAY_ADDRESS=172.31.27.241:6379 CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  /base/distributed=ray_rollout_cross_node \
  distributed.resources.rollout.num_gpus=1 \
  distributed.resources.rollout.num_workers=1
```

Manual physical device pinning is an advanced override for debugging or mixed jobs:

```bash
python -m vrl.scripts.train --config experiment/diffusion/sd3_5/online_grpo_ocr \
  distributed.resources.visible_devices='[0,1,2,3]' \
  distributed.resources.trainer.devices='[0]' \
  distributed.resources.rollout.devices='[1,2,3]'
```

VRL does not manage SSH hosts, Ray worker startup, cloud security groups, or cluster lifecycle —
use `ray start`, the Ray VM launcher, KubeRay, or a managed Ray platform for that layer.

### Runtime environment inputs

Deployment facts that change by machine/cluster/shell stay **outside** experiment YAML:

| Input | How to set | Purpose |
| --- | --- | --- |
| `RAY_ADDRESS` | Environment variable | Connect to an already-running Ray head for cross-node rollout. |
| `CUDA_VISIBLE_DEVICES` | Environment variable | Limit the trainer process to local CUDA devices before Python starts. |
| `VRL_DATA_ROOT` | Environment variable | Root for artifact-backed datasets outside git (default `data/external`). |
| `HF_HOME` / `HF_HUB_CACHE` | Environment variables | Hugging Face model/cache location shared by model loaders. |
| `RANK` / `WORLD_SIZE` / `LOCAL_RANK` | Launcher environment | Torch distributed rank metadata. |
| `data.manifest`, `data.eval_manifest` | YAML / dotlist override | Prompt manifest selection (experiment data, not deployment state). |
| `trainer.output_dir` | YAML / dotlist override | Run output location for metrics, checkpoints, evals, reward artifacts. |

## Reward layers

Reward scoring is a decoupled pipeline so inference can run in-process or on a Ray pool:

- **RewardRollout** (`vrl/rewards/types.py`) — the data being scored.
- **RewardScorer** (`vrl/rollouts/collector/rewards.py`) — collector-side adapter: engine output → `RewardRollout` → device tensor.
- **RewardFunction** (`vrl/rewards/base.py`) — the reward objective (name, `score_key`, artifact build); aesthetic / CLIP / PickScore / OCR / … subclass it.
- **RewardInferenceRuntime** (`vrl/rewards/inference.py`) — local vs Ray transport that runs the scoring **RewardModel**.

## Memory policy

VRL keeps memory behavior at the layer that owns it. `model.memory` carries two structural
sub-blocks: `vae_decode` (model-build VAE tiling/slicing, applied by
`vrl.models.diffusion.common.vae_decode_memory`) and `frozen_offload` (parking frozen
driver-only modules on CPU before Ray rollout, applied by `vrl.trainers.frozen_module`).
Training memory stays under `actor` / distributed config (gradient checkpointing, FSDP CPU
offload). Rollout actor release stays in the Ray generation runtime; replay tensor storage stays
in trajectory/replay helpers because it affects correctness as well as memory.

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

Each sub-block is parsed and validated by its owner; unknown keys fail fast against the typed
dataclass fields. Execution lives with the owners, never in a shared "memory" module.

## Training examples

Curated qualitative results live under `docs/training_examples/` — concrete examples with
visible output; raw checkpoints and full generated artifacts stay under `outputs/`.

- `docs/training_examples/sd3_5_ocr_grpo/` — SD3.5 OCR GRPO qualitative result.

## Repository layout

```text
vrl/
  models/      diffusion (sd3_5, flux, qwen_image, wan_2_1, cosmos) + ar (janus_pro, nextstep_1) families
  generation/  pipeline executors + Ray generation runtime
  rollouts/    collector, orchestration (schedule modes), family registry
  rewards/     reward objectives, models, local/ray transport
  algorithms/  GRPO, flow-matching, DPO, DiffusionNFT
  trainers/    online (CEA) + offline trainers, weight sync, checkpointing
  trajectory/  trajectory build / resolve / storage
  config/      OmegaConf loading + typed schema (schema.py, validation.py)
  nn/ ray/ math/ utils/    shared kernels, Ray plumbing, helpers
  scripts/     train.py (vrl-train) + data/setup.py (dataset prep)
configs/    layered YAML: base / model / sampling / reward / dataset / distributed / experiment
datasets/   committed prompt datasets + per-dataset build scripts
docs/       architecture notes, sprints, training_examples/
```

## Roadmap

- Promote the first 🧪 video recipe to ✅ once a real run clears the validation bar.
- Broaden vLLM-Omni (`ar-vllm`) rollout coverage across AR families.
- Validate DiffusionNFT and DanceGRPO end-to-end on FLUX / Cosmos.
- Expand multi-GPU / cross-node FSDP2 online training beyond single-card colocated rollout.

## Acknowledgements

`visual-rl` builds on the open-source ecosystem — PyTorch, Ray, Hugging Face `diffusers` /
`transformers` / `peft`, and reward models including PickScore, CLIP, Kling VideoReward, and
PaddleOCR. The RL training layer draws on ideas from Flow-GRPO and related diffusion-RL work.
