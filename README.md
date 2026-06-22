# visual-rl

RL post-training for visual generative models.

`visual-rl` trains diffusion and autoregressive image/video generators with one
layered-config trainer and one Collector -> Evaluator -> Algorithm loop. Recipes are
marked as validated only after real training runs show optimizer updates, non-flat
rewards, generated artifacts, and changed weights.

## Why It Exists

- **One training loop.** The same online RL loop drives diffusion and AR families
  instead of keeping a separate training script per model.
- **Layered configs.** Model, sampling, reward, dataset, algorithm, rollout, and
  distributed choices are composed from YAML layers under `configs/`.
- **Decoupled rewards.** OCR, aesthetic, CLIP, PickScore, Kling VideoReward, physics,
  and safety-style rewards share the same scoring contract.
- **Validation-first recipes.** Runnable wiring is not treated as a working recipe
  until a real run clears the promotion bar.

## Status Policy

| Status | Meaning |
| --- | --- |
| ✅ **Validated** | A real run proved optimizer updates, non-flat reward, artifacts, and changed weights. |
| 🧪 **Runnable** | Config, entrypoint, runtime path, and structural tests exist; training quality is not yet proven. |
| 🚧 **Planned** | Targeted, not wired end-to-end yet. |

## Supported Models

| Family | Modality | Algorithms | Status |
| --- | --- | --- | --- |
| **SD3.5** | text -> image diffusion | GRPO | ✅ OCR GRPO |
| **FLUX** | text -> image diffusion | GRPO-Guard, DanceGRPO, DiffusionNFT, Flow-DPPO | 🧪 Runnable |
| **Qwen-Image** | text -> image diffusion | GRPO | 🧪 Runnable |
| **Wan2.1** | text/image -> video diffusion | GRPO, DPO | 🧪 Runnable |
| **Wan2.2** | image -> video diffusion | GRPO | 🧪 Runnable |
| **Cosmos-Predict2** | video diffusion | GRPO | 🧪 Runnable |
| **Cosmos-Predict2.5** | video diffusion | DiffusionNFT | 🧪 Runnable |
| **Cosmos-Anima** | video diffusion | GRPO | 🧪 Runnable |
| **Janus-Pro** | autoregressive image | GRPO, R1-GRPO | 🧪 Runnable |
| **NextStep-1** | autoregressive image | GRPO | 🧪 Runnable |

## Supported Algorithms

| Algorithm | Config base |
| --- | --- |
| GRPO | `configs/base/algorithm/grpo.yaml` |
| GRPO-Guard | `configs/base/algorithm/grpo_guard.yaml` |
| DanceGRPO | `configs/base/algorithm/dance_grpo.yaml` |
| DiffusionNFT | `configs/base/algorithm/diffusion_nft.yaml` |
| Flow-DPPO | `configs/base/algorithm/flow_dppo.yaml` |
| Token-GRPO | `configs/base/algorithm/token_grpo{,_multisegment}.yaml` |
| DPO | `configs/base/algorithm/dpo.yaml` |

## Architecture

The online trainer runs:

```text
collect -> evaluate -> advantage -> loss -> backward -> step
```

Core contracts:

- **Collector** produces images, video, or AR tokens and records the trajectory.
- **Reward** scores the rollout through a common reward interface.
- **Evaluator** replays the trajectory through the current model.
- **Algorithm** consumes trajectory signals and computes the loss.
- **Trainer** applies the update and syncs weights for the next rollout.

## Repository Layout

```text
vrl/
  models/      diffusion and autoregressive model families
  generation/  executors and generation runtimes
  rollouts/    collector, orchestration, family registry
  rewards/     reward objectives, reward models, scoring transport
  algorithms/  GRPO, flow-matching, DPO, DiffusionNFT
  trainers/    online and offline trainers, weight sync, checkpointing
  trajectory/  trajectory build, resolve, and storage
  config/      OmegaConf loading and typed schema
  nn/ math/ utils/    shared kernels and helpers
  scripts/     training and data preparation entrypoints
configs/    layered YAML configs
datasets/   committed prompt datasets and dataset build scripts
docs/       architecture notes, sprint notes, training examples
```

## Current Focus

- Promote video recipes only after real training validation.
- Broaden AR rollout coverage.
- Validate DiffusionNFT and DanceGRPO on more model families.
- Expand multi-card and cross-node online training coverage.
