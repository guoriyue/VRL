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
  distributed choices are composed from bundled YAML layers under
  `vrl/config/presets/`.
- **Decoupled rewards.** OCR, aesthetic, CLIP, PickScore, Kling VideoReward, physics,
  and safety-style rewards share the same scoring contract.
- **Validation-first recipes.** Runnable wiring is not treated as a working recipe
  until a real run clears the promotion bar.

## Why not an LLM-RL framework?

RL frameworks built for text LLMs (slime, verl, OpenRLHF, TRL) assume a shape that
visual generation does not have, so bending them to diffusion/video RL fights the
core abstractions at every step:

- **Rollout is a multi-step denoising trajectory** — 20–50 full DiT forwards per
  sample, not one token-by-token pass over a KV-cache.
- **The log-prob is a continuous-latent Gaussian density** (flow-matching / SDE
  transition), not a categorical token cross-entropy.
- **The reward is on decoded pixels/video** (VAE decode → image/video reward
  model), not on text.
- **Conditioning is world-model-shaped** (reference image/video, I2V, V2W), not a
  text prefix.

`visual-rl` is built around exactly these — diffusion *and* autoregressive image
*and* video — behind one rollout / replay / algorithm contract. See
[`docs/NORTH_STAR.md`](docs/NORTH_STAR.md) for the full positioning and roadmap.

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
| **SANA** | text -> image diffusion | GRPO | 🧪 Runnable |
| **Lumina-Image 2** | text -> image diffusion | GRPO | 🧪 Runnable |
| **HunyuanImage 2.1** | text -> image diffusion | GRPO | 🧪 Runnable |
| **PixArt-Σ** | text -> image diffusion | GRPO | 🧪 Runnable |
| **Wan2.1** | text/image -> video diffusion | GRPO, DPO | 🧪 Runnable |
| **Wan2.2** | image -> video diffusion | GRPO | 🧪 Runnable |
| **HunyuanVideo** | text -> video diffusion | GRPO | 🧪 Runnable |
| **Mochi 1** | text -> video diffusion | GRPO | 🧪 Runnable |
| **CogVideoX** | text -> video diffusion | GRPO | 🧪 Runnable |
| **Echo** | text -> video diffusion | GRPO | 🧪 Runnable |
| **Cosmos-Predict2** | video diffusion | GRPO | 🧪 Runnable |
| **Cosmos-Predict2.5** | video diffusion | DiffusionNFT | 🧪 Runnable |
| **Cosmos-Anima** | video diffusion | GRPO | 🧪 Runnable |
| **Janus-Pro** | autoregressive image | GRPO, R1-GRPO | 🧪 Runnable |
| **NextStep-1** | autoregressive image | GRPO | 🧪 Runnable |
| **Emu3** | autoregressive image | GRPO | 🚧 Recipe pending |
| **GLM-Image** | autoregressive image | GRPO | 🚧 Recipe pending |
| **LlamaGen** | autoregressive image | GRPO | 🚧 Recipe pending |
| **Cosmos3** | text -> video diffusion | GRPO | 🚧 Run-gated |

## Supported Algorithms

| Algorithm | Config base |
| --- | --- |
| GRPO | `vrl/config/presets/base/algorithm/grpo.yaml` |
| GRPO-Guard | `vrl/config/presets/base/algorithm/grpo_guard.yaml` |
| DanceGRPO | `vrl/config/presets/base/algorithm/dance_grpo.yaml` |
| DiffusionNFT | `vrl/config/presets/base/algorithm/diffusion_nft.yaml` |
| Flow-DPPO | `vrl/config/presets/base/algorithm/flow_dppo.yaml` |
| Token-GRPO | `vrl/config/presets/base/algorithm/token_grpo{,_multisegment}.yaml` |
| DPO | `vrl/config/presets/base/algorithm/dpo.yaml` |

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
  config/      OmegaConf loading, typed schema, and bundled YAML presets
  nn/ math/ utils/    shared kernels and helpers
  scripts/     training and data preparation entrypoints
datasets/   committed prompt datasets and dataset build scripts
docs/       architecture notes, sprint notes, training examples
third_party/  vendored submodules + editable-install wrappers
```

## Setup

A bare `git clone` does **not** fetch submodules, so run once after cloning:

```bash
make setup
```

That fetches the vendored submodules and editable-installs the **base** package
plus the vendored submodule wrappers. It is the only setup step; re-run it after a
submodule bump. Base install ≠ feature extras — see **Dependencies** below for the
one or two extras your use case needs (the quickstart needs `.[cosmos,ocr]`).

### Why a setup step (vendored submodules)

Some model/reward backends are upstream code that ships no Python packaging
(JoyAI-Echo's `ltx_*`, videophy's `mplug_owl_video`). They live as git
submodules under `third_party/`, each paired with a thin editable-install wrapper
(`third_party/<name>_packaging/`) that exposes its packages — so `vrl/` contains
**no** `sys.path` injection. `make setup` fetches the submodules, then discovers
and editable-installs **every** `third_party/*_packaging/` wrapper, so adding a
vendored dependency needs no edit outside `third_party/`. See
[`third_party/README.md`](third_party/README.md) for the convention.

## Dependencies

`make setup` installs the **base** package only. Each use case adds one or two
optional-dependency groups; groups **compose** in a single `pip install`:

| Use case | Install | Brings (why) |
|---|---|---|
| Diffusion families (SD3.5 / Flux / Cosmos / Wan / Qwen …) | `.[cosmos]` | diffusers + transformers + peft + torchvision |
| AR-image families (Janus-Pro / NextStep) | `.[cosmos]` | transformers/peft model runtime (vLLM accel is separate — see note) |
| OCR reward (the validated quickstart) | `.[ocr]` | paddleocr |
| Video / VLM reward (Kling, VideoScore2, UnifiedReward) | `.[reward]` | transformers≥5.13, qwen-vl-utils, opencv |
| Pose / motion / anatomy eval | `.[pose]` (CPU) · `.[pose-gpu]` (GPU) | onnxruntime + opencv |
| Dataset prep (video-world, pickapic) | `.[data]` | datasets, pyarrow, av |
| Fixed video-eval suite (VBench) | `.[videoeval]` | vbench |
| Full-param 8-bit Adam (Cosmos trustworthy-curve recipe) | `.[optim8bit]` | bitsandbytes (int8 Adam state, RL-safe) |
| Tests / lint | `uv sync --group test --group lint` | pytest, ruff |

Example — the SD3.5-OCR quickstart below needs `pip install -e ".[cosmos,ocr]"`.
(The `cosmos` group is the core model-runtime extra and is misnamed for history —
it serves *all* diffusion and AR families, not just Cosmos.)

> **`ar-vllm` is optional; a separate environment is recommended.** AR-image families
> run in the main env without it via `sampling.attention_backend=torch_native`;
> `.[ar-vllm]` only adds vLLM's internal paged-attention / blockwise-fp8 kernels.
> vLLM pins its Torch/TorchVision/TorchAudio ABI. The current lock resolves it with
> `.[cosmos]`, but a dedicated venv keeps this large, tightly pinned accelerator
> stack isolated — the repo already ships one at `.venvs/vllm-omni`.

## Quickstart

After `make setup`, install the two extras the validated recipe needs and launch
it — SD3.5 text-to-image GRPO with an OCR reward:

```bash
pip install -e ".[cosmos,ocr]"
vrl-train --config experiment/diffusion/sd3_5/online_grpo_ocr
```

`--config` accepts a bundled config name (no extension) or an absolute YAML path;
trailing args are OmegaConf dotlist overrides (`vrl-train --help`):

```bash
# shorter smoke run
vrl-train --config experiment/diffusion/sd3_5/online_grpo_ocr \
    trainer.total_epochs=2 trainer.seed=0
```

Within the first few epochs you should see optimizer steps and a **non-flat**
`reward_mean`. A flat reward is a bug, not a result (see Status Policy). Every
recipe lives under `vrl/config/presets/experiment/` — browse it to see what runs.

## Current Focus

- Promote video recipes only after real training validation.
- Add training recipes for the runtime-verified Emu3, GLM-Image, and LlamaGen families.
- Validate DiffusionNFT and DanceGRPO on more model families.
- Expand multi-card and cross-node online training coverage.

## Docs

- [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md) — positioning, moat, and roadmap (why visual-rl, not slime/verl).
- [`docs/ADDING_A_MODEL_FAMILY.md`](docs/ADDING_A_MODEL_FAMILY.md) — add a model module, registry descriptor, presets, and contract tests without forking the trainer.
