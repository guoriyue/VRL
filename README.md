# visual-rl

RL post-training for visual generative models.

`visual-rl` trains visual generative policies with one layered-config trainer and
one Collector -> Evaluator -> Algorithm loop. Policies are classified by generation
regime (`full_sequence`, `token_autoregressive`, or `chunk_autoregressive`) and
policy step (`denoise` or `token`), rather than forcing every model into an
AR-or-diffusion bucket.
Recipes are marked as validated only after real training runs show optimizer
updates, non-flat rewards, generated artifacts, and changed weights.

## Why It Exists

- **One training loop.** The same online RL loop drives full-sequence denoise and
  token-autoregressive policies instead of keeping a separate training script per model.
- **Layered configs.** Model, sampling, reward, dataset, algorithm, rollout, and
  distributed choices are composed from bundled YAML layers under
  `vrl/config/presets/`.
- **Decoupled rewards.** OCR, aesthetic, CLIP, PickScore, Kling VideoReward, physics,
  and safety-style rewards share the same scoring contract.
- **Validation-first recipes.** Runnable wiring is not treated as a working recipe
  until a real run clears the promotion bar.

## Why not an LLM-RL framework?

RL frameworks built for text LLMs (slime, verl, OpenRLHF, TRL) assume one causal
categorical-token shape. Visual policies span several shapes, so forcing all of
them through that abstraction fights the core contracts:

- **Generation regime varies** — policies update a full latent sequence, emit
  one token from a prefix, or advance one causal temporal chunk while denoising
  inside that chunk. These are distinct time organizations, not AR/diffusion
  synonyms.
- **The policy step varies** — denoise policies record continuous flow/Gaussian
  transitions, while token policies record categorical or continuous token
  actions.
- **The reward is on decoded pixels/video** (VAE decode → image/video reward
  model), not on text.
- **Conditioning is world-model-shaped** (reference image/video, I2V, V2W), not a
  text prefix.

`visual-rl` keeps these policy semantics explicit behind one rollout / replay /
algorithm contract. See [`docs/MODEL_TAXONOMY.md`](docs/MODEL_TAXONOMY.md) for
the typed semantics for the full
positioning and roadmap.

## Status Policy

| Status | Meaning |
| --- | --- |
| ✅ **Validated** | A real run proved optimizer updates, non-flat reward, artifacts, and changed weights. |
| 🧪 **Runnable** | Config, entrypoint, runtime path, and structural tests exist; training quality is not yet proven. |
| 🔌 **Integrated** | Model/runtime wiring and rollout parity exist, but a complete experiment recipe or environment contract is still missing. |
| 🚧 **Planned** | Targeted, not wired end-to-end yet. |

## Supported Models

| Family | Modality | Generation regime / step | Algorithms | Status |
| --- | --- | --- | --- | --- |
| **SD3.5** | text -> image | full_sequence / denoise | GRPO, V-GRPO | ✅ OCR GRPO |
| **FLUX** | text -> image | full_sequence / denoise | GRPO-Guard, DanceGRPO, DiffusionNFT, Flow-DPPO | 🧪 Runnable |
| **Qwen-Image** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **SANA** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Lumina-Image-2** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **HunyuanImage-2.1** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **PixArt-Sigma** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **CogVideoX** | text -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **HunyuanVideo** | text -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Mochi-1** | text -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Wan2.1** | text/image -> video | full_sequence / denoise | GRPO, DPO | 🧪 Runnable |
| **Wan2.2** | image -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Cosmos-Predict2** | video -> world | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Cosmos-Predict2.5** | text -> world | full_sequence / denoise | GRPO, DiffusionNFT | 🧪 Runnable |
| **Cosmos-Anima** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Echo** | text -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Janus-Pro** | text -> image | token_autoregressive / token (R1: multisegment) | GRPO, R1-GRPO | 🧪 Runnable |
| **NextStep-1** | text -> image | token_autoregressive / token (continuous action) | GRPO | 🧪 Runnable |
| **Emu3** | text -> image | token_autoregressive / token | Token-GRPO | 🔌 Integrated |
| **GLM-Image** | text -> image | token_autoregressive / token | Token-GRPO | 🔌 Integrated |
| **LlamaGen** | text -> image | token_autoregressive / token | Token-GRPO | 🔌 Integrated |
| **Cosmos3** | text -> video | full_sequence / denoise | — | 🔌 Integrated |
| **MiniMax-H3** (Hailuo 3.0) | text -> video (+ audio side stream) | full_sequence / denoise | GRPO (recipe only) | 🔌 Integrated (33B + 32B conditioner, ~144 GB: needs multi-GPU FSDP; CPU tiny-real parity only; diffusers>=0.40) |
| **VDN-H3** (VideoDeltaNet on MiniMax-H3) | text -> video (+ audio side stream) | full_sequence / denoise | GRPO (recipe only) | 🔌 Integrated (hybrid window-softmax + linear attention grafted on the H3 backbone; 8-NFE distilled artifact; vendored submodule; CPU tiny-real only) |
| **CausVid** | text -> video | chunk_autoregressive / denoise | GRPO | 🔌 Integrated (real-weight proof pending; non-commercial weights) |
| **MAGI-1 4.5B** | text/image/video -> video | chunk_autoregressive / denoise | generation only | 🔌 Integrated (isolated upstream runtime; no replay API) |

`FAMILY_REGISTRY` is the canonical runtime roster. This table reports user-facing
recipe readiness as well, so a registered family remains Integrated until a complete
experiment config and its dependency contract are committed.

## Supported Algorithms

| Algorithm | Config base |
| --- | --- |
| GRPO | `vrl/config/presets/base/algorithm/grpo.yaml` |
| GRPO-Guard | `vrl/config/presets/base/algorithm/grpo_guard.yaml` |
| DanceGRPO | `vrl/config/presets/base/algorithm/dance_grpo.yaml` |
| DiffusionNFT | `vrl/config/presets/base/algorithm/diffusion_nft.yaml` |
| V-GRPO | `vrl/config/presets/base/algorithm/v_grpo.yaml` |
| Flow-DPPO | `vrl/config/presets/base/algorithm/flow_dppo.yaml` |
| Token-GRPO | `vrl/config/presets/base/algorithm/token_grpo{,_multisegment}.yaml` |
| DPO | `vrl/config/presets/base/algorithm/dpo.yaml` |

## Architecture

The online trainer runs:

```text
collect -> evaluate -> advantage -> loss -> backward -> step
```

Core contracts:

- **Collector** produces decoded images/video and records the policy trajectory.
- **Reward** scores the rollout through a common reward interface.
- **Evaluator** replays the trajectory through the current model.
- **Algorithm** consumes trajectory signals and computes the loss.
- **Trainer** applies the update and syncs weights for the next rollout.

## Repository Layout

```text
vrl/
  models/
    families/  family-owned checkpoints, backbones, and replay projections
    steps/     shared denoise/token model contracts and builders
  generation/
    steps/        denoise loop and token-step protocol
    composition/  reusable generation-regime state machines
    bindings/     full-sequence, token-AR, and temporal-chunk denoise assemblies
    execution/ ray/  step-neutral execution and distributed lifecycle
  families/    policy semantics and canonical runtime registry
  rollouts/    collector, orchestration, and replay evaluation
  rewards/     reward objectives, reward models, scoring transport
  algorithms/  GRPO, flow-matching, DPO, DiffusionNFT
  trainers/    online and offline trainers, weight sync, checkpointing
  trajectory/  trajectory build, resolve, and storage
  config/      OmegaConf loading, typed schema, and bundled YAML presets
  nn/ math/ utils/    shared kernels and helpers
  scripts/     training and data preparation entrypoints
datasets/   committed prompt datasets and dataset build scripts
docs/       architecture notes, sprint notes, training examples
third_party/  vendored submodules (+ the CountGD Bazel package)
```

## Setup

Bazel owns dependencies, builds, tests and entry points. Install
[Bazelisk](https://github.com/bazelbuild/bazelisk) (it reads `.bazelversion`),
then fetch the vendored submodules once after cloning:

```bash
make setup          # git submodule update --init --recursive; builds the entry points
```

No virtualenv, `pip install`, `CUDA_HOME` or system CUDA toolkit is used:
Bazel downloads the pinned CPython 3.12.13, every wheel from `uv.lock`, the
CUDA 13.0.2 toolkit and the LLVM host compiler. What the host must still provide
is the NVIDIA driver (CUDA 13 capable), glibc ≥ 2.28 and `git`/`patch`.

Python packaging (`pyproject.toml`, `uv build`) stays for wheel/sdist
publication; it is not the way to set up a development or training
environment any more.

### Vendored submodules

Some model/reward backends are upstream code that ships no Python packaging
(JoyAI-Echo's `ltx_*`, videophy's `mplug_owl_video`, CausVid, VDN-H3). They
live as git submodules under `third_party/`; `//third_party:vendored` puts their
source roots on the import path of every `vrl` target, so `vrl/` contains no
`sys.path` injection. See [`third_party/README.md`](third_party/README.md).

## Dependencies

`uv.lock` is the single dependency source. Bazel exports it per dependency
stack at fetch time (`tools/dependencies/uv_exports.bzl`) and builds the same
`vrl/` sources against one stack per target:

| Target | Stack (`uv.lock` extras) | Used by |
|---|---|---|
| `//:vrl` | core + cosmos, reward, reward-service, data, ocr, detection, optim8bit + test/lint | trainer, reward service, all CPU/GPU lanes |
| `//:vrl_vllm` | core + ar-vllm + test | vLLM paged attention and CuMem memory parking (`//tests:gpu_vllm_tests`) |
| `//:vrl_shared_gpu` | both of the above (identical shared pins) | shared-GPU topologies: model stack plus vLLM's allocator (real-weight lane) |
| `//:vrl_countgd` | `third_party/countgd/requirements.txt` (transformers 4.48, numpy 1.26, torch cu128) | CountGD counting reward service |
| `//:vrl_videoeval` | core + videoeval + test (tokenizers 0.13.3 built from source with a pinned Rust toolchain) | `//:video_reward_suite` (VBench) |

Not migrated: **MAGI-1** (its official code needs a flash-attn 2.4.2 / torch 2.4 source build;
create `third_party/MAGI-1/.venv` from that submodule's requirements and point
`model.python_executable` at it). Ray never uploads virtual environments: the
driver resolves a path-like executable to an absolute path before launch, and
every rollout node must provide that path through a shared mount or container
image. The official MAGI 4.5B process fixes the DiT to BF16 and T5 to FP32, so
its launch config must use rollout `dtype: bf16`, `outer_autocast: false`,
prompt-encoder `dtype: fp32`, and `float32_precision: ieee`.

CausVid stays in the main stack, but its pinned Wan cross-attention calls
FlashAttention directly, which is not in the lock; CausVid rollouts need a
CUDA-compatible `flash-attn` build in the environment that runs them.

### Commands

```bash
make verify                                              # = uv lock --check && bazel test //...
bazel test //tests:config_tests //tools/lint:ruff_check  # one lane
bazel test --config=gpu //tests:gpu_tests //tests:gpu_vllm_tests   # real GPU (manual)
HF_HOME=~/.cache/huggingface WM_REAL_MODEL_RL_CASES=sd3_5 \
  bazel test --config=gpu --config=real_weights //tests:e2e_real_checkpoint_tests
bazel run //:vrl_train -- --config experiment/sd3_5/online_grpo_ocr
bazel run //:vrl_supervise -- --config experiment/sd3_5/online_grpo_ocr   # torchrun DDP/FSDP on one host
bazel run //:vrl_reward_service -- --config vrl/config/reward_service/<service>.yaml
bazel run //third_party/countgd:reward_service -- --config vrl/config/reward_service/countgd.yaml
bazel run //:video_reward_suite -- --video-dir <dir>       # VBench stack
bazel build --build_python_zip //:vrl_train              # self-contained bazel-bin/vrl_train.zip for other nodes
bazel build --build_python_zip //tools/delivery:node_probe   # copy to a node: python3 node_probe.zip
```

Every `py_binary`/`py_test` runs from a runfiles venv whose `sys.executable`
carries the whole closure, so torchrun ranks, Ray workers and reward workers
inherit it. See [docs/research/bazel_migration.md](docs/research/bazel_migration.md)
for the acceptance record.

## Quickstart

After `make setup`, launch the validated recipe — SD3.5 text-to-image GRPO with
an OCR reward:

```bash
bazel run //:vrl_train -- --config experiment/sd3_5/online_grpo_ocr
```

`--config` accepts a bundled config name (no extension) or an absolute YAML path.
Trailing arguments compose independent presets with `+group=option` and set
individual values with OmegaConf dotlist overrides (`vrl-train --help`):

```bash
# shorter smoke run
bazel run //:vrl_train -- --config experiment/sd3_5/online_grpo_ocr \
    trainer.total_epochs=2 trainer.seed=0
```

For a new combination, use a reward/data-neutral execution recipe rather than
adding another model-by-reward experiment YAML:

Anima has only two execution templates: `online_grpo` (LoRA) and
`online_grpo_fullparam`. Neither selects rewards or data; compose those at launch.

```bash
python -m vrl.scripts.train \
    --config experiment/anima_preview3/online_grpo \
    +reward=ocr +dataset=ocr \
    actor.optim.lr=1e-5 trainer.total_epochs=2 \
    trainer.output_dir=outputs/anima_ocr_composed
```

This demonstrates composition, not a recommended Anima training recipe. The
standard OCR dataset must be available at the paths declared by `dataset/ocr`.
The same arguments work with `python -m vrl.scripts.supervise`. Preset overlays
merge in order; ordinary dotlist values apply last. `+reward=` is additive, not
replacement: multiple different reward presets produce a multi-reward config.
See [runtime configuration composition](docs/CONFIGURATION.md) for independent
judge/rubric selection, evaluation, and historical recipe migration.

Within the first few epochs you should see optimizer steps and a **non-flat**
`reward_mean`. A flat reward is a bug, not a result (see Status Policy). Every
recipe lives under `vrl/config/presets/experiment/` — browse it to see what runs.

## Current Focus

- Promote video recipes only after real training validation.
- Add training recipes for the runtime-verified Emu3, GLM-Image, and LlamaGen families.
- Validate DiffusionNFT and DanceGRPO on more model families.
- Expand multi-card and cross-node online training coverage.

## Docs

- [`docs/MODEL_TAXONOMY.md`](docs/MODEL_TAXONOMY.md) — policy axes, current family profiles, and physical layout.
- [`docs/ADDING_A_MODEL_FAMILY.md`](docs/ADDING_A_MODEL_FAMILY.md) — add a model module, registry descriptor, presets, and contract tests without forking the trainer.
- [`docs/PRECISION.md`](docs/PRECISION.md) — base dtypes, selective FP8 quantization, protected diffusion math, and frozen rollout components.

### Online training batch sizes

Online training uses collection sizes; `actor.gradient_accumulation_steps` is
reserved for offline DPO and is rejected by the online batch plan.

- `rollout.prompts_per_batch`: planned prompt count per training rank for one online optimizer update.
- `rollout.n_samples_per_prompt`: generated samples in each prompt group.
- `actor.prompts_per_collection`: prompt groups collected together during streaming.
  Zero or unset keeps the full-batch path. A positive value must divide
  `prompts_per_batch`; the plan derives `collections_per_update`.
- `actor.training_microbatch_size`: samples processed together within each prompt
  group during training. Zero keeps the group whole; the default is one sample.

For example, 4 prompts with 8 samples each plan 32 generated samples per update.
Collecting 2 prompts at a time gives 2 collection batches; a training microbatch
size of 2 splits each prompt group into four training computations. Filtering and
distributed execution can change the number of contributing samples. These
settings do not introduce a second independent `minibatch_size` parameter.

The previous actor keys `microbatch_size` and `samples_per_replay_batch` are
replaced by `prompts_per_collection` and `training_microbatch_size`, respectively.
