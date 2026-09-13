# MiniMax-H3 / VDN-H3 four-L40S preflight

Status: technical preflight completed; no released weights downloaded and no
model GPU job launched. The user's name "dqn" is provisionally interpreted as
VDN-H3 / VideoDeltaNet, pending confirmation. This does not close either family
acceptance or the broader four-GPU hardware goal.

## Current evidence

- All four L40S GPUs had an empty compute-process inventory at preflight.
- Existing family sprints cover CPU tiny-model integration, not released-weight
  generation, replay, backward, or training on this machine.
- Main `.venv`: Torch 2.12.0+cu130, Diffusers 0.38.0, Transformers 4.57.6,
  Accelerate 1.13.0. Diffusers 0.38.0 lacks the H3 implementation.
- Isolated Diffusers 0.40.0 installed at
  `/mnt/nvme/venvs/h3-runtime-overlay`; the existing Transformers 5.13.0 overlay
  is `/mnt/nvme/venvs/transformers-5.13-overlay`. The combined environment
  successfully imported H3 transformer/scheduler and Qwen3-VL classes. No shared
  dependency was modified. Imports are not a model execution test.
- Candidate worktree: `/home/ubuntu/VRL-h3-l40s`, branch `feat/h3-four-l40s`,
  based on `75d69be2`. Existing SD3 acceptance artifacts remain untouched.
- NVMe has approximately 1.2 TB available; root filesystem has only 9.5 GB.
  Put weights, download caches, temporary files and output on NVMe.

## Released checkpoint inventory

Hugging Face model metadata (`?blobs=true`) reports MiniMax-H3 revision
`42ed227ee7df40d41602854ae760620d6eb651fe`, matching the pinned preset.
Only the root t2va components below are needed; do not download the duplicated
FL2VA/Ref2VA trees or `transformer_ref` for the text-to-video acceptance.

| Component | Safetensor files | Serialized bytes |
| --- | ---: | ---: |
| Transformer | 14 | 66,280,504,216 |
| Qwen3-VL text encoder | 14 | 66,714,912,872 |
| Video VAE | 3 | 10,415,558,888 |
| Audio VAE | 1 | 605,429,340 |

These sum to 144,016,405,316 serialized bytes, not peak runtime memory. Native
FP32 modules, FP32 VAE loading, activations, caches and training state require
additional space. Each of the two largest components exceeds one L40S.

VDN-H3 metadata revision: `51eeecefdb5b524c0df5539446d1dd54a17aa439`.
The 8-NFE `stage-dmd-step-250` adds 4,279,428,112 bytes of linear-branch weights,
334,026,912 bytes of default adapter and 851,452,696 bytes of turbo adapter.
Do not duplicate its `h3-base` download if the same base is already available.

## Runtime work still required

The SD3 four-rank adapter-only FSDP recipe is not an H3 deployment recipe:
replicating the 66 GB frozen transformer on each 48 GB card cannot work.
`MiniMaxH3Model.from_build` currently loads the workflow and moves the entire
encoder to `build.device`. The shared trainable preparation also uses explicit
single-device moves. The generic full-sequence probe has a single `--device`,
not a verified four-GPU component-placement implementation.

Required next implementation: partition the large conditioner and denoiser
internally, budget VAE decoding and activation memory, and ensure lifecycle
hooks do not gather or move the whole model onto one GPU. Preserve the H3
FP32 input/output projections, timestep modules and RoPE; do not flatten them
to BF16 just to satisfy an existing FSDP precision gate. Upstream modular
`load_components` accepts per-component loading kwargs, but this has not yet
been wired into or validated through our family runtime.

After deployment prerequisites are confirmed, proceed sequentially:

1. Download only the pinned required H3 components into NVMe.
2. Implement and verify model-parallel placement with real per-device memory
   evidence, then generate one valid native-geometry clip with the real weights.
3. Verify video output, finite trajectories and rollout/replay agreement.
4. Load the VDN 8-NFE artifact and repeat, using differentiable supported
   attention rather than enabling inference-only kernels on a training path.
5. Only after generation/replay pass, run a bounded real backward/update smoke.
   No long training queue and no automatic SD3 rerun are authorized here.

## Deployment prerequisite

The [official license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE)
was retrieved during preflight. Its territorial grant excludes the US, EU, UK
and South Korea and describes obtaining separate authorization for excluded
territories. The deployment region/authorization has not been confirmed by
the user; do not infer it from their timezone. The VDN derivative notice also
references this agreement. Asked the user to confirm eligibility or separate
authorization before downloading/running the released weights.

## Reproduce environment check

```bash
env CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=/mnt/nvme/venvs/h3-runtime-overlay:/mnt/nvme/venvs/transformers-5.13-overlay:/home/ubuntu/VRL-h3-l40s \
  /home/ubuntu/VRL/.venv/bin/python -c \
  'import torch, diffusers, transformers; print(torch.__version__, diffusers.__version__, transformers.__version__); from diffusers import MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler; from transformers import Qwen3VLForConditionalGeneration; print("H3 component imports available")'
```

Observed: `2.12.0+cu130 0.40.0 5.13.0`, then successful component imports.
All preflight command sessions exited. GPUs remain available; this document is
not a GPU claim or an assertion that H3 fits or runs successfully.

## 2026-09-13: released-config meta placement

Executed a weight-free placement probe against the pinned revision above.
The probe retrieved only the transformer and text-encoder JSON configurations,
constructed their real architectures with Accelerate `init_empty_weights`, and
asserted every parameter and buffer remained on the meta device. No released
weights were downloaded; no CUDA model was created. Process exited 0.

Artifacts:

- `/mnt/nvme/outputs/wan22_i2v_cache/h3_meta_placement_probe.py`
- `/mnt/nvme/outputs/wan22_i2v_cache/h3_meta_placement_20260913.json`

The JSON retains both input configurations and complete inferred device maps.
Accelerate `infer_auto_device_map` used the models' native no-split classes,
BF16 weight accounting, and explicit FP32 exceptions for the transformer's
`_keep_in_fp32_modules`. It counted 33,122,992,896 transformer parameters and
33,357,390,064 text-encoder parameters. Transformer accounting preserves 13
FP32 parameter/buffer tensors, including the computed RoPE buffer.

With a 32 GiB per-device weight budget, all mapped tensors stayed on the
requested GPU IDs, without CPU or disk fallback:

| Component | Device | Estimated resident weights and buffers (GiB) |
| --- | --- | ---: |
| Transformer | 0 | 30.408 |
| Transformer | 1 | 31.321 |
| Text encoder | 2 | 30.468 |
| Text encoder | 3 | 31.664 |

36 and 40 GiB budgets also produced GPU-only maps but packed more weights on
the first device of each pair. These are capacity estimates, not measured GPU
memory, executed dispatch, generation, or training acceptance. They exclude
both VAEs, activations, LoRA, gradients, optimizer state and transfer buffers.
In particular, fitting the two large components does not prove the full
pipeline fits with the FP32 video VAE co-resident.

Next technical gate: execute a small random-weight H3 with the proposed
cross-device dispatch and compare against its unsharded forward. Native H3
performs functional scatter/select operations outside leaf modules, so a
valid weight map alone cannot establish device-correct execution. Then wire
placement through the family loader and lifecycle without whole-model moves.
Released-weight execution still awaits the deployment and model-name
confirmations above; do not label this meta probe a model GPU run.

## 2026-09-13: two-device random-weight execution

Candidate `/home/ubuntu/VRL-h3-l40s` now contains the opt-in gate
`tests/models/families/minimax_h3/test_device_dispatch.py`. With physical GPUs
0 and 1 reserved, the final run exited 0: 2 passed in 6.59 seconds. Both tests
use a real two-block Diffusers H3 transformer, native family layout and fixed
conditioning from the tiny Qwen fixture, with identical single-device weights.

- FP32 full-parameter forward/backward passes `atol=rtol=1e-5`.
- Frozen BF16 base with native FP32 exceptions and FP32 rank-2 Q/K/V/out LoRA
  passes `atol=rtol=1e-3`. LoRA B is deliberately nonzero (normal, std 0.01).
- Both video and audio outputs are checked for finiteness and closeness.
  Parameter names, gradient presence and every present parameter gradient are
  compared. A nonzero gradient on the remote block is required.

The explicit map places each non-stack top-level component on GPU 0 and
individually assigns block 0 to GPU 0 and block 1 to GPU 1. Input projections,
normalization and selection heads stay on GPU 0. Accelerate dispatch hooks
perform the differentiable block transfers. This is not tensor parallelism
and does not establish simultaneous compute utilization or a speedup.

Failures retained as limitations, not converted into passes:

1. The abbreviated overlapping map `{"": 0, "transformer_blocks.1": 1}`
   failed forward: a nested Linear hook received CUDA 0 input with CUDA 1
   parameters. The successful test uses complete, non-overlapping mappings.
2. Training all BF16 base parameters passed the output gate but failed the
   `1e-3` gradient gate: one 32-element gradient had two mismatches and maximum
   absolute difference 0.0029296875. FP32 LoRA acceptance does not close this
   full-BF16-gradient failure; no tolerance was relaxed to claim it passed.
3. An intermediate test incorrectly shared the tiny conditioner autograd graph
   between two backward calls. Detaching the fixed input conditioning corrected
   that test fixture; text-encoder backward is outside this frozen-conditioning
   gate.

Reproduce from the candidate worktree:

```bash
env CUDA_VISIBLE_DEVICES=0,1 VRL_H3_DISPATCH_CUDA=1 \
  PYTHONPATH=/mnt/nvme/venvs/h3-runtime-overlay:/mnt/nvme/venvs/transformers-5.13-overlay:/home/ubuntu/VRL-h3-l40s \
  /home/ubuntu/VRL/.venv/bin/python -m pytest \
  tests/models/families/minimax_h3/test_device_dispatch.py -q
```

Ruff and formatting checks passed. All test processes are terminal and fresh
compute-process inventory is empty; GPU claims released. This does not test
released weights, full geometry, full LoRA rank, activation checkpointing,
optimizer updates, trainer lifecycle, four-device encoder/denoiser composition
or throughput. The original full deployment gates remain open. Next integrate
the validated non-overlapping block mapping with the loader/lifecycle and
validate the conditioner pair; do not feed the earlier automatic capacity map
unchanged into a production training claim.
