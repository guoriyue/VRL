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
