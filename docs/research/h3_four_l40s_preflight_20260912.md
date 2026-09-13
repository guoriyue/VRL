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

## 2026-09-13: explicit partitioned replay loader

The H3 candidate now accepts an explicit `block_devices` tuple in
`build_minimax_h3_replay_runtime_bundle`. The existing registry/default call is
unchanged; this is not a public distributed-config or trainer-strategy entry.
The family-owned `placement.py` builds a meta skeleton from the checkpoint
configuration, derives complete non-overlapping block ownership, and passes
the map into native Diffusers `from_pretrained` with low-CPU-memory loading.
Upstream mixed-precision loading remains responsible for FP32 exceptions.
Both schedulers still use the existing replay loader. The local build copy
sets `defer_trainable_device_move=True` for native PEFT preparation, avoiding
its whole-transformer `.to(root_device)` without mutating the caller's build.

The explicit path rejects full finetuning, compile, quantization, CPU roots
and implicit CUDA roots. Block counts and device index types are validated.
It has not established trainer ownership or compatibility with whole-model
offload/restore; callers must not infer those from this loader.

Final verification on GPUs 0 and 1: **31 passed in 7.30 seconds**, process
exited 0, covering the complete H3 family test directory. The new integration
test writes a two-block random-weight transformer as local 50 KB safetensor
shards plus both native scheduler configurations, then loads single-device
and partitioned replay bundles via the real loaders. Tests confirm:

- Native LoRA preparation leaves parameters on both GPUs and trainables FP32.
- Identical base/adapter state with nonzero B gives close video/audio outputs
  and present gradients (`atol=rtol=1e-3`); remote gradients are nonzero.
- One AdamW update (learning rate 1e-4) changes parameters on both GPUs,
  preserves every parameter's device and stays within the same single-device
  parameter comparison tolerance.
- Invalid map ownership and unsupported load modes fail; original H3 loading,
  backbone, scheduler and decode tests still pass.

Test artifacts reside at
`/mnt/nvme/outputs/wan22_i2v_cache/h3_partitioned_load_final_pytest`.
Reproduction uses the previous CUDA/overlay environment and runs
`python -m pytest tests/models/families/minimax_h3 -q` from the candidate.
Ruff, formatting and diff checks passed. All jobs terminal, fresh GPU compute
inventory empty, GPUs 0 and 1 released. No released weights downloaded.

Still open: full-sized checkpoint execution, actual rank-32 policy, encoder
pair and VAE placement, full native generation/replay, trainer lifecycle,
checkpoint recovery, quality, and controlled throughput. The tiny local shard
test does not prove released H3 peak memory or real video generation.

## 2026-09-13: partitioned conditioner and four-device connection

The candidate's family-owned placement module now provides explicit Qwen3-VL
decoder-layer mapping and a frozen conditioner loader. It derives maps from a
meta skeleton, loads local/native checkpoint shards directly to owners using
the rollout prompt dtype, and keeps the complete encoder frozen in eval mode.
Embedding, vision, norm and head modules stay on the encoder root; decoder
layers receive individually specified, non-overlapping device assignments.
No whole-encoder `.to()` follows the dispatched load.

Final H3 family suite with both CUDA opt-ins: **38 passed in 5.78 seconds**, exit
0. New four-device test stores the tiny real Qwen3-VL model in local 20 KB
safetensor shards and loads its two decoder layers across GPUs 2 and 3. It
uses the H3 family's actual `encode_prompt`, which reads the encoder's internal
model and intermediate hidden states rather than its top-level LM forward.
The first decoder layer is deliberately on GPU 3, and an execution hook
confirms that layer actually runs there. The selected intermediate embedding
crosses back to the DiT root on GPU 0.

Against identical weights with an unsharded encoder on GPU 2 and unsharded
DiT on GPU 0:

- Partitioned prompt embeddings pass `atol=rtol=1e-3`; repeated partitioned
  encoding is tensor-exact and executes the remote decoder layer again.
- A real tiny H3 DiT with its block on GPU 1 and packing/heads on GPU 0 consumes
  that conditioning through native sampling preparation and `forward_step`.
  Initial video latents are identical; finite video noise predictions pass
  `atol=rtol=1e-3`.
- Map tests reject incorrect layer counts, negative/string/bool devices and
  verify exactly one owner for every encoder parameter and buffer.

Local artifacts:
`/mnt/nvme/outputs/wan22_i2v_cache/h3_four_device_final_pytest`.
Reproduction uses `CUDA_VISIBLE_DEVICES=0,1,2,3`, `VRL_H3_FOUR_GPU=1` and
`VRL_H3_DISPATCH_CUDA=1` with the documented overlay environment, running
`python -m pytest tests/models/families/minimax_h3 -q` from the candidate.
An initial fixture preparation failed because deep-copying the audio VAE's
weight-norm tensors is unsupported; independently constructing components and
strictly copying encoder/DiT state fixed the fixture before successful runs.

Ruff, formatting and diff checks passed. All processes terminal; fresh compute
inventory empty, all four GPUs released. This is random-weight component
connection, not full-size model execution, parallel speedup, full generation,
audio decode or training acceptance. VAEs stayed on CPU and were not decoded
by the new four-device test. A unified generation loader and its component
offload/restore lifecycle, full geometry, released weights, quality and timing
remain open. No H3 weights downloaded while deployment confirmation is pending.

## 2026-09-13: independent video and audio VAE devices

H3 video decode previously created normalization tensors on the input latent
device and called the VAE without transferring its input. Audio decode had
the same assumption. Both now transfer latent inputs to their respective
VAE's actual device, perform native normalization/decode/postprocessing there,
and return the output to the input device. VAE parameters are not moved or
recast; video keeps the existing CUDA FP16 autocast behavior and both VAEs
retain FP32 storage. This allows explicit VAE placement without moving the
policy to the decoder's device.

Final family suite with GPUs 0 and 1 reserved: **39 passed, 1 skipped in 10.12
seconds**, exit 0. The four-device conditioner test is the skipped test because
its separate four-GPU opt-in was not enabled this turn. New parameterized
tests use CUDA 0 video/audio latents with real tiny VAEs on CPU or CUDA 1:

- Decoded video and stereo waveform match same-VAE-device controls exactly.
- Outputs return to CUDA 0; shapes and sample rate match the native contract.
- Inputs remain unchanged, outputs are finite, VAE locations remain unchanged,
  and VAE parameter dtype remains FP32.
- Existing CPU normalization/reference decode tests continue to pass.

Artifacts: `/mnt/nvme/outputs/wan22_i2v_cache/h3_independent_vae_pytest`.
Reproduction uses the two-GPU environment documented above and runs the full
H3 family test directory. Ruff, formatting and diff checks passed. All jobs
terminal, fresh compute inventory empty, GPUs 0 and 1 released.

This is device-correct decode, not full-size VAE capacity evidence. The released
FP32 video VAE may not fit beside a 31 GiB conditioner partition; a staged
conditioner parking/restoration policy and unified generation builder remain
necessary. No released weights or production training queue were started.

## 2026-09-13: staged conditioner parking and restoration

Added explicit `park_partitioned_text_encoder` in the H3 placement module.
This exclusive context requires a frozen eval-mode encoder with resident CUDA
parameters/buffers and an explicit GPU-only map. It removes Accelerate hooks,
clears the active map and moves the encoder to CPU. Its `finally` block
reinstalls the original dispatch map and execution hooks. Nested parking is
rejected. Callers must finish VAE work and move the VAEs off the vacated GPUs
before leaving the context; no concurrent prompt encoding is supported.

The four-device random-weight test now executes both normal and injected
decode-error paths. During parking it verifies every encoder parameter/buffer
is CPU-resident and Accelerate hooks are absent. The video VAE moves to GPU 2
and audio VAE to GPU 3, decodes finite outputs from GPU 0 latents, and returns
to CPU before the encoder is restored. Both paths verify the original map,
all parameter/buffer device assignments, dtypes and tensor values exactly.
Subsequent native H3 prompt encoding matches the pre-parking embedding exactly.
The injected error is observed rather than silently swallowed by the context.

Full H3 family suite with both GPU opt-ins: **40 passed in 6.41 seconds**,
exit 0. Artifacts:
`/mnt/nvme/outputs/wan22_i2v_cache/h3_park_restore_pytest`.
Use the documented four-device environment to reproduce the family suite.
Ruff, formatting and diff checks passed. All jobs terminal; fresh compute
inventory empty, all four GPUs released.

This closes the tested tiny staged component lifecycle, not a unified
generation-entry or trainer lifecycle gate. Real 32B encoder transfer latency,
host-memory peak, full VAE activation capacity and restoration under allocator
failure remain unmeasured. Full released-weight generation, quality and fair
performance comparisons remain open; no released weights were downloaded.

## 2026-09-13: unified explicit generation bundle

Added `partitioned_generation.py` with `H3GenerationPlacement`, a staged H3
model subclass, and `build_partitioned_h3_generation_runtime_bundle`. This
explicit Python entry reuses the shared generation-bundle builder and its
native LoRA/memory passes. It loads DiT and encoder with the validated maps,
loads only the remaining small components through the modular pipeline, and
keeps both VAEs on CPU until their decode call. Video/audio decode automatically
parks the encoder, moves the relevant VAE onto an encoder device, returns the
VAE to CPU, and restores the encoder. Whole-frozen-component moves are rejected.

Placement rejects policy/encoder overlap, VAEs outside the encoder device set,
invalid CUDA indices, empty ownership, non-LoRA, compile, quantization and
generic pipeline offload. The default family/registry path is unchanged; this
entry is not yet selected by public config or Ray.

The first construction attempt correctly hit the `ModelBuild` rule that
`defer_trainable_device_move` is replay-only. That rule remains unchanged.
Instead, the shared LoRA mixin now has a default-false protected device-preserve
hook, enabled only by the explicitly partitioned H3 generation subclass. The
original rollout build and its policy semantics remain intact.

Verification:

- Four-GPU H3 suite before adding the CPU placement guards: **41 passed in
  6.75 seconds**, exit 0. The new integration case loads real local random DiT
  and encoder shards, uses native LoRA preparation and a three-step family
  test denoise loop, and calls automatic video/audio decode. Outputs are finite,
  native shapes/rate hold, VAEs return to CPU, encoder ownership is restored,
  and subsequent prompt embeddings are exact. The tiny fixture explicitly uses
  intermediate layer 1 instead of released-model layer 50.
- Additional CPU placement guards plus shared LoRA, FP8-build ordering, Wan
  LoRA and Cosmos LoRA regression tests: **32 passed in 4.35 seconds**, exit 0.
- Ruff, formatting and diff checks passed.

The integration test substitutes modular metadata/small-component loading
with real already-constructed tiny components; DiT/encoder checkpoint loaders
and all component execution are real. It does not yet test native modular
metadata from disk, the production executor/collector, a complete released
trajectory, or MP4 output. Artifacts are under
`/mnt/nvme/outputs/wan22_i2v_cache/h3_unified_generation_rolefix_pytest`.
All jobs terminal, fresh compute inventory empty, all GPUs released. Released
weights, full geometry/peak memory, production ownership and controlled timing
remain open; no H3 download or long queue was started.

## 2026-09-13: native modular reload and production batch execution

Removed the substituted modular metadata/small-component loader from the
unified integration test. It now constructs a native `MiniMaxH3ModularPipeline`
for `t2va`, registers all eight real tiny components, installs the upstream
video scheduler before serialization, and calls native `save_pretrained`.
The resulting local directory includes modular metadata, tokenizer, processor,
both VAEs, both schedulers, encoder and transformer weights. The partitioned
generation builder reloads this directory with no loader monkeypatches.

The same test now submits an actual `GenerationRequest` to the production
`MiniMaxH3BatchExecutor.forward_batch`: one prompt/sample, 16x16, 8 frames,
3 steps, guidance 1, 24 fps and seed 17. In addition to the existing staged
decode assertions, it verifies finite returned video, observations, actions
and nonempty log-probs; observation/action shapes match, video is
`[1,3,8,16,16]`, and audio per-step replay tensors are present. After execution,
both VAEs are CPU-resident and the encoder is restored across GPUs 2 and 3.

Complete H3 family suite with both GPU opt-ins: **50 passed in 7.11 seconds**,
exit 0. Ruff, formatting and diff checks passed. Local native pipeline files
are retained under
`/mnt/nvme/outputs/wan22_i2v_cache/h3_native_executor_pytest`.
The documented four-GPU environment and full H3 family pytest command reproduce
this gate. All jobs terminal, fresh compute inventory empty, all GPUs released.

This replaces the earlier substituted-loader limitation for the tiny native
pipeline and adds local production batch execution. It does not establish
released-model geometry/weights, rollout/replay numeric agreement for these
executor trajectories, Ray/collector integration, reward/learning, full trainer
recovery, MP4 quality or throughput. Test-suite wall time is not a model speed
measurement. No released weights were downloaded or long queue started.

## 2026-09-13: native executor actions replayed by an independent model

Extended the native modular/executor test with a separately loaded partitioned
replay bundle. Before executor generation, LoRA B is initialized nonzero
(normal, std 0.01). The replay bundle independently loads its checkpoint and
receives the generator's complete policy state through strict state loading.
It reconstructs each recorded observation with the exported prompt/audio
payload and native geometry, visiting steps in reverse order. Recorded actions
are re-scored using the executor's resolved SDE type/noise level and schedule.

All three recorded steps pass the predeclared absolute `1e-3` log-probability
and policy-ratio gates, with measured **log-prob max absolute error 0** and
**max absolute deviation of ratio from 1 equal to 0**. Backpropagating the
negative mean re-scored log-probability gives 16 finite gradient tensors,
12 nonzero. This is a differentiation check, not a reward-weighted training
objective or proof of learning.

Final H3 suite: **50 passed in 7.27 seconds**, exit 0. Raw metrics:
`/mnt/nvme/outputs/wan22_i2v_cache/h3_executor_replay_metrics_pytest/test_unified_partitioned_gener0/executor_replay_metrics.json`.
Local modular checkpoint files remain beside this JSON. Reproduction uses the
documented four-GPU environment and H3 family suite. Ruff, formatting and diff
checks passed. All jobs terminal, fresh compute inventory empty, all GPUs
released.

This closes executor-to-replay log-prob agreement for the tiny nonzero-LoRA
checkpoint, not full-size released weights, collector/trainer contracts,
optimizer/recovery semantics for this trajectory, quality or performance.
The test co-resides separate tiny rollout and replay models on GPUs 0-1;
released-model replication would require its own memory/scheduling design.

## 2026-09-13: standalone local generation probe and MP4

Added `python -m vrl.scripts.generation.partitioned_h3_probe`. It requires an
existing local checkpoint, a new output directory, prompt and explicit
geometry/step count. It derives contiguous layer ownership from checkpoint
configs, uses the existing H3 rank-32/alpha-64 LoRA preset and VAE tiling, and
runs one production executor batch. Outputs include MP4, JSON timings/memory
counters and optional serialized trajectory. It neither starts training nor
selects a public Ray strategy. Full-size default block ownership is a balanced
count heuristic, not a capacity guarantee.

The first standalone attempt failed before model loading because CUDA peak
counters were reset before the allocators had been initialized. Initializing
each device before measurement fixed that. The next attempt exposed an actual
builder bug absent from prior tests: populated rollout-only generation-memory
policy was carried into the component's replay-only loading build. The helper
now clears that field only in its temporary replay loading slice; the original
rollout retains tiling. Unified generation regression tests now enable tiling.
Both failed output directories are retained, not presented as successful runs.

Successful offline CLI artifacts:
`/mnt/nvme/outputs/wan22_i2v_cache/h3_standalone_probe_memoryfix`.
Input is the retained tiny native modular directory from the executor/replay
test. DiT block on GPU 1 with root on GPU 0; encoder layers on GPUs 2 and 3;
selected tiny hidden state 1, text cap 8, 16x16, 8 frames, 3 steps, seed 42.
Native rank-32 LoRA and VAE tiling were active. Process exited 0:

- Load: 3.188846 seconds; generation: 0.872050 seconds.
- Stage counters: encode 0.411656, prepare 0.001893, denoise 0.168102,
  decode 0.290205 seconds.
- Independent MP4 read verified 8 RGB frames at 16x16 and 24 fps; decoded
  pixel standard deviation 23.9333. This is nonblank output, not model quality.
- `trajectory.pt` and `result.json` retained. JSON explicitly warns that the
  executor resets current-device peak counters at stage boundaries; its
  post-executor counters are not guaranteed whole-generation high-water marks.
  Load peaks and executor stage memory are reported separately. These tiny
  measurements are not estimates of released H3 capacity or throughput.

Reproduce from the H3 candidate, using a new output directory:

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 \
  PYTHONPATH=/mnt/nvme/venvs/h3-runtime-overlay:/mnt/nvme/venvs/transformers-5.13-overlay:/home/ubuntu/VRL-h3-l40s \
  /home/ubuntu/VRL/.venv/bin/python -m vrl.scripts.generation.partitioned_h3_probe \
  --path /mnt/nvme/outputs/wan22_i2v_cache/h3_executor_replay_metrics_pytest/test_unified_partitioned_gener0 \
  --output /mnt/nvme/outputs/wan22_i2v_cache/h3_standalone_repeat \
  --prompt 'a wooden block slides across a table' \
  --policy-root 0 --policy-devices 1 --encoder-devices 2 3 \
  --encoder-layer 1 --max-text-tokens 8 \
  --height 16 --width 16 --frames 8 --steps 3 --save-trajectory
```

H3 and probe partitioning regressions: **56 passed in 7.28 seconds**, exit 0.
Ruff, formatting and diff checks passed. The CLI subsequently adds policy root
explicitly to new reports; this run used root 0 as the command records. All
jobs terminal, fresh compute inventory empty, all GPU claims released. No
released weights downloaded; deployment/name confirmations and full-size
correctness, quality and fair performance gates remain open.
