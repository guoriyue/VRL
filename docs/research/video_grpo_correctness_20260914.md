# Video GRPO correctness evidence

The user clarified that the objective is RL/framework correctness on a video
model, preferably VDN-H3 linear attention, not a long full-resolution quality
experiment. The cancelled 81-frame Wan update is not a required proxy for every
correctness question and is not automatically restarted.

## Wan real-weight evidence revalidated

The independent CPU audit `revalidate_video_grpo_20260914.py` reopened the actual
single-card, four-card and cold-resume Wan 2.2 checkpoints under
`/mnt/nvme/outputs/wan22_i2v_cache`. Its new report is
`video_grpo_revalidation_20260914.json`; historical reports were not overwritten.

The existing native runs use real model weights and real Kling reward, 320x320,
17 frames, two optimizer updates and eight global samples per update. Verified:

- Both updates in both arms have zero pre-update rollout/replay log-prob error
  and zero initial clipping, with positive finite recorded gradient norms.
- Step two changes all 1280 adapter tensors. Complete finite model/Adam state,
  progress and rank-local RNG are present in the saved checkpoints.
- Single/four-card final model maximum absolute difference is
  4.656612873077393e-10; this is measured near-equivalence, not bitwise identity.
- Cold resume equals uninterrupted training across 6412 tensor leaves, four
  arrays and 8962 scalar leaves. The sole representation difference remains
  optimizer betas tuple versus list, with identical values.
- Existing native acceptance receipts report clean process exit and no Ray
  memory-threshold/worker-kill records for the passing runs and resume.

This is strong bounded evidence for the tested video GRPO update and distributed
state contracts. It is not a theorem about every configuration, a new native run
on the cleaned main branch, or evidence of improved held-out video quality.

## VDN reward-to-update integration

The fixed upstream source revision 57edaf696f19f5c0997d2dc63e14863f926dfeee was
checked out separately at `wan22_i2v_cache/vdn-pinned-57edaf69`, without changing
the original dirty vendor tree. No official model weights were downloaded.

`tests/models/families/vdn_h3/test_grpo_update.py` now exercises:

- Two distinct trajectories through the production VDN batch executor, real
  tiny hybrid transformer and video/audio components, with deterministic seeds.
- A separately constructed VDN training-side replay model, loaded strictly with
  the same transformer state. All timesteps replay in reverse order and match
  recorded log-probs to absolute tolerance 1e-6.
- Production GRPO advantages from controlled rewards [0, 1], and the unclipped
  on-policy loss derivative equal to negative advantage divided by group size.
- Reversing advantages reverses the hybrid projection gradient; tied rewards
  produce exactly zero gradient with KL disabled.
- Finite nonzero gradient, an actual AdamW parameter update and decreased
  surrogate loss on the fixed recorded trajectories.

The test uses random tiny weights, the differentiable reference attention
backend and controlled rewards. It does not exercise the native OnlineTrainer,
distributed execution, production checkpoint loading, LoRA placement or an
external reward service. In particular, it must not be labeled official VDN
GRPO acceptance or semantic learning improvement.

Validation from the locked review venv: 69 tests passed across the VDN family,
scheduler sample/replay parity and continuous GRPO suites. Scoped Ruff passed.
The existing decord 0.6.0 wheel metadata mismatch still makes uv dry-run propose
the same-version reinstall after frozen sync; no lock or vendor patches hide it.
All work in this continuation was CPU-only. GPU inventory was empty at entry.

## VDN OnlineTrainer and CUDA gate completed

`tests/trainers/online/test_vdn_grpo_composition.py` now passes on CPU and one
L40S with real tiny VDN computation. The collector lifecycle and [0, 1] rewards
are controlled doubles, while the batch executor, training-side replay model,
DiffusionSDELogProbEvaluator, GRPO, OnlineTrainer.step, AdamW, EMA and checkpoint
save/load/restore APIs are production implementations.

It executes two distinct samples per update, three generation steps, replay
microbatch width one and two updates. A separate rollout model receives CPU
weight snapshots before subsequent sampling. Both updates have positive
gradient norm and replay error below 1e-6. The runtime's recorded first-step
CUDA replay error is exactly zero at its unchanged 0.01 gate.

After checkpoint-1, an independently built trainer restores model, optimizer,
EMA and RNG, then executes update two. The sampled actions, all model tensors,
Adam state and EMA state exactly equal the uninterrupted update; gradient norms
also match. This trains the hybrid output projection (256 FP32 parameters),
not the released LoRA recipe or the whole model.

The initial CUDA fixture correctly failed the framework's CPU-snapshot guard:
the test getter returned CUDA tensors. Fixing the test getter to export detached
CPU copies resolved it; no production guard, math or threshold was changed.

Final results:

- Related CPU regression: 113 passed, three GPU deselections, 4.98s.
- CUDA gate: one passed, one CPU deselection, 8.96s in the persisted run.
- Scoped Ruff check/format and git diff checks passed.
- GPU compute inventory is empty after the test exits.

Persisted artifacts under `/mnt/nvme/outputs/wan22_i2v_cache`:
`vdn_grpo_cuda_20260914.xml` and `vdn_grpo_cuda_20260914/`, including the real
checkpoint-1 and control/resume replay debug receipts.

## Acceptance boundary

The clarified short correctness objective has evidence at complementary levels:
real-weight Wan native updates plus single/multi-GPU and resume agreement;
VDN reward-signed gradient/optimizer checks; and VDN CPU/CUDA OnlineTrainer
composition with strict state restoration. These support the tested video RL
framework contracts, not universal correctness for every future configuration.

Released-weight VDN loading, its LoRA/device-partitioned recipe, multi-GPU VDN,
full-resolution throughput and held-out semantic quality remain unverified.
They are distinct deployment/quality experiments, not claims made by these
tests. Do not restart the cancelled full-size Wan queue merely to answer
whether the tested small-video RL framework works.
