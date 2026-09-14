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

## Remaining VDN gate

Use the existing video GRPO contracts to add bounded native trainer and GPU
coverage, including actual trainable placement, synchronization and checkpoint
resume. Keep random-weight architectural tests explicitly separate from
released-weight loading and quality. Do not restart the cancelled full-size
Wan queue merely to answer whether the tested small-video RL framework works.
