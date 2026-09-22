# Visual RL engine: implementation and evidence ledger

## Goal and time boundary

- Started: 2026-09-22 07:36:41 UTC.
- User requested at least 24 hours of work: do not claim this duration satisfied before
  2026-09-23 07:36:41 UTC. Elapsed time alone does not establish completion.
- Target: independent reward reliability tooling, rollout admission, real Qwen-Image-2.1
  editing, and bounded agentic visual RL with explicit credit assignment.
- Distinguish CPU/fake tests, real inference, actual policy training, and independently
  measured capability improvement. No deployment, push, paid API/cloud allocation, or
  large model download is authorized.

## Initial authoritative state

- Workspace: `/home/mingfeiguo/Desktop/vrl2/VRL`.
- Existing modifications: reward and admission sprint documents; untracked NGU sprint
  and `third_party/PhyMotion/`. Preserve these; do not sweep into implementation commits.
- Local GPU: RTX 5090, 32 GB. At startup another process owns approximately 16.8 GB and
  is actively evaluating a Qwen-Image-2.1 checkpoint. Do not interrupt it.
- Observed process: PID 4175992, working directory `/home/mingfeiguo/Desktop/VRL`,
  command `vrl.scripts.eval.reference_image_checkpoint_eval`, checkpoint
  `outputs/qwen_image_21_edit_rl/main/checkpoint-final`, label `rl20`.
- That checkout has `qwen_image_21` family/editing documentation and commit
  `90852130` (reference-conditioned EditReward training). Treat it as a read-only
  integration reference; inspect dependencies and actual diffs before porting.
- Model config there names `Qwen/Qwen-Image-2.1`; resolve the cached artifact path before
  any model launch. Existing documentation does not prove this workspace supports it.

## Work completed / underway

1. Synchronized current workspace environment from its frozen lock with test/lint,
   cosmos, reward and reward-service extras using `--inexact` to retain existing
   optional installations. Did not modify the active external Qwen environment.
2. Implemented propagation of nested reward observations using `parent/axis` keys,
   preserving weighted training totals and zero-weight audit components. Reject
   namespace collisions instead of overwriting measurements. Ruff checks passed;
   `tests/rewards/functions/test_multi.py`: 42 passed (CPU, 0.37 seconds).

## Next implementation steps

1. Finish component propagation checks and make an isolated code commit.
2. Implement independent media-manifest scoring using existing reward runtime/service;
   persist raw axes and immutable scoring provenance before report/calibration layers.
3. Reuse existing score-report statistics; validate cached rescoring and comparisons.
4. Inspect Qwen family integration and dependency compatibility, then port bounded
   reference editing support without copying the other checkout's unrelated changes.
5. Add bounded episode/tool execution and training ownership/credit assignment with
   explicit controller/generator semantics; validate with real model when available.
6. Continue rollout admission work after preserving distributed advantage semantics.

## Evidence not yet obtained

- Real Qwen inference and a one-step LoRA gradient probe have now run (below).
  Full reward-weighted policy training and capability gains remain unverified.
- No preference labels or capability improvement have been manufactured or claimed.
- No 24-hour completion, production readiness, or working agent policy RL is established.

## Independent scoring implementation

- Added `vrl.rewards.evaluation` and CLI
  `python -m vrl.scripts.rewards.score_manifest --help`.
- Supports local and existing HTTP scoring, all raw score axes, prompt/media/
  auxiliary-file fingerprints, atomic per-sample evidence, exclusive writers,
  explicit failed-batch records, and validated resume without model loading.
- Fixed another diagnostic loss in `InferenceRewardFunction`: retain model
  axes and reject inconsistent axes across one batch.
- Targeted initial suite: 58 passed, including actual CPU Laplacian scoring and
  real localhost HTTP upload (no learned model or CUDA).
- Real cached Qwen outputs: scored two `armchair_seat_blue` rl20 images using CPU
  sharpness only. Artifact location: `outputs/reward_evaluation/qwen21_smoke/`.
  First run scored 2; resume scored 0 and reused 2. Run fingerprint:
  `b9650742846f065c40682d62bad170a17a96b7aafc8bde6e31f33a2712a877d3`.
  This proves rescoring works on real generated media, not editing improvement.
- Initial smoke manifest incorrectly resolved a source image beside the old
  manifest; it failed before scoring. Corrected to the actual artifact-data-root
  reference file and reran successfully. No source checkout files were changed.
- Original Qwen evaluation process PID 4175992 has exited. Subsequent GPU query
  showed 139 MiB used / 0% utilization; recheck before any model launch.
- Qwen source integration currently requires pinned Diffusers git revision
  `80c7ed262aeffbeb43ef13ae04baeb9b84515a69` and Transformers >=5.17;
  current workspace uses Diffusers 0.40 / Transformers 5.13. Do not copy its
  environment or family without compatibility tests.
- Expanded verification: `pytest tests/rewards/functions tests/rewards/test_evaluation.py -q`
  completed with 102 passed, 2 skipped. Changed-file Ruff and `git diff --check` passed.

## Qwen Image 2.1 integration

- Ported only family adapters, reference-input plumbing, RGBA media boundaries,
  and relevant tests from the read-only sibling checkout (integration 80b012dd,
  source HEAD 90852130). Registered the family using the existing descriptor.
- Updated and froze the local dependency lock: Diffusers git
  `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`, Transformers 5.17.0.
- Real local checkpoint revision: `b3179ad355be050328e483a9dfdd9e60cd62adfa`.
  All model probes used HF_HUB_OFFLINE=1; no weights were downloaded.
- RTX 5090: single-reference edit, two-reference edit, and RGBA extraction at
  512x512 / 20 native Euler steps succeeded. Evidence:
  `outputs/qwen_image_21/port_parity_512/report.json` and adjacent PNGs.
- Compared against the upstream pipeline with shared initial latent, disabled
  prefix caching, CPU text encoding, and matched VAE preprocessing precision.
  All three RGBA outputs had byte MAE 0 and first-action replay max error 0.
  This does not assert equivalence to every default upstream configuration.
- Separate 256x256 / 2-step SDE LoRA probe: log-prob replay error 0,
  gradient norm 0.2742737933, optimizer parameter max delta 9.6713857e-06.
  The loss is a log-prob gradient probe, NOT reward-weighted RL or a capability gain.
- Fixed an additional transport inconsistency: RGBA tensors in video layouts
  now composite over white like still-image reward views, instead of dropping alpha.
  PNG exports retain alpha. Alpha-aware metrics must consume the preserved artifact.
- Validation after dependency change: family/model/denoise/config/binding/media/
  reward/data suites: **1489 passed, 25 skipped** (94.06s). Includes CPU real tiny
  transformers and localhost/Ray reward transport tests, not GPU training tests.
- Architecture: retain family runtime.py as a real executor adapter; retain
  `_LATENT_TOKENS_PER_MASK_SLOT` as model geometry and tiny-model dimensions as
  fixture constants. Preserve the family-wide encode/prepare/forward/decode shape;
  reducing line count or reorganizing unrelated families is not a goal.

## Reward diagnostics and frozen calibration

- Added independent health/ranking CLI (`analyze_scores`) and preference-combination
  CLI (`calibrate_scores`). Neither loads a model or starts a trainer.
- Reports validate stored provenance, preserve errors/missing axes, expose group
  ranges and timings, and compare only identical complete media grids. Ranking
  agreement is explicitly not labeled human accuracy.
- Linear logistic calibration freezes score scales/weights on calibration only;
  source groups receive equal weight. Explicit ties and unsure annotations remain
  visible. Holdout validates scorer recipe and rejects known prompt/source/media/
  auxiliary-asset leakage. It does not automatically change training rewards.
- Reused the existing deterministic bootstrap implementation; confidence intervals
  use prompt/source units, never the number of correlated seeds/annotations.
- Tests: 11 passed across scoring, diagnostics and calibration. Synthetic judgments
  test fitting, holdout-label invariance, tie handling, leakage and nonconvergence;
  they are not supplied as real preference evidence.
- Ran the health CLI against real cached Qwen output scores at
  `outputs/reward_evaluation/qwen21_smoke/health.json`. Two images from one prompt
  are insufficient for population conclusions; no confidence interval is reported.
- Still needed: independently labeled real evaluation set, candidate-model scoring,
  frozen-combination training integration, and reward-weighted online experiments.
