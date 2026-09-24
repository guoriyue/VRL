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
  `python -m vrl.scripts.rewards.rescore_media --help`.
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

## First real reward-weighted online Qwen run

- Ported the reference-conditioned EditReward adapter and explicit HTTP parking
  lease exception from source 90852130. The service advertises memory_parking=true
  and generation_overlap_safe=false; no local reward GPU reservation is fabricated.
- Ran our adapter in the existing isolated `/tmp/vrl-edit-reward-env` environment
  (upstream EditReward revision 77a93aaa461fe9187e0ff841b59ecc0d0620bb7f).
  Service port 18315, process created by this goal; no existing service interrupted.
- Attempt 1 failed config resolution because an external reward owned no local GPU.
  Fixed the explicit lease case; resource/reward/service tests: 162 passed.
- Attempt 2 failed because offline Hub revision lookup was attempted. Attempt 3
  used the absolute cached Qwen snapshot. No weights downloaded or guard relaxed.
- Successful run: `outputs/qwen_image_21/edit_rl_smoke_attempt3`;
  final checkpoint global_step=2, completed_epoch=2. LoRA rank 4/alpha 8,
  2 prompts x 2 samples, 256x256, 4 denoise steps, float32 stored trajectories.
- Pre-update max log-prob difference: 1.1920928955078125e-07 in both updates.
  Gradient norms: 0.0379986912 and 0.0300357677. No zero-advantage groups.
- Reward means: -1.9595036507, -1.9664444923. These two training observations
  DO NOT establish improvement (the second is lower). No heldout gain claimed.
- Final checkpoint tree digest: c9e4b0c2c064825cee681533671f1f87e2c812203babd60b49e5ec0081015ab1.
  Launch ID: 37a854cc6ec647249b32e5a3d7a4ef85; runtime captured artifact evidence.
- Found remaining diagnostics gap: fixed-schema CSV exports configured top-level
  reward names only. Nested axes survive collection but need a complete observation
  sidecar before claiming end-to-end training diagnostics.

## Checkpoint resume and complete training observations

- Added `reward_components.jsonl` beside the stable CSV views. It retains every
  nested axis at full JSON float precision without changing legacy CSV headers.
  Resume discards observations at/after the checkpoint's next epoch and incomplete
  final appends. Nonfinite axes fail before any metrics sink is appended.
- Metrics/resume tests: 51 passed. Real resume from the two-update checkpoint into
  `outputs/qwen_image_21/edit_rl_resume_step3` completed global_step=3, epoch=3.
- The resumed step's observation record includes `editreward/editreward_log_sigma`
  (-1.4814392626) alongside total EditReward (-1.9464911819), proving the new axis
  propagation survives the real service -> composite -> collector -> trainer -> IO.
- This proves resume execution and artifact continuity, not bitwise equivalence to
  an uninterrupted three-update run; that comparison remains to be executed.

## Terminal reward parking failures

- Fixed the original unsafe reward-side CuMem retry contract: failed sleep, wake,
  or pool close quarantines the parking owner. Later park/activate/score/shutdown
  refuse allocator access and require process termination. Successful park followed
  by a cache-release error remains distinguishable from a broken CuMem operation.
- Reward inference/service/Ray regression: 187 passed, 4 skipped. Added explicit
  partial-wake regression afterward: in-process suite 17 passed, 4 skipped.
- EditReward experiments use reload parking; their active service was not restarted
  mid-run. The changed CuMem behavior is covered by fault-injection tests, not a
  deliberately corrupted live GPU allocator.

## Resume reproducibility qualification (in progress)

- A separately initialized uninterrupted three-update run completed. Comparing it
  against the earlier resumed run is NOT a controlled resume equivalence test:
  their first-update gradient norms already differ (0.03799869 vs 0.03803318).
  Subsequent trajectories diverge. Max final LoRA difference 0.0005340081 is recorded
  at `outputs/qwen_image_21/resume_comparison.json`, without blaming resume.
- Next comparison resumes the uninterrupted run's OWN checkpoint-2 and compares
  its next update with that same run's step 3. This isolates checkpoint continuation
  from different prefixes. Any remaining non-bitwise result needs qualification.

## Bounded visual episode seam

- Added `vrl.rollouts.episodes` above existing one-call generation/reward semantics.
  Typed task/action/artifact/observation/decision/tool records carry separate policy
  stamps and observation digests. Traces persist original/current media identities,
  decisions, tool costs, final reward and finite-horizon return-to-go.
- The runner acknowledges sequential initial/operation parking; failed handoff
  aborts before another owner activates. An action sees the newly produced image.
  Stop has no fabricated generator transition; budget exhaustion does not invent a
  controller stop likelihood. Failed tools publish error traces without returns.
- CPU fake-role tests: 3 passed, covering image-dependent stop, cost/terminal credit,
  exhausted budget, stale versions, failed parking and failed editing. These tests
  do not establish a learned agent or real multi-step Qwen operation.
- New implementation plan: `docs/sprints/planned/SPRINT_qwen_visual_episodes.md`.
  It supersedes the deleted Janus starting point without restoring token-AR families.
- Next: concrete runtime adapters and a real image-conditioned, trainable controller;
  use the existing clipped objective and preserve separate generator credit.

## Real categorical visual controller

- Added a separate Qwen3-VL-compatible controller with explicit edit/stop labels,
  seeded categorical sampling, tensor-backed hashed replay, observation/policy
  binding, and differentiable current-policy likelihoods. It does not share the
  generator's text encoder. Temperature belongs to the replay contract.
- Tiny real Qwen3-VL forward/backward test verifies visual gradients, likelihood
  parity, an improving selected-action update, and rejection of changed context
  or corrupted tensors. Controller and episode suites: 4 passed.
- Local full checkpoint probe uses Qwen 2.1's cached Qwen3-VL weights, separately
  loaded with 1,916,928 trainable LoRA parameters. Initial run exposed an ignored
  max_pixels processor argument and ran out of GPU memory during replay. Replaced
  it with the current processor's explicit size configuration.
- Successful real probe: `outputs/qwen_image_21/controller_probe_temperature2`.
  Likelihood replay error 0; gradient norm 0.06159721; maximum parameter change
  9.99998e-6. BF16 post-update selected likelihood did not visibly change on this
  single update. This is a gradient probe, not reward-driven controller training
  and not evidence of learned stopping or improved editing.
- Matched-prefix generator resume: reward scores match sample by sample, but
  gradient norms differ: uninterrupted 0.06085065, resumed 0.05996274, second
  identical resume 0.06075246. Therefore non-bitwise behavior is reproducible even
  between two resumes; deterministic execution still needs investigation.

## Deterministic continuation comparison

- Repeated that same checkpoint-2 continuation twice with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and trainer-process
  `torch.use_deterministic_algorithms(True)`. Both completed the real rollout,
  EditReward scoring, backward, optimizer step and checkpoint publication.
- All 256 final LoRA tensors are bitwise equal; max absolute delta 0. Gradient
  norm matches exactly at 0.06049364060163498. Evidence:
  `outputs/qwen_image_21/deterministic_resume_comparison.json` and the two
  `edit_rl_matched_resume3_deterministic*` run directories.
- Scope: two restored continuations on this hardware and software configuration.
  This supports execution nondeterminism as the earlier difference's source. It
  is not a claim of cross-hardware determinism or a full uninterrupted-vs-resumed
  equivalence proof. The Ray generation worker retained its original settings.

## Real bounded visual episode

- `vrl.scripts.generation.visual_episode` connects the independent visual controller,
  existing Qwen batch executor and HTTP EditReward runtime. It preflights and parks
  the external judge before allocating the editor. Native editing uses current
  media; scoring uses original instruction/reference. Each intermediate PNG has a
  distinct path so later edits cannot overwrite earlier replay evidence.
- Real run: `outputs/qwen_image_21/visual_episode_armchair_attempt2`, 256 pixels,
  8 native steps, two edit calls, temperature 2, per-call cost 0.01. Successful
  termination was tool budget exhaustion, NOT a learned stop. Initial score
  -0.43923175, first-edit 0.50081563, second-edit 0.48951447; return 0.46951447.
- Visual inspection reveals an actual reward blind spot: the requested chair
  became blue, but the neighboring brown magazine holder also became blue.
  Therefore the high score does not establish locality preservation. Repeated
  editing did not improve this sample. Preserve it for independent locality
  diagnostics and stopping-policy evaluation, not as proof of capability gain.
- First launch failed before model loading because a model-reward constructor
  cannot directly consume HTTP deployment config; corrected the adapter wiring
  to inject the existing HttpRewardScorer, as the registry does.
- Controller/episode/adapter tests: 5 passed. Source/current reward boundaries,
  alpha compositing, distinct intermediate paths and terminal parking failure are
  covered. These adapters are genuine runtime boundaries; no serving scheduler
  or generator policy-gradient interpretation was added.

## Exact-parity preset consistency

- Changed the four-L40S Wan HPSv3 preset's trajectory storage from BF16 to
  float32, matching its existing zero-tolerance replay gate and the reference
  recipe. BF16 model compute remains supported; lossy storage of fp32 sampled
  latents was the inconsistency. No unmeasured tolerance was introduced.
- This removes one known drift source, not every possible compiled/eager numeric
  difference. Host trajectory memory increases and must be checked on the target
  four-GPU host. No four-L40S experiment was available here.
- Existing full experiment compilation suite: 36 passed.
- Existing `trainer.deterministic=true` exposes the trainer determinism settings
  used by the continuation investigation; no new determinism option is needed.

## Real controller RL and recovery

- Added strictly on-policy controller updates using the existing GRPO clipped
  surrogate. Each decision gets future return-to-go and its time discount;
  other same-task episodes supply a leave-one-out baseline. Sum decisions within
  each episode, average episodes. Tool policy remains frozen and separate.
- Successful traces are validated for source/current lineage, policy versions,
  budget, stop semantics, reward arithmetic and returns before training. Likelihood
  replay is gated before optimizer mutation. Optimizer state parks with the policy.
- Checkpoints atomically save trainable parameters, Adam state, RNG, cursor,
  version and exact task/reward/sampling/optimizer contract. Ordered parameter
  names protect optimizer state mapping. CPU tests verify a hand-derived variable-
  length gradient and exact Adam/RNG continuation, plus stale/changed-data rejection.
- First real training attempt collected two full episodes and updated, then failed
  saving metrics because the metrics directory was missing. Fixed initialization.
  Do not count that unsaved update as a completed run.
- Recovered the same two real episodes from the saved initial checkpoint, without
  regeneration: `outputs/qwen_image_21/controller_rl_recovered1/checkpoint-1.pt`.
  Four decisions, gradient norm 0.01269512, replay error 0, controller version 1.
  Recovery script and source trace hashes are preserved in outputs.
- CLI resume into `outputs/qwen_image_21/controller_rl_resume2` completed the second
  update: 2 episodes / 3 decisions, gradient norm 3.57558823 (clipped to configured
  limit), replay error 0, controller version 2. Mean return 0.07835855, mean tool
  calls 1, stop fraction 0.5. First group's returns were 0.28951447 and 0.52821462.
  This does NOT demonstrate improvement; the second group has lower mean return.
- Runtime settings: 256-pixel, 8-step frozen Qwen editor; temperature 4; two tool
  calls maximum; cost 0.1; one training source. No held-out claim. Next acceptance
  work is independent reward/locality checks and matched-budget evaluation.

## Post-advantage admission audit

- Moved the existing exact-zero sample selection into `vrl.rollouts.admission`.
  It consumes already-normalized advantages; global/group statistics, component
  objectives, streaming preflight and training denominators remain unchanged.
  It does not replace nonzero advantage with raw reward variance or binary pass rate.
- Per-attempt, per-rank JSONL retains request/sample IDs, prompt identity,
  synchronization policy version, full reward axes, actual advantages and selected
  rows. Keep/drop/partial decisions explicitly leave failure cause undetermined.
  Selection is not mislabeled as an applied optimizer update. New attempts never
  rewrite previous audit records. Rank-local audit failures stop peers before backward.
- Tests cover tiny nonzero advantages, continuous negative rewards, partial group
  retention, disabled filtering, immutable attempt files and local/peer audit
  failure. Two legacy fakes were corrected to supply their declared contracts
  (string sample prompt and strategy rank). Online trainer/collector/admission
  regression: 253 passed, 3 skipped, including streaming/FSDP equivalence tests.
- Real deterministic continuation to Qwen generator step 4 completed:
  `outputs/qwen_image_21/edit_rl_admission_step4`. Two groups/four samples kept;
  complete axes and sample identities are recorded under `admission/`.
  Reward -1.93346810, gradient norm 0.03421023, replay max error 2.3841858e-7.
  No quality or filtering-benefit claim follows from this smoke update.

## Independent masked-edit diagnostics and corrective edit

- Added CPU `masked_edit` through the existing reward adapter/transport. It reports
  protected-region RGB errors and editable-region change separately, requires an
  explicit binary mask on the same canvas, and supports zero-weight observations.
  It is not a semantic verifier: returning the source unchanged maximizes locality.
- The thin reward class stays as the standard cross-family transport adapter;
  pixel logic stays in the model. No new workflow vocabulary/configuration table
  or parallel reward lifecycle was introduced.
- Targeted reward/inference/service regression: 221 passed, 4 skipped. Tests include
  a valid local edit, unchanged-source exploit, outside-mask edit and invalid masks.
- Real audit: `outputs/reward_evaluation/masked_edit_audit`. Four candidates share
  an explicitly resized 256-pixel source and a coarse protected magazine-holder
  rectangle; this is not a full background-preservation mask. Independent scoring
  recipes and all input/asset digests are retained. No human labels were invented.
- EditReward / protected-region RGB L1: unchanged -0.362050 / 0; first blue edit
  0.500816 / 0.197280; repeated edit 0.489514 / 0.207311; corrective edit
  0.175492 / 0.176056. The unchanged score differs from the earlier episode because
  this audit uses the explicit common-canvas source preprocessing.
- Corrective 20-step Qwen edit: `outputs/qwen_image_21/visual_repair_magazine`.
  Visual inspection finds the holder's outer panel brown again but its interior
  still blue; chair remains blue. This is partial repair following an explicit
  instruction, not learned agent improvement. The score disagreement is evidence
  for separate diagnostics, not proof that either score measures overall truth.

## Cross-scorer calibration inputs

- Added a strict in-memory join for independently persisted scoring runs. It
  requires identical complete sample/input grids, scopes every axis by an explicit
  scorer alias, and retains each original result/version/timing. No row intersection,
  imputation or model inference occurs. Source runs and observed results bind the
  derived view ID; the frozen scorer recipe remains reusable on disjoint holdout.
- Calibration CLI accepts repeated `--component NAME=DIRECTORY` as an alternative
  to one `--evaluation`. This uses existing analysis and calibration modules rather
  than adding a parallel reward service or a thin re-export module.
- Tests exercise source-disjoint holdout fitting, changed verifier rejection,
  missing/changed assets, missing samples, invalid aliases and result identity.
  Evaluation/calibration/diagnostics suite: 12 passed.
- Real four-candidate join is saved in
  `outputs/reward_evaluation/masked_edit_audit/joined_view.json`, with nine axes and
  a separate health report. No preference fit was claimed: real human labels are
  not yet available and one source is insufficient for the fitting contract.

## Checkpoint contract serialization fix

- Held-out evaluation exposed a restore false negative: training task actions are
  tuples in the Python checkpoint and lists in its JSON contract manifest. Compare
  the existing canonical JSON fingerprints so equivalent representations restore;
  changed values/order still fail. Model, precision, temperature, image budget,
  optimizer and checkpoint-file integrity checks remain enforced.
- The checkpoint credit/RNG/Adam continuation test now restores a JSON-style action
  list from a tuple contract. Controller replay/comparison/credit tests: 3 passed.
- First held-out evaluation attempt stopped before scoring; it is not a completed
  capability result. A new output directory preserves that failed attempt.

## Held-out bounded controller comparison

- Added an evaluation-only baseline controller and comparison runner over the
  existing episode protocol: immediate stop, first-edit-then-stop, seeded uniform
  actions, real controller, and independent-source best-of-N. Best-of-N pays for
  discarded samples and preserves their traces; it never fabricates a trainable
  selected episode. Role handoff failure stops the comparison with a partial audit.
- Added `vrl.scripts.eval.visual_control`: verified local controller restore,
  exact-source training overlap rejection before runtime startup, source-balanced
  descriptive means, cost/decision/judge/latency accounting, and a selected-media
  manifest for independent rescoring. Cross-source near duplicates remain a
  curation responsibility. Same editor-call ceiling does not imply equal FLOPs.
- Protocol baseline class is a necessary implementation boundary. CLI wiring stays
  separate from reusable scheduling; no duplicated model/reward lifecycle was added.
  Existing editor/controller family and checkpoint structures stay unchanged.
- Seven comparison, episode, replay and controller-credit tests passed. The new
  checks cover discarded-candidate costs, source restarts, failed handoffs and
  leakage rejection before model loading.
- Real held-out sweater experiment completed at
  `outputs/qwen_image_21/controller_comparison_sweater_attempt2`: 256 square,
  20 editor steps, two-call maximum, cost 0.1, trained controller version 2.
  Net returns: stop -0.291445; fixed one edit 2.195264; random 2.195264;
  controller two edits 1.842208; best-of-2 2.171453. One source/seed only.
- This is a negative controller-benefit result. Visual review of the fixed edit
  finds blue fabric but damaged lettering and changed framing. The square output
  changes the portrait geometry; follow-up should preserve aspect ratio and use
  independent protected-region/text checks. High EditReward does not establish
  successful instruction following.

## Explicit visual editor canvas

- Collection, training and evaluation share explicit `--width`/`--height` options.
  Validate the pair and Qwen's 32-pixel geometry before model allocation. The shared
  sampling function is the configuration boundary, replacing duplicated literal
  sampling dictionaries; existing square checkpoint contracts remain compatible.
  Explicit training geometry enters the resume contract.
- Family/adapter/controller regression: 17 passed. A real 384x576, 20-step portrait
  comparison completed in `outputs/qwen_image_21/controller_comparison_sweater_portrait`.
  Fixed one-edit net return 2.327336; controller two-edit return 1.471589;
  best-of-2 return 2.227336. Same source/seed as the square diagnostic, not another
  independent evaluation source. The controller still does not beat the baseline.
- Visual inspection finds blue fabric and improved portrait framing, but lettering
  remains altered. Geometry correction is not evidence that semantic preservation
  is solved; this high reward remains an independent-check counterexample.

## Text-only evaluator conditioning guard

- Auditing RGBA evaluation found that the existing text-only checkpoint evaluator
  rejected single image/video conditioning but omitted the newly supported
  `reference_images` list. It now rejects that list as well, preventing silent
  evaluation without required references. Reference-conditioned evaluation remains
  on the family executor probe/visual-session path until explicitly supported here.
- Extended the existing conditioning guard test. Image evaluator plus RGBA reward
  tests: 23 passed. No generation behavior changed for supported text-only inputs.

## Executable RGBA reference objective and real RL

- Added `rgba_reference` through the existing reward adapter and CPU model path.
  It retains four channels and separately measures soft alpha IoU, premultiplied
  color, black/white composites and leaked background alpha. Exact-reference score
  is the minimum of alpha IoU and bounded foreground-normalized color accuracy.
  It is a specified pixel-matching objective, not a general perceptual oracle.
- Tests cover soft-edge oracle, arbitrary hidden transparent RGB, empty alpha,
  all-opaque output, wrong colors, flattened RGB, bad canvas, nonfinite input and
  degenerate oracle. Native tensors and files retain alpha. Reward adapters remain
  uniform transport boundaries; target_image reuses existing reward-only artifact
  ownership. No new agent environment or task taxonomy is introduced.
- Added a deterministic synthetic circle/square extraction builder and composable
  dataset/reward/two-update Qwen recipe. 200 generated scenes split 160/40;
  all 200 exact oracles scored 1 through standalone persisted scoring.
- Extended the existing Qwen family probe to accept prompt manifests, retain
  reference/manifest hashes and refuse output-directory reuse. Four 20-step native
  outputs scored 0.8783–0.9374. Forty held-out 8-step outputs averaged 0.920335
  (scene-bootstrap 95% interval 0.915211–0.924799), with native first-action replay
  error zero for every output. This is a narrow synthetic baseline, not a learned gain.
- First training launch used an inadmissible in-process online reward and failed
  before updates. Recipe now uses the framework's CPU Ray reward actor, preserving
  alpha through the real rollout/reward path. Two real updates completed in
  `outputs/qwen_image_21/rgba_reference_smoke_attempt2`, final global step 2.
  Training reward means 0.7105/0.6958 and gradient norms 0.010198/0.016548; all
  reward axes and admission decisions persisted. This stochastic SDE training
  distribution differs from the native baseline and is not a paired comparison.
- Config/reward composite regression: 80 passed. Image evaluator/RGBA regression:
  23 passed. Native baseline and actual optimizer evidence are recorded separately
  from executable fixture tests. Documentation: `docs/rgba_reference_tasks.md`.

## Earlier online reward deployment admission

- The failed RGBA launch showed that the existing online in-process-reward
  rejection happened after replay-model allocation. Moved the shared admission
  check to `RewardRuntimeConfig.require_online_training`, called by online run
  resolution before resource setup and retained at the direct factory boundary.
  The same method checks that a positive training reward exists. Offline
  in-process scoring and zero-weight auditing remain valid configuration uses.
- This is an explicit runtime-config admission API, removing duplicated policy
  from construction rather than adding a wrapper module. Test verifies rejection
  before resource resolution. Factory/builders/presets/RNG tests: 85 passed,
  1 skipped.
- Full-precision RGBA smoke replay maxima were 1.1920929e-7 and 2.3841858e-7;
  both satisfy the unchanged 1e-5 gate. Rounded CSV zeros are not exact-zero claims.

## Paired RGBA checkpoint evaluation

- The Qwen executor probe now restores owned VRL checkpoints through the existing
  strict model-identity/restore APIs. Adapter topology comes from the recorded
  identity; actual local base files are resolved and reverified during loading.
  Report includes model identity and checkpoint digest. Evaluation mode is explicit;
  checkpoint evaluation cannot also run the fresh-adapter mutation probe.
- Checkpoint/family/reference regression: 30 passed. Real step-2 checkpoint restored
  and generated all 40 prior evaluation scenes with the same 8-step native settings
  and seed; every first-action replay error was zero.
- Standalone scores: base mean 0.9203345891; step-2 mean 0.9155270002. Paired mean
  delta -0.0048075889, scene-bootstrap interval [-0.0097811524, -0.0012331656];
  6 increases / 34 decreases. See `outputs/qwen_image_21/rgba_step2_paired.json`
  and its reproducible `compare_rgba_checkpoints.py` script. This is a negative
  task-matching result, not evidence of improved transparency or human quality.
- These 40 scenes are now a monitoring set. A fixed continuation to 16 total
  updates will test longer-run/resume behavior without changing the optimization
  recipe; a separately generated, unviewed source seed will be used for a final
  comparison. No adaptive hyperparameter search or positive-result promise.

## Independent analysis dependency direction

- Moved the unchanged pure distribution/bootstrap functions into
  `vrl.utils.score_statistics`. Reward diagnostics/calibration now import that
  library directly instead of importing a CLI implementation. The score-report
  CLI retains its public imports and behavior for existing callers.
- This small shared module is justified by the library/CLI ownership boundary;
  no algorithms, intervals, report schemas, registry tables or family adapters
  were changed. Existing report/calibration/diagnostic/join tests: 15 passed.
- Reserved final synthetic source seed 20260922 under
  `outputs/datasets/rgba_extraction_final_seed20260922` (160 unused training rows,
  40 final evaluation rows). No model output on these scenes has yet been
  generated or inspected. Fixed 16-update continuation is in progress.

## Fixed 16-update RGBA continuation and fresh-source evaluation

- Resume from step 2 completed at step 16 in
  `outputs/qwen_image_21/rgba_reference_continue16`. The 14 continuation updates
  retained the original recipe; maximum replay log-prob difference 2.3841858e-7.
  Final training reward 0.81484264, gradient norm 0.01571325, KL penalty 0.00181193.
  These are stochastic training observations, not a held-out improvement claim.
- Both base and step 16 generated the reserved seed-20260922 evaluation scenes
  with identical native 8-step sampling, seed 42 and canvas. All 40 first-action
  replay checks per arm were exact. Source hashes were disjoint from the entire
  earlier 200-scene dataset. Strict checkpoint source identity was reverified.
- Fresh-source means: base 0.9138962271; step 16 0.9062281512. Paired difference
  -0.0076680760, bootstrap interval [-0.0131391838, 0.0002962993]; 3 scenes improved,
  37 declined. Median paired difference -0.00848290. One baseline failure improved
  substantially, so the mean interval includes zero; this is not evidence of an
  average gain. Color error and leaked alpha also increased on average.
- Results: `outputs/qwen_image_21/rgba_final_step16_paired.json`; raw per-axis
  scoring under `outputs/reward_evaluation/rgba_final_*`. The predeclared continuation
  failed its improvement objective. Keep the base as the better measured native
  extraction baseline. Any further use of these scenes is diagnostic/exploratory,
  not a fresh final test. Next diagnosis may compare the stochastic training sampler
  separately; it must not retroactively replace this native-deployment result.

## Structured verifier evidence across the reward boundary

- Added finite JSON diagnostics alongside numeric `RewardInferenceResult.scores`.
  Runtime accepts structured model results, checks artifact/version ownership and
  retains runtime-measured timing. Existing numeric plugins remain supported.
  Evidence is detached at construction, survives HTTP/offline/debug records, and
  never becomes a loss component. Health reports expose evidence/why counts;
  independent-scorer joins retain each scorer's diagnostics.
- GenEval exports why/spec from one detector pass through `score_results`; its
  numeric score_batch facade remains for compatibility. The bundled recipe enables
  a per-sample debug sidecar. These reasons are detector judgments, not ground truth.
- HTTP wire version is now 6; incompatible peers fail at preflight instead of
  silently discarding the new field. This protocol constant is a real compatibility
  boundary. The detailed hook/numeric facade is justified by the existing plugin
  API, not by a desire for more wrapper files. No scalar objective changed.
- Inference/service/GenEval/analysis/preset regression: 251 passed, 4 skipped.
  Added real local HTTP round-trip with a fake detector, offline reason persistence,
  single-inference evidence, JSON validation, artifact/revision mismatch rejection
  and runtime timing ownership. This is transport evidence, not model accuracy.
- Gracefully restarted only the owned EditReward server (old PID 7990; new PID
  123746, session 17524) on wire 6, same model revisions. Four real prior candidates
  rescored through HTTP with maximum score-axis delta exactly zero. Then explicitly
  parked the service before the next GPU handoff. Artifacts:
  `outputs/reward_evaluation/masked_edit_audit/editreward_wire6_recheck`.

### Sharpness shortcut and invalid configuration guard

- A real CPU image test confirms that random high-frequency noise saturates the
  sharpness score, while blurring an actual edge reduces its score. Pairing this
  axis with a learned reward is not a proof that shortcuts are rejected.
- Fixed a concrete nonfinite-scale bug: NaN passed the old positivity check and
  Python's `min(1, NaN)` returned a perfect score. NaN/infinity now fail at setup.
- Removed the unsupported claim that PickScore guarantees rejection of noise.
  The transport adapter, score definition and cross-family shapes stay unchanged;
  this is validation and corrected documentation, not a reward redesign.
- Seven standalone evaluation/shortcut tests passed.

### Sampling-distribution diagnosis after the negative native RGBA result

The fixed step-16 checkpoint was additionally evaluated with training-matched
Flow-GRPO SDE sampling (noise 0.7, eight steps, seed 42). This is an exploratory
follow-up on the same 40 final scenes, not a replacement for the primary native
result. The probe now supports explicit native/SDE selection and records the
appropriate first-transition replay metric; all 80 SDE replay errors were zero.

- Base SDE mean RGBA match: 0.6956822462.
- Step-16 SDE mean: 0.8132594400.
- Paired gain: 0.1175771938; descriptive 95% interval [0.09707117, 0.13982610].
- 39 scenes improved, one declined.
- Artifacts: `outputs/qwen_image_21/rgba_diagnostic_sde_{base40,step16_eval40}`,
  corresponding standalone reward evaluations, and
  `outputs/qwen_image_21/rgba_diagnostic_sde_step16_paired.json`.

This supports learning under the trained stochastic sampler. Native base quality
(~0.914) remains higher than either stochastic arm, and the trained native model
still has the previously reported negative paired result. Do not deploy this as
an improvement to default inference. Next confirmation is frozen in advance:
new scene seed 20260923, the same base/step-16 pair and eight-step SDE recipe,
generation seeds 314159 and 271828; no additional training or coefficient tuning.

### Broad CPU regression and RGBA reward preflight

The frozen test environment ran the non-GPU, non-distributed, non-optional,
non-slow suite: 4,809 passed, 17 skipped, 155 deselected, one failure. The failure
was the existing preflight test's old exact component set: full subscore retention
now intentionally adds `image_sharpness/image_sharpness` alongside its aggregate.
The test now also checks equality of the aggregate and this single subscore.

Inspection uncovered a separate integration bug: reward preflight always produced
three-channel synthetic images even when sampling requests RGBA. It now respects
`sampling.output_mode`. A real local CPU Ray RGBA preflight over generated targets
passes without loading a generator. All ten preflight tests pass after the fix.
Full suite log: `/tmp/vrl2-production-cpu-suite.log`; focused log:
`/tmp/vrl2-preflight-fix.log`. This is CPU coverage, not GPU/distributed certification.

### Independent reward perturbation audits

Added `vrl.scripts.rewards.stress_media` and `analyze_scores stress`. The immutable
recipe preserves task/reference/target metadata, records content/seed provenance,
and produces baseline plus six explicit RGB/alpha probes. Reports compare only
same-source/task pairs, retain failed/missing pairs and report numerical direction
without inventing semantic labels. It is independent of training and agent control.

Real CPU audit: 200 synthetic RGBA oracles, 1,400 media files, independently scored
by exact RGBA reference and sharpness (2,800 observations). All six perturbations
lowered RGBA match on every source. Noise increased sharpness on all 200 sources
(mean +0.598), checkerboard +0.871. Even RGB blur increased sharpness here (+0.066):
blurring straight RGB while preserving alpha can create dark edge artifacts when
the RGB scorer composites over white. This is why perturbation names alone are
not universal quality labels. Alpha probes are also interpreted through each
scorer's actual preprocessing, not assumed to be ignored.

Artifacts: `outputs/reward_evaluation/rgba_oracle_stress200`, including recipe,
both scoring runs and paired reports. Six stress/diagnostics/calibration tests
passed; the real CLI report matches the programmatic audit. Existing trainer and
reward aggregation remain unchanged; no new business vocabulary table or thin
workflow wrappers were added.

Visual inspection of sampler comparisons: the first eight source-ordered scenes
show native extraction with clean circles, while both SDE arms retain visible
texture and edge noise. Better SDE exact-match scores are not a claim of native
visual quality. All 40 scenes are exported in five contact sheets at
`outputs/qwen_image_21/rgba_sampler_comparison/page_{1..5}.jpg`.

### Frozen SDE confirmation: fresh sources and two additional seeds

Completed the predeclared new-source confirmation without further updates or
reward changes. All 200 generated source hashes are disjoint from both previous
200-scene datasets. Evaluation uses 40 scenes with two fixed seeds, 160 actual
base/checkpoint outputs, all first-step likelihood replay errors zero.

- Seed 314159: base 0.7368146871, step-16 0.8001754907, gain 0.0633608036;
  38 improved / 2 declined.
- Seed 271828: base 0.7870726360, step-16 0.7839351617, delta -0.0031374743;
  22 improved / 18 declined, interval includes zero.
- Averaging the two seeds within each source: base 0.7619436616, step-16
  0.7920553262, delta 0.0301116646. The source-bootstrap interval
  [0.01998007, 0.04263187] is conditional on these two fixed seeds; it does not
  estimate uncertainty over all diffusion seeds, and these are not 80 sources.

The positive training-seed result is therefore not uniformly robust across noise
seeds. Native-inference improvement remains unestablished. This is narrow
synthetic exact-reference task evidence, not independent human quality.
Artifacts: `rgba_confirm_sde_{base,step16}_{314159,271828}` generation/scoring
runs, paired reports, and `outputs/qwen_image_21/rgba_confirmation_summary.json`.
Reproduction/aggregation script: `outputs/qwen_image_21/summarize_rgba_confirmation.py`.

### Controller checkpoint content identity (real resume validation in progress)

The previous controller checkpoint only bound a supplied revision label and
trainable tensor shapes. The v2 path now binds local encoder/processor bytes,
LoRA scaling/topology and relevant runtime versions, checking the sources again
after load. Controller policy stamps derive from that identity. The editor now
uses the existing strict model resolver, with its content-bound policy in the
controller training contract and evaluation admission check. V1 experimental
controller checkpoints are rejected rather than silently assigned unverifiable
historical identities. Existing diffusion checkpoints retain their normal format.

The shared task-record parser resolves paths against the declaring file in all
three visual CLIs and rejects unknown fields. This parser is a schema boundary;
the session builder remains the lifecycle boundary. No family adapters or model
workflow constants were flattened for line-count reduction.

Nine controller/replay/episode/admission boundary tests pass, including rejection
of a changed frozen base before mutating parameters and exact Adam/RNG continuation.
A real v2 first update has completed with two four-decision-total editing episodes:
parity error 0, gradient norm 0.0126951234, mean return 0.4088645458, two tool calls
per episode. Artifacts: `controller_identity_v2_step1`. Continuation into
`controller_identity_v2_resume2` is running; no capability improvement is claimed.

The real v2 continuation completed successfully: update 2 restored model/Adam/RNG
and collected two new episodes, three decisions total, replay max error 0,
gradient norm 3.5755882263, optimizer stepped. Mean return 0.0783585548 is not
comparable as a quality-gain estimate to the previous fresh episode group.
Artifacts: `controller_identity_v2_resume2/checkpoint-2.pt`. The new identity-bound
evaluation path is now being exercised on the previously used sweater diagnostic
source; it is controller-training-disjoint but no longer a fresh final holdout.

### Isolate first-step trainer diagnostics

- Changed: moved the inline likelihood/invariant evidence collection into
  `OnlineTrainer._first_step_parity_probe`, shortening the optimizer path by
  roughly 120 lines and replacing temporary-style local names with their roles.
- Kept: debug activation, NFT's optional invariant hook, distributed parity
  verdicts, JSON event/field names, failing-probe persistence and post-update
  evidence. The mandatory full-update gate still owns enforcement.
- Boundary rationale: this method owns a substantial diagnostic branch and its
  bounded microbatch state; it is not a thin forwarding wrapper. No new constants
  or files were introduced.
- Non-goals: splitting the trainer/checkpoint modules, changing precision or
  sampler rules, or extending optional diagnostics into the streaming path.

Validation: 53 focused diagnostics/restore/composition tests passed (one skipped),
then 212 online-trainer tests passed in the non-GPU/non-distributed lane. Logs:
`/tmp/vrl2-parity-refactor-tests.log`, `/tmp/vrl2-trainer-refactor-suite.log`.

### Invalid reward weights fail before construction

Public reward configs and direct composite-reward construction now reject NaN and
infinite weights before constructing reward components. Existing negative-weight
configuration errors, zero-weight diagnostic behavior, and finite direct-API
weighting semantics are preserved. Ninety-one composite/config/preset tests pass.
This closes a setup-time failure that otherwise reached resource allocation or
only failed after scoring; it does not change valid reward arithmetic.

The v2 controller evaluation also completed: one previously used sweater source,
eight native editing steps, net returns stop -0.2914454, fixed 1.9872922,
random 1.9872922, controller 1.9148562, best-of-N 1.8872922. The controller remains
below the single-edit baseline. This validates content-bound restoration and
comparison execution, not controller capability gain.

### Real GPU parking and crash-resumable independent scoring

Both existing real GPU parking tests passed on the RTX 5090: CPU-offload cleanup
releases the BLAS-workspace-pinned allocation, and CuMem sleep releases this
process's physical pages while another process owns a separate GPU allocation.
Wake preserves tensor contents. This tests normal allocator operation, not an
intentionally corrupted live allocator. The isolated EditReward environment has
no vLLM installation, so its proven reload parking remains in use.

Independent scoring now holds a kernel directory lock for each complete write or
snapshot read. Process death releases the lock. A persistent versioned marker
blocks older sentinel-only writers; legacy unversioned locks are still rejected.
The helper is a real concurrency/protocol boundary shared by scorer and reader,
not a workflow wrapper. Score/provenance formats and input identity checks remain
unchanged. This guarantee requires the local POSIX directory-lock semantics;
remote computations may continue after their client dies, and a kill before
initial provenance publication has no committed samples to recover.

A real subprocess completed the first CPU score and blocked during the second.
Concurrent read/write attempts failed; SIGKILL released ownership; the next run
reused the completed sample and scored only the missing one. Thirteen targeted
crash/evaluation/diagnostics/calibration tests passed, then 194 reward inference,
HTTP service and snapshot tests passed. Existing real RGBA scoring also resumed
with 40 reused / zero recomputed rows and remained readable. Logs:
`/tmp/vrl2-real-gpu-parking.log`, `/tmp/vrl2-evaluation-crash-tests.log`,
`/tmp/vrl2-reward-snapshot-regression.log`.

### Fixed early-window SDE experiment (running)

To investigate the native/SDE quality gap, a new base-initialized 16-update run
uses the same synthetic training data, optimizer, seed, eight denoise steps and
noise 0.7, but fixes the stochastic window to steps [0,4), with native steps
thereafter. `actor.timestep_selection=sde_window` and fraction 1 ensure only the
four actual stochastic transitions train. This is a separate predeclared probe,
not a relabeling of the full-window checkpoint. It will first be evaluated under
native inference on the original monitoring split. No new final-holdout claim
will be made from the previously used evaluation sets.

Artifacts: `outputs/qwen_image_21/rgba_early_window16`; log:
`/tmp/vrl2-rgba-early-window16.log`. At ten completed updates the latest reward is
0.9243889451 and replay max difference 2.3841858e-7; training scores alone do not
establish a native-inference improvement.

### Early-window native monitoring result and frozen confirmation plan

The 16-update early-window run completed. Native eight-step evaluation on the
original 40-source monitoring split gives base 0.9203345891 versus checkpoint
0.9244253900: paired +0.0040908009, 30 improvements / 10 declines, descriptive
interval [0.00106153, 0.00747535]. All native first-action replay errors are zero.
This split was used for diagnosis and recipe selection; it is not final evidence.
Artifacts: `rgba_early_window16_native_eval40` generation/scoring and
`outputs/qwen_image_21/rgba_early_window16_native_paired.json`.

Freeze the step-16 checkpoint and confirm on 100 new scenes from dataset seed
20260924, with two independent per-source noise streams (master seeds 314159 and
271828), eight native steps, no further updates or reward tuning. The probe now
supports `--seed-mode independent`: each effective seed is derived from the master
seed, prompt and ordered reference content hashes, and is recorded per output.
This avoids reusing one identical initial-noise realization across every scene.
The prior shared-seed default stays available for exact historical reproduction.
Source hashes are checked before/after generation. Base/checkpoint comparisons
must match effective per-case seeds and seed mode. Source-level uncertainty will
average the two samples within each source rather than counting 200 sources.

### Compile maintained documented launches

Added a bounded Markdown config gate over root docs and `docs/sprints/info`.
Six current bundled experiment/recipe commands compile through the actual loader
and strict config parser. The test also proves a removed `production.*` override
fails, including multiline shell commands, and supports both --config forms.
No commands, backends or model downloads are executed. External scorer YAMLs,
dynamic config variables, CLI option validity and filesystem/GPU feasibility are
explicitly outside this static gate.

Eleven shell fences in seven measured/historical info documents are now explicitly
labelled historical; their command text and reported measurements are unchanged.
The Cosmos runbook has a separate current preflight command. This avoids rewriting
recorded experimental conditions to pretend they ran with today's schema.
Done/research archives and the user's pending sprint edits remain untouched.
Both documented-command tests pass. Log: `/tmp/vrl2-doc-config-gate2.log`.

### Apply frozen reward combinations without inference

The independent calibration CLI now applies a frozen combination to new raw
scoring snapshots or an exact join of scorers, without labels, model loads, or
refitting. It reports each signed standardized axis contribution and preserves
original scorer diagnostics. Failed/missing measurements remain unscored; changed
recipes, absent axes and nonfinite arithmetic are rejected. The application ID
binds observed values, not just the raw evaluation's input/recipe run ID.

A shared validation helper stays in the calibration module because both holdout
evaluation and application need identical artifact/vector checks. No additional
wrapper modules, business constants, or changes to reward transport/registration
were introduced. Training integration and automatic weight promotion remain out
of scope for this change. Six calibration/join/documented-command tests pass;
all preference examples are explicitly synthetic test evidence. No real human
labels or validated deployed combination are claimed.

### Frozen early-window native confirmation completed

All 400 Qwen generations and independent CPU RGBA scores completed: 100 new
sources, two independent content-derived seeds per source, base versus the same
frozen early-window step-16 checkpoint. All first-action replay errors are zero;
within each master-seed arm all 100 effective seeds are distinct and exactly
matched between base and candidate. All 500 newly generated scene hashes are
disjoint from the previous 600 source images.

Averaging the two draws within each source gives base 0.9064855255 versus trained
0.9190424232, paired mean +0.0125568977, median +0.0036112041, source bootstrap
95% interval [0.00754794, 0.01849159], 79 improved / 21 declined sources. Both
master-seed means improve. A few recovered large errors contribute more to the
mean than typical cases; the median is reported explicitly. This supports a
native-inference gain on this frozen synthetic extraction distribution, not
natural-photo matting, semantic decomposition, or independent human preference.

The probe's shared-seed default remains backward compatible; independent seeds
bind master seed, prompt and ordered reference hashes. The small inline derivation
stays at the generation request boundary; it does not warrant a new helper module.
Artifacts: `outputs/qwen_image_21/rgba_early_confirmation_summary.json`,
`summarize_early_confirmation.py`, `rgba_early_confirm_paired_{314159,271828}.json`,
and matching `rgba_early_confirm_{base,step16}_{314159,271828}` generation/scoring
folders. The frozen checkpoint remains `rgba_early_window16/checkpoint-16`.

### Verifier-backed visual episodes and controller experiment

Visual sessions now accept the existing RewardConfig YAML for local CPU and
version-pinned HTTP components, including mixed multi-axis rewards. Shared-device
HTTP release remains required; local GPU/Ray configurations fail before Qwen
allocation. No second reward registry or serving scheduler was introduced.
Recipe identity participates in the controller resume contract.

VisualTask accepts content-bound reward-only assets, such as target_image and
edit_mask. The judge preserves RGBA, passes these files only to rewards, and
verifies them around scoring. Episode replay restores and verifies the assets;
held-out checks include exact asset overlap. The task serialization helper is a
protocol boundary: omitting empty assets preserves older no-assets identities.
Existing thin family/reward adapters and the direct EditReward mode remain intact.
Thirteen focused visual lifecycle/replay/credit/config tests pass, including a
real CPU pixel oracle, a failed opaque candidate, and a mutated target rejection.

A predeclared controller-only development experiment is running: two training
sources, each in needs-edit and already-done form, two episodes per task, eight
updates, frozen base Qwen editor, native eight steps, temperature 4, cost 0.1.
Monitor sources are separate. Targets are not shown to the controller or editor;
controller images remain RGB composites, so alpha observability is a known limit.
Artifacts: `outputs/datasets/rgba_visual_control_seed42` and
`outputs/qwen_image_21/controller_rgba8_attempt2`; log:
`/tmp/vrl2-controller-rgba8-attempt2.log`. The first launch used a mistyped local
snapshot path and failed before model allocation; its output was retained.
No learned-control gain is claimed until trained/untrained and fixed-policy
comparisons complete.

The verified early-window recipe is now a bundled experiment preset, inheriting
the existing smoke configuration and changing only timestep selection, SDE window,
run length/save interval and output location. With the five recorded local path
bindings supplied, its complete resolved configuration exactly equals the real
16-update run's saved configuration. Thirty-eight preset/documentation tests pass.
A selected-case contact sheet shows the two worst, two median and two largest
source-level changes with both seeds; large recoveries visibly include formerly
translucent foregrounds becoming opaque. The selection is explicitly rank-based,
not a representative visual-quality sample:
`outputs/qwen_image_21/rgba_early_confirmation_selected.jpg`.

### Full CPU regression and checkpoint integrity

The frozen-environment CPU regression completed with 4,821 passed, 17 skipped,
155 deselected (GPU/distributed/optional/slow lanes excluded), in 388.41 seconds.
Log: `/tmp/vrl2-production-cpu-suite2.log`. This covers the verifier-backed visual
path and the preceding reward/controller changes; the subsequent checkpoint
hardening has its own 128 passing checkpoint/controller tests.

New diffusion checkpoint saves bind checkpoint.pt to a SHA-256 metadata field;
load checks it before deserialization. Legacy sidecars without that optional field
remain readable. Structural discovery remains a cheap metadata/size scan; it does
not hash every historical checkpoint. A real same-length tensor-storage byte flip
remains readable by torch.load but is rejected by the new integrity gate.

Publication now flushes the complete staged tree (payload, metadata, adapter
exports and directory entries) before replacement, then flushes the parent.
Controller checkpoint publication also flushes its directory. The shared
fsync_directory helper is an exception-safe filesystem durability boundary, not
a workflow wrapper; checkpoint filename/schema constants remain protocol data.
A directory-flush fault leaves the previously published checkpoint intact.
Same-name directory replacement still has the existing removal/rename gap after
successful staging; no claim of atomic exchange or physical power-loss testing
is made. Hashes detect accidental corruption, not a party rewriting both payload
and sidecar. Log: `/tmp/vrl2-checkpoint-integrity-tests2.log`.

Visual HTTP admission now uses each component's advertised placement guarantee,
rather than assuming every HTTP service needs a GPU lease. A live CPU service
passes without parking; a service advertising neither isolation nor a parking
lease fails before Qwen allocation. Every external component must independently
satisfy that requirement, so a mixed recipe cannot hide an unsafe sibling behind
another component's successful park. Eighty-two visual/service tests pass,
including real HTTP startup and shutdown; direct shared-GPU EditReward still uses
its acknowledged parking lease. Log: `/tmp/vrl2-visual-http-placement-tests.log`.

### Long-running independent service soak started

A detached read-only worktree at `2aaf81fc6` isolates the soak from ongoing source
edits: `/tmp/vrl2-reward-soak-20260922`. Its owned CPU RGBA HTTP server uses port
18316, independent of the parked EditReward service on 18315 and all GPU work.
Every minute it checks file and tensor transport, exact oracle/empty-alpha scores,
idempotent result payloads, bad-digest rejection, service health, RSS and open FDs.
The cache is capped at 32 requests and is repeatedly exceeded. Only the owned
server is restarted (after cycle 3 and every 60 cycles), with a typed outage and
recovery checked through the same client. Planned end: 2026-09-23 07:40 UTC.

The first harness attempt incorrectly demanded identical client-measured HTTP
transit timing on a cached reply. Source inspection confirmed that field is
remeasured on every delivery. The retained initial attempt and script document
that harness failure; attempt 2 excludes only that one timing field and compares
all other evidence exactly. Six cycles / 60 successful requests / six rejected
bad digests and the first controlled restart have passed so far. This is an
ongoing CPU service/lifecycle test, not a completed soak or GPU release claim.
Artifacts: `outputs/reward_evaluation/service_soak_20260922_attempt2`; driver:
`outputs/reward_evaluation/service_soak_20260922/run_soak.py`; log:
`/tmp/vrl2-reward-service-soak-attempt2.log`.

### Additional frozen distribution checks prepared

Two new conditions are frozen before evaluation: native 512-resolution transfer
(50 new sources, two per-source seeds) and explicit 40% foreground-opacity
extraction (another 50 new sources and two seeds). Both compare the same early-
window step-16 checkpoint to base, without further training. The latter is an
exploratory challenge motivated by the observed recovery of overly translucent
foregrounds; it is not relabelled as the original confirmation set.

The data builder now supports foreground_opacity, records its quantized alpha and
writes an explicit opacity instruction. It scales antialiased coverage before
compositing the exact target into the source. The three verifier tests pass,
including a translucent oracle and an opaque-foreground exploit. Four default
scenes regenerate byte-identically to the historical opaque source/target files;
the local fixture palette and all training/reward interfaces remain unchanged.
Protocols: `outputs/qwen_image_21/rgba_resolution512_protocol.json` and
`outputs/qwen_image_21/rgba_translucent_protocol.json`. The latter's 250 exact
oracles were independently scored before any real generation.

### Reusable source-balanced paired output comparisons

Added `analyze_scores paired` to compare baseline/trained images scored by the
same frozen recipe, accepting multiple matched sampling runs. It checks sample
grids, prompts, auxiliary identities and metadata; output images may differ.
Repeated measurements average within declared source groups (or exact reference
hash / prompt ID fallback). Failures retain unmeasured values and explicit coverage,
never zeros. Sampling parameters absent from metadata remain a separate operator
validation responsibility; this tool does not infer them from generated pixels.

Five paired/diagnostics/join tests pass. Running the public CLI on the completed
400-image confirmation yields 200 paired observations / 100 sources and reproduces
the prior source-mean delta to floating-point precision (+0.0125568977). Its fixed
bootstrap seed differs from the initial experiment script, giving the compatible
interval [0.00749752, 0.01884564]; the original predeclared report is retained.
Artifact: `outputs/qwen_image_21/rgba_early_confirmation_paired_cli.json`.
The analysis stays in the existing diagnostics module and CLI; no new registry,
backend table, or one-function wrapper module was introduced.

### Controller exploration is now explicit evidence

The eight-update CPU-verifier controller run completed 64 real episodes with
zero replay error at every update. It usually exhausted both editing calls.
Binary action likelihoods show why exploration is limited: before training,
mean stop probability on already-done sources was about 2.6%, versus 0.7% on
needs-edit sources. By policy version 6 these were about 4.7% and 1.2%.
This is a distribution diagnostic, not a demonstrated held-out behavior gain.

New replay traces retain the full categorical behavior distribution, bound to the
saved processor-tensor artifact. Replay checks normalization, chosen-action
likelihood and exact agreement with that artifact. Trainer metrics expose mean
entropy/stop probability and coverage; legacy records without these fields remain
valid and emit missing diagnostic values. Sampling and the credit objective are
unchanged. Ten visual replay/credit/session tests pass, including distribution
mutation rejection and original-format replay compatibility. Log:
`/tmp/vrl2-controller-exploration-metrics-tests.log`.

### Initial controller monitor completed; next exploration protocol frozen

On two source-separated development sources, each in needs-edit/already-done
form with two evaluation seeds, checkpoint-start and checkpoint-8 comparison
runs completed. Source-balanced controller net return moves from 0.6093720440
to 0.6199010701, still below fixed-one-edit 0.8039998570 and random 0.6227084076.
The other baseline means are bit-identical across runs; stop is 0.4999999968 and
best-of-N is 0.7128534832. This does not pass the controller capability gate.
Artifacts: `controller_rgba_monitor_start` and `controller_rgba_monitor_8` under
`outputs/qwen_image_21`.

A new development protocol is frozen in
`outputs/datasets/rgba_visual_control_exploration/protocol.json`: four training
sources / eight state-specific tasks, four episodes per task, 16 updates (512
episodes), temperature 16, unchanged learning rate 1e-4, frozen base editor,
native eight steps at 256, and tool cost 0.1. The previously used two-source
monitor remains diagnostic. Sixteen new sources / 32 state-specific tasks from
seed 20260927 are reserved for final matched trained/untrained comparisons.
The existing controller objective and reward remain unchanged; this tests richer
sampling under weak initial stop exploration. It is prepared, not yet running.
The Qwen sprint created in this goal is updated in Chinese with actual passed and
unpassed gates; preexisting user sprint edits remain untouched.

### Offline human preference collection is connected to calibration

Added `calibrate_scores prepare-review` and `import-review`. Pair selection and
source-separated calibration/holdout splits are explicit input; the exporter
reuses calibration leakage validation, copies content-verified original media,
randomizes order and A/B sides, and keeps scores/model IDs/splits out of the page.
The separate audit manifest binds the mapping. Import accepts only explicit answers
and checks that identity, restores original left/right orientation, omits unanswered
pairs, and refuses to overwrite label files. This is presentation blinding, not
access control for someone inspecting the audit file. No preferences are generated
from model scores, and no human-calibrated weights are claimed.

The browser template is a packaged HTML asset; the new annotation module owns the
portable review artifact boundary and identity mapping. Scoring/runtime APIs and
reward adapters stay unchanged. No prompt vocabulary, alternate reward registry,
or generic UI framework is added. Existing thin model adapters remain useful
transport/lifecycle boundaries, and schema/file constants remain protocol data.

Six calibration/join/paired/review tests passed after frozen environment sync.
Real headless Chrome loaded the six-pair armchair development packet; all images,
explicit A/unsure selection, navigation, clearing and export availability passed.
Automated choices were cleared and never exported as human labels. Screenshot was
visually reviewed. Artifacts: `outputs/reward_evaluation/armchair_blind_review/`,
`armchair_blind_review_browser_check.json`, `/tmp/vrl-reward-review.png`.
This one-source packet is a usable demonstration, not enough for source-separated
calibration or holdout evidence. Tests: `/tmp/vrl2-preference-review-tests2.log`.

### Frozen checkpoint transfer to 512 pixels: tail gains with broad small regressions

All 200 native generations on 50 new sources and two independent per-source draws
completed. The fixed 256-trained early-window checkpoint has source-balanced mean
RGBA match 0.8877593782 versus base 0.7927326440; delta +0.0950267342, descriptive
source bootstrap [0.04053817, 0.15790327]. Both seed means improve (+0.09388881 and
+0.09616466), but only 17/50 source-average deltas are positive and 33 negative;
the median delta is -0.01666264. This is reduced severe-tail failure alongside
small regressions on most scenes, not uniform quality improvement.

Input/task/seed/sampler identities were checked separately per seed before using
the generic source-balanced paired report. All native replay errors are zero.
The selected worst/median/best contact sheet was viewed: large blue-object gains
recover almost transparent outputs, while small regressions include boundary and
background-alpha differences. Even a large-recovery output retains visible texture.
For seed 314159 mean background alpha increases from 0.00444813 to 0.00672858;
this secondary regression remains visible beside the scalar improvement.
Artifacts: `rgba_resolution512_source_paired.json`, per-seed paired JSON, and
`rgba_resolution512_selected.jpg` under `outputs/qwen_image_21`.

The translucent challenge continues without tuning this checkpoint. The prepared
controller experiment is queued after all eight RGBA generation reports complete,
checks the owned predecessor process and GPU availability, then runs from frozen
checkout 2050e6d6f. Driver: `run_controller_exploration_queue.py`; log:
`/tmp/vrl2-controller-exploration-queue.log`. It is not yet counted as completed.

### Real HTTP stale auxiliary-file cache and repeated-cancel defects fixed

A real CPU RGBA HTTP regression reproduced stale success: changing target.png
while leaving the candidate and request payload unchanged still returned the
cached score of 1.0. The service now recognizes built-in file metadata at the
artifact schema boundary, confines it to configured roots even for tensor-uploaded
candidates, snapshots auxiliary hashes, verifies files after scoring, and compares
those hashes on cached-success replay. Changed/outside files return typed
`path_not_allowed`; a deliberately changed target requires a new request identity.
Server-owned hashes do not replace caller provenance or provide an atomic file
snapshot. Custom rewards with new file metadata must extend this schema boundary.
HTTP fields/version remain unchanged; running old service processes do not acquire
this fix automatically. Existing frozen-soak evidence still refers to 2aaf81fc6.

A second regression used a blocked file reader and repeated HTTP DELETEs. Before
this change the second cancel acknowledged success while the reader was live and
released its admission slot. Validation now retains ownership through repeated
cancellation until the reader exits. The test verifies another request remains
`overloaded` until that exit. No new scheduler or model adapter is introduced;
metadata names are actual input-schema keys, not a workflow business taxonomy.

112 service/offline-evaluation/visual-session tests passed, including real RGBA
changed-target/new-request behavior, uploaded-candidate root enforcement,
mid-scoring target mutation, and the live HTTP repeated-cancel regression.
Before logs: `/tmp/vrl2-auxiliary-integrity-before.log`,
`/tmp/vrl2-auxiliary-repeat-cancel-before.log`. After:
`/tmp/vrl2-auxiliary-integrity-tests3.log`.

### Translucent transfer fails; a separate dense-axis hypothesis

The frozen opaque-trained early-window checkpoint completed 200 native generations
on 50 fresh 40%-opacity sources, using both predeclared independent per-source
seeds. Source-balanced RGBA match is base 0.0440094400 versus trained 0.0076746694;
delta -0.0363347706, descriptive interval [-0.05972882, -0.01842714]. No source has
a positive average delta. Many outputs are opaque circles colored like the input's
white-background composite, rather than actual translucent foregrounds. The selected
contact sheet was viewed. This is a failed transfer gate, not broad RGBA capability.
Artifacts: `rgba_translucent_source_paired.json`, per-seed paired JSON and
`rgba_translucent_selected.jpg` under `outputs/qwen_image_21`.

The default match also clips several visibly different poor predictions to zero.
Added an explicitly selectable measurement, `rgba_dense_match = alpha_soft_iou /
(1 + foreground_normalized_premultiplied_l1)`, preserving order away from that floor.
Default `rgba_match` and all existing error axes keep their definitions. Oracles,
hidden-RGB invariance and empty-alpha rejection remain; a test shows progressive
opacity repairs receive a strict ordering where all original scalar values are
zero. Eight RGBA/service/session tests pass (`/tmp/vrl2-rgba-dense-axis-tests.log`).
This is an unverified reward-shaping hypothesis to compare in a fresh mixed-opacity
training experiment, not a retroactive change to the failed challenge.

The controller exploration run started from frozen 2050e6d6f after all generation
audits completed. A second long-running CPU service soak now uses fixed checkout
337702cb5 on port 18318 and checks changed-target cache rejection every cycle,
while retaining the original frozen soak. Its output is
`outputs/reward_evaluation/service_auxiliary_soak_20260922`; driver
`outputs/reward_evaluation/run_auxiliary_service_soak.py`, log
`/tmp/vrl2-reward-auxiliary-soak.log`. Both end after the 24-hour goal minimum.

### Follow-up GPU experiments prepared and serialized

The higher-temperature controller completed its first 32-episode update with
replay error 0, gradient norm 0.12107866, mean net return 0.59969338 and stop
fraction 0.5. Mean recorded stop probability is 0.25574352 and action entropy
0.56644051. This verifies richer exploration, not a held-out control advantage.

Prepared a separate mixed-opacity dataset with 400 training and 100 monitoring
sources across alpha 64/102/153/204/255. All 500 exact RGBA oracles score one under
both scalar definitions; exact train/monitor source overlap is zero. Data, digests,
parsed resolved training configurations and the predeclared 32-update protocol are
under `outputs/datasets/rgba_mixed_opacity_20260928`. Clipped and dense arms start
from the same base model and differ only in the selected scalar (plus output paths).
This is new training data after the failed transfer experiment, not a redefinition
of that held-out result. A further fresh confirmation is required for a final claim.

Owned `run_followup_gpu_queue.py` waits for the current controller process to exit
successfully, verifies GPU availability and pinned checkouts, and runs sequentially:
controller start/step-16 monitoring and final comparisons; both 32-update diffusion
arms; and native inference for base/clipped/dense on two independent per-source
seeds. It aborts on the first failure without automatic retries. Commands and
running/completed state are durable under
`outputs/qwen_image_21/followup_gpu_queue_20260922`; driver log:
`/tmp/vrl2-followup-gpu-queue.log`. These queued jobs have not yet completed.

### Full CPU regression after review and service-integrity changes

Frozen-environment CPU lane completed with **4,831 passed, 17 skipped,
155 deselected, 72 warnings in 392.21 seconds**. Command:
`HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/pytest -q -m 'not gpu and not distributed and not optional and not slow_test'`.
Log: `/tmp/vrl2-production-cpu-suite3.log`. This includes the optional dense RGBA
axis, blinded-review import/export, checkpoint durability/integrity, auxiliary-file
HTTP cache revalidation, and repeated file-reader cancellation ownership fixes.
No GPU/distributed acceptance is inferred from this lane.

### Optional controller alpha observations remove a concrete perceptual alias

Default white-composited RGB perception cannot distinguish a translucent white
image from an opaque white image. Added `--controller-observation rgba` to episode,
training, comparison and likelihood-probe commands: append the original/current
alpha masks as two explicit grayscale views. Only the actual observed image masks
are exposed, not oracle targets, edit masks or reward scores. The editor/reward
and finite-action/credit interfaces stay unchanged. Four views have greater
perception cost than the default two; this is not free compute.

Replay tensors and checkpoint payloads bind observation mode; evaluation checks
it before loading the model, and restore/replay reject a mismatch. Old artifacts
without the optional field mean `rgb`, preserving existing experiments. Nine
controller replay/credit/session/comparison tests pass, including a real tiny-VL
forward/gradient, default-mode compatibility, opposite-mode rejection, and two
images identical on white but distinct through actual alpha. Log:
`/tmp/vrl2-controller-alpha-observation-tests2.log`. The full real-Qwen four-view
probe has not run yet. Current training and queued comparisons remain frozen in
RGB mode and cannot be cited as validation of the new perception mode.

### 13:49 UTC — supervisor drains workers after leader exit

A real subprocess regression reproduced a leaked worker: the group leader died
by SIGKILL, its worker ignored SIGTERM, and the next attempt observed that worker
still alive. The supervisor now drains its owned process group after leader exit
and on operator stop, waits a bounded grace period, escalates to SIGKILL, and
refuses to restart if live members remain. Linux procfs distinguishes dead
zombies from live workers without adding an undeclared psutil dependency.
Cleanup is scoped to the group established by start_new_session; this does not
claim control of descendants that deliberately detach into another session.

The new crash/restart regression and existing operator-stop test both exercise
TERM-resistant workers with a readiness handshake. All 66 supervisor tests pass
(`/tmp/vrl2-supervisor-orphan-tests2.log`); touched-file Ruff checks pass. This is
real CPU process cleanup evidence, not yet a real GPU interruption/resume result.
The group-membership and drain helpers remain separate because they centralize
ownership/liveness and bounded termination shared by exit and stop paths; no
unrelated architecture or naming cleanup was performed.

### 13:57 UTC — queued actual Qwen leader-interruption recovery

`outputs/qwen_image_21/run_real_recovery_after_queue.py` waits for the owned
four-view probe driver (PID plus start-time identity) and its successful report,
then verifies GPU availability. It uses detached source `e87241690` and the
existing real Qwen checkpoint-16. Both arms continue through epoch 18 with the
same deterministic settings and save every epoch. The interrupted arm kills only
its owned training leader by SIGKILL immediately after checkpoint-17 is complete;
the supervisor must clean workers and resume automatically. Final comparison
includes model, trainer/optimizer, progress and RNG payloads, with mismatch paths
retained rather than assuming equality. No result is available yet.

Output: `outputs/qwen_image_21/real_recovery_20260922`; log:
`/tmp/vrl2-real-recovery-queue.log`. Both configurations pass schema validation and
all referenced inputs exist. An initial waiting-only harness was replaced before
GPU execution to correct DictConfig construction; its ownership/status records
remain in `real_recovery_20260922_preflight_attempt1`. This is a queued acceptance
experiment, not evidence that abrupt GPU recovery already passes.

### 13:56 UTC — controller CLI interruption boundary verified on CPU

Added an end-to-end CLI regression using a small differentiable categorical
controller and deterministic pixel editor/judge. It executes the real episode
collector, controller trainer and checkpoint code, injects failure after the
sixth collected episode (two episodes into the second uncommitted update), then
resumes checkpoint-1 into a fresh directory. Compared with uninterrupted training,
final trainable parameters and Adam moments are bitwise equal, policy version and
cursor match, and resumed episodes 4–7 retain identical seeds and returns. The
failed run publishes neither checkpoint-2 nor a completion result. No production
code change was needed for this existing recovery contract.

Four controller CLI/credit/replay tests pass in
`/tmp/vrl2-controller-resume-regression.log`; touched-file Ruff passes. This proves
CPU integration across the collection/commit boundary, not real-model SIGKILL
recovery or learned capability. The queued GPU interruption protocol above remains
pending. The earlier 13:57 queue-entry timestamp is approximate; this entry was
recorded against the actual clock at 13:56 UTC.

### 14:01 UTC — evaluation source audit and queued controller analysis

Recomputed exact input/reward-asset hashes for the frozen controller manifests:
train has four source groups/eight tasks, monitor two/four, and final sixteen/32.
No assets overlap across the three splits. This is exact content separation, not
a claim about automatic near-duplicate detection. The four training source pairs
are not RGB-white-composite aliases: their mean absolute 8-bit RGB differences are
15.80, 45.43, 13.13 and 22.76. Therefore hidden alpha alone does not explain the
current controller's behavior on this dataset.

`outputs/qwen_image_21/analyze_controller_exploration.py` is queued behind the
owned evaluation process. It will require complete comparisons and intact final
media, verify unchanged stop/fixed/random/best-of-N baselines across initial and
trained evaluations, and aggregate repeated seeds and both task states within
original source groups. It reports trained-minus-initial and trained-minus-each-
baseline deltas, per-source signs, medians and descriptive unadjusted bootstrap
intervals. It does not automatically promote a policy or label statistical
intervals as general visual capability. Log:
`/tmp/vrl2-controller-exploration-analysis.log`.

A fresh full CPU regression is running after the alpha-observation, supervisor
cleanup and controller CLI recovery changes. Existing GPU jobs and HTTP soaks
remain live and continue from their own frozen checkouts.

### 14:07 UTC — full CPU regression after recovery and observation changes

The full configured CPU lane passed: **4,834 passed, 17 skipped, 155 deselected,
72 warnings in 396.65 seconds**. Command:
`HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/pytest -q -m 'not gpu and not distributed and not optional and not slow_test'`.
Log: `/tmp/vrl2-production-cpu-suite4.log`. This includes the new alpha-observation,
supervisor orphan cleanup and controller CLI interrupted-collection coverage.
The active Qwen training, serialized follow-up queue and both CPU reward soaks
were verified live after completion. No skipped hardware lane is counted as passed.

### 14:18 UTC — admission records retain inspectable task/source metadata

Inspection of real RGBA admission records found task/source metadata contributed
to prompt_key but was not retained in the record; task_id was also not recognized
as a prompt_id fallback. Added a detached JSON-normalized input_metadata snapshot
using exactly the existing hash input, and the task_id fallback after prompt_id/id.
This makes dropped groups traceable without changing selection, advantages, reward
values or prompt-key construction. Existing records remain unchanged and may lack
the new optional field. Metadata paths still are not file-content attestations.

Sixteen focused admission/online-step tests pass in
`/tmp/vrl2-admission-metadata-regression.log`, including distinct source keys and
metadata mutation after selection without changing the persisted earlier record.
Touched-file Ruff passes. No new constants/modules or generic verifier abstraction
were introduced; the shared admission boundary remains intact. The previous full
CPU run predates this additive audit-field change. Active frozen GPU jobs also
retain their original record shape.

### Mixed-opacity budget interpretation

The frozen ablation's `updates: 32` protocol shorthand means 32 collection epochs,
not an assertion of 32 successful parameter updates. The queued
`outputs/qwen_image_21/audit_mixed_training_budget.py` waits for both training jobs,
verifies checkpoint file digests, checks all 32 metric epochs, and independently
reports trainer/global counters, per-parameter Adam step counts, admission group
keep/drop/partial counts and collected/selected samples. Its output is
`outputs/qwen_image_21/rgba_mixed_training_budget.json`; log:
`/tmp/vrl2-mixed-training-budget.log`. It does not adjust either arm's budget after
seeing the valid-group rate. This is not yet a completed training result.

### 14:27 UTC — training-state diagnosis and predeclared alpha perception arm

Training-only diagnostics now separate needs-edit and already-done initial states.
Across collection policy versions 0 to 8, mean initial stop probability falls from
0.22752 to 0.14660 for needs-edit and from 0.28698 to 0.20499 for already-done.
The already-done net return falls from 0.79996 to 0.71987 in those sampled groups,
while needs-edit rises from 0.39943 to 0.63972. These are small training groups, not
independent evaluation; they suggest a broad edit preference rather than verified
conditional stopping. Script/report:
`outputs/qwen_image_21/inspect_controller_training_states.py` and
`controller_training_state_diagnostics.json`.

Before any final-holdout results are available, a second observation arm is frozen:
four alpha-aware views, with the same training sources, seed, temperature 16,
learning rate 1e-4, 16 updates/512 episodes, frozen editor and judge, and matching
monitor/final comparisons. Its protocol is
`outputs/qwen_image_21/alpha_controller_queue_20260922/protocol.json`; frozen source
is f1391b5a5. This tests perception, not an assumed benefit. The final split is
shared by these predeclared observation arms; it must not subsequently become a
hyperparameter tuning set.

The owned queue waits for the four-view probe and actual GPU interruption/recovery
comparison through the existing dependency chain, requires successful bitwise
recovery, then checks available GPU memory before each job. It stops on failure
and does not overlap current GPU work. Driver PID 300707, log
`/tmp/vrl2-alpha-controller-queue.log`. No alpha training has run yet.

### Independent repeat-scoring diagnostics

Added `analyze_scores repeat`: same frozen scorer/config and exact input grid,
per-sample repeated scores/statuses, range/sample standard deviation, and source-
balanced summaries. Missing/error/absent-axis observations remain explicit and
samples with fewer than two scores are not assigned zero noise. The CLI rejects
reusing the same directory; equal recipe/input run IDs remain valid for genuinely
fresh executions. This is a report function in the existing diagnostics module,
not a new scorer, service or training-time adaptation layer.

Eleven focused evaluation/join/diagnostic tests pass in
`/tmp/vrl2-repeatability-regression.log`; touched-file Ruff passes. Real CPU audit:
three fresh executions × 600 existing RGBA media, 200 sources, all nine axes with
zero observed variation and zero cached/reused records. Artifacts:
`outputs/reward_evaluation/rgba_repeatability_20260922`; log:
`/tmp/vrl2-real-repeatability-audit.log`. Two existing real EditReward executions
also match exactly on four armchair candidates, but span a service transport
update/restart and are only narrow compatibility evidence. No general zero-noise,
quality, or human preference claim is made. Earlier full CPU regression predates
this additive report feature.

### 14:56 UTC — actual CPU Ray leader-death cleanup

Ran a separate local CPU-only Ray cluster under the fixed supervisor, created a
real actor, recorded eight descendant PID/start-time identities, then SIGKILLed
the driver. All eight were in the owned process group (GCS, monitor/dashboard
processes, raylet, dashboard/runtime-env agents and actor). Kernel fate-sharing
was available, but the group still needed bounded SIGKILL escalation after the
two-second grace. The second attempt observed no live original descendants and
completed successfully. Rechecked PID identities afterward; none remained live.
No global Ray stop or unrelated process termination was used.

Evidence: `outputs/qwen_image_21/ray_supervisor_cleanup_20260922_attempt2/`, including
process identities, empty survivors, provenance and successful result. Log:
`/tmp/vrl2-real-ray-cleanup-attempt2.log`. The first attempt failed before actor
creation because its Unix socket path exceeded 107 bytes; those records remain in
`ray_supervisor_cleanup_20260922/`, and the rerun used an owned short /tmp directory.
This validates this installed Ray/Linux process lifecycle on CPU, not model/GPU
continuation. The existing queued actual Qwen checkpoint comparison is unchanged.
Current training and both reward soaks remained live throughout the check.

### 15:28 UTC — bind reward requests to the validated service instance

Reproduced a real stale-handshake failure with two actual aiohttp servers bound
sequentially to the same port: the old client accepted a score from the replacement
model while retaining the original accelerator-isolation flag. The initial test
fixture omitted sample_id; after correcting that fixture, the before-fix run
failed because no identity error was raised (`/tmp/vrl2-service-instance-before.log`).

Wire version 7 now exposes a per-service UUID and requires its matching protocol
header before scoring, wake, park or cancellation touches model state. The client
pins the validated instance across connection-pool closure/reconnect; replacement
returns non-retryable service_identity_changed (409). A newly constructed and
preflighted client can use the replacement. Cancellation against another instance
cannot establish that old work stopped, so ambiguous shared files remain retained.
This is request-ownership binding, not authentication or durable distributed leases.
Client/service upgrades must be coordinated; existing frozen v6 experiments remain
on their original revisions, including the parked EditReward server.

Tests cover replacement with identical model labels but different isolation,
rejected old phase leases, missing headers, cancellation ambiguity, and fresh-client
recovery. Rewards CPU lane: 493 passed, 1 skipped, 13 deselected in 35.48 s
(`/tmp/vrl2-service-instance-rewards-tests.log`); service-only: 105 passed. Ruff and
diff checks passed on touched files. Earlier full CPU suite 5 completed with 4,836
passed, 17 skipped, 155 deselected in 398.27 s; it predates this instance fix.

Architecture scope: keep the existing transport facade, protocol dataclasses and
owner-thread boundary. The new ALL_CAPS header constant is an actual HTTP protocol
name in the protocol module. No new thin modules, business vocabularies or trainer
refactor are introduced. Existing model scoring/reward aggregation stays unchanged.

### 15:35 UTC — controller training complete; v7 restart soak active

RGB-observation controller training completed all 512 episodes successfully and
all 16 updates stepped with replay error 0. Verified checkpoint-16 SHA-256
`c0915bc4afd6e9abc0e69446c7747fef1bd527918a435a6e70e99365c9bc7239`
and progress episode_cursor=512 / next_update=16. Last update net return was
0.714054895 with 1.84375 mean tool calls. Last collection policy (version 15,
not final checkpoint 16) had initial stop probabilities 0.08237 already-done
and 0.05390 needs-edit; the tendency to keep editing remains a concern. Complete
training-state diagnostics were refreshed without reading held-out results.
The owned follow-up GPU queue started monitor-start evaluation at 15:33:01 UTC.

The third CPU service soak uses frozen cff89eea4, port 18319, driver PID 345380,
`outputs/reward_evaluation/service_instance_soak_20260922/`. It began 15:28:55 UTC
and runs to the existing 07:40 UTC deadline. First restart rejected the old
instance's scoring and ambiguous cancellation, then a fresh preflighted client
continued scoring. This is about 16 hours of v7 soak if completed, not 24 hours.
Earlier v6 soaks remain live on their own code. A new full CPU suite is running.

Queued alpha-controller source-paired analysis uses the same fixed resampling
protocol and baseline-integrity checks as RGB and also verifies matching
observation mode. Script: `outputs/qwen_image_21/analyze_alpha_controller_exploration.py`;
log `/tmp/vrl2-alpha-controller-analysis.log`. It waits for the existing owned
alpha queue; it cannot turn a partial run into a successful capability result.

### 15:36 UTC — full CPU suite after wire v7

Full frozen-environment CPU regression completed: **4,838 passed, 17 skipped,
155 deselected, 72 warnings in 380.22 s**. Log:
`/tmp/vrl2-production-cpu-suite6.log`. This includes the new real-HTTP replacement
and lease/cancellation checks; it does not replace queued real GPU recovery or
held-out controller evaluations. All three CPU service soaks remain healthy;
latest totals were 1,920 original, 1,573 auxiliary and 88 v7 successful requests.
The alpha-analysis watcher is PID 349439; its predecessor remains the owned
alpha GPU queue PID 300707. No user-owned sprint or third-party files were staged.

### 15:46 UTC — 512-episode controller monitor still fails the baseline gate

Both monitor evaluations completed (two sources, needs-edit/already-done states,
two seeds each = eight paired comparisons). Exact baseline scores, costs and
actual output hashes match across the initial and trained runs. Initial controller
net return 0.542484875 rose to 0.619901070, below random 0.622708408, fixed one-edit
0.803999857 and best-of-two 0.712853483. These are descriptive monitor results,
not a significance claim. The final 16-source evaluation began at 15:45:13 UTC.

Condition breakdown explains the limited aggregate gain. Needs-edit return rose
0.343075216 -> 0.521080538, with initial stop probability 0.22679 -> 0.04779.
Already-done return fell 0.741894533 -> 0.718721603; initial stop probability fell
0.28777 -> 0.07159 and mean edit calls rose 1.5 -> 1.75. This is increased editing
propensity with an unresolved stopping problem, not demonstrated conditional
control. The predeclared alpha-observation ablation remains unchanged.

Artifacts: `controller_rgba_exploration_monitor_{start,16}/result.json` and
`controller_rgba_exploration_conditions_monitor.json` under `outputs/qwen_image_21/`.
The condition diagnostic script accepts the RGB/alpha prefix and monitor/final
split, verifies completed traces and media hashes, and groups repeated seeds by
source. It should also be run for the final and alpha results once complete:
`outputs/qwen_image_21/summarize_controller_conditions.py`.

### 15:57 UTC — optional exact-target checks for masked RGB edits

Extended the existing masked_edit scorer with explicit `exact_target=true` mode.
It requires same-canvas target_image whose protected pixels equal the source;
contradictory targets fail. It retains inside target mean/max error and an optional
`target_match = 1 - max(outside_mean_error, inside_target_mean_error)` axis. Regions
are normalized separately. Default locality semantics, scalar selection, CPU
adapter and public reward boundaries are unchanged; existing target metadata is
not reinterpreted unless the mode is selected. No new TaskSpec layer, thin module
or workflow taxonomy was introduced. The existing thin adapter remains necessary
for standard model transport and CPU placement.

Three targeted tests passed, including nonzero training-scalar selection with
all diagnostic axes preserved, no-op/wrong-color/spill behavior, invalid target
geometry/region and already-satisfied targets. Ruff/diff checks passed. The previous
full suite predates this opt-in scorer extension.

Actual standalone CPU scoring processed 100 fresh procedural PNG fixtures across
20 sources, with no reused records; verified all candidate and auxiliary hashes.
All 20 exact targets scored 1; all 80 non-targets were lower. Mean target matches:
unchanged 0.6140, wrong-color 0.6090, inside noise 0.6519, outside spill 0.9393.
Noise exceeding some wrong-color candidates is an explicit limitation of pixel
distance; exact-oracle separation does not establish perceptual ranking quality.
This is a task-specific measurement, not a promoted general reward or evidence
that Qwen learned local recoloring. Evidence and frozen scoring recipe:
`outputs/reward_evaluation/exact_masked_edit_20260922/`, especially audit_report.json,
protocol.json and scores/. The 16-source train/four-source monitor manifests are
fixtures for later native-model qualification, not completed RL runs.

Separately requested actual user judgments for the existing six-pair armchair
review via asynchronous input. No answer has been received and no human labels
were fabricated; the single-source packet alone cannot satisfy broad calibration.

### 15:59 UTC — expose existing RGB decode in the native edit probe

The native Qwen edit probe now accepts --output-mode rgb/rgba (historical default
rgba). Both generation and optional LoRA probe requests forward the mode. Reports
verify channel count, retain alpha measurements only for RGBA, and compare RGB
against the same white composite of the official pipeline's RGBA tensor. This
exposes existing family behavior; no model architecture or training defaults
changed. CLI help and 12 relevant CPU family/input/reward tests passed
(`/tmp/vrl2-qwen-rgb-target-tests.log`). Actual RGB probe qualification is being
queued after the existing alpha-control experiment; it is not yet GPU-verified.

### 16:01 UTC — serialized exact-edit native qualification

Queued driver PID 372275 (`outputs/qwen_image_21/run_exact_edit_after_alpha.py`)
waits for owned alpha queue PID 300707 and its recorded start identity. It requires
all five alpha jobs complete and no unexpected GPU allocation before each job.
Frozen source is 18b890515 in `/tmp/vrl2-exact-edit-20260922`. State/protocol:
`outputs/qwen_image_21/exact_edit_queue_20260922/`; log:
`/tmp/vrl2-exact-edit-queue.log`. No new GPU work has started.

Predeclared jobs: two RGB official-pipeline comparisons, followed by 20 untrained
local recoloring sources at each of seeds 314159/271828. Comparison gates are
native replay error 0, RGB channel/canvas validation, mean official pixel error
<=1 on the 0..255 scale and fraction with channel error >8 <=0.01. Native baselines
use ordinary independent per-source seeds without the shared-latent comparison
override. Their outputs are rescored independently using the frozen exact-mask
recipe. These measure baseline behavior; they are not RL improvements or human
preference results. Failure stops this queue without automatic retry.

The existing final controller evaluation and all three HTTP soaks are confirmed
live. User-owned sprint edits remain untouched, and the human review question is
pending without fabricated or assumed answers.

### 16:18 UTC — real CPU controller action-order counterfactuals

Completed 32 forward passes with the actual Qwen3-VL controller: eight training
initial states, original/reversed action orders, checkpoint-start/checkpoint-16.
Used frozen 18b890515, CPU BF16 SDPA with two threads and a 900-second process
limit; no CUDA allocation, sampling, policy update or held-out tuning. The process
exited successfully. Asset/checkpoint identities and all rows are retained in
`outputs/qwen_image_21/controller_action_order_cpu_20260922/`; script:
`outputs/qwen_image_21/probe_controller_action_order_cpu.py`, log:
`/tmp/vrl2-controller-action-order-cpu.log`.

Mean initial stop probabilities:

| Training state | Initial original / reversed | Trained original / reversed |
| --- | --- | --- |
| Needs edit | 0.22892 / 0.17840 | 0.04752 / 0.09322 |
| Already done | 0.28379 / 0.20181 | 0.07351 / 0.11620 |

There is presentation sensitivity, and its direction changes with training, but
both orders still reduce stopping after training. Reversal alone did not recover
conditional stopping; this does not justify claiming either no positional effect
or a sole positional cause. Reversal changes both displayed order and letter
assignment, so those mechanisms are not isolated. CPU initial original-order
probabilities differed from stored GPU probabilities by at most 0.006405; this is
not a log-prob parity acceptance test. CPU work overlapped final-start evaluation,
so its wall times are descriptive rather than isolated performance measurements.

RGB train states have different visible images, but this does not expose actual
pixel alpha: an opaque white background and transparent white composite can look
identical in general. The existing predeclared alpha-observation ablation remains
the relevant next test; no action-reordering feature or new training recipe was
introduced based on these training-only diagnostics.

### 16:25 UTC — final initial-policy evaluation complete

Final-start completed at 16:24:37 UTC: 16 source groups, 64 comparisons, all
successful. Verified every one of the 320 method-output hashes. Source-balanced
net returns: initial controller 0.659480864, fixed one-edit 0.805295713, random
0.621977231, best-of-two 0.716365368, stop 0.499999996. Controller mean tool calls
were 1.3125. These establish the initial baseline; checkpoint-16 final evaluation
has just started and no paired final training-gain conclusion is available yet.
The frozen queue remains on 2050e6d6f. Main source changes and CPU diagnostics did
not alter its checkpoint, input, editor or reward recipe.

### 16:34 UTC — exact-edit CLI reuse and queued-input preflight

The frozen RGB queue's 20 tasks passed the actual prompt-dataset loader. Reference
images populate generation conditioning; exact targets and edit masks remain
scoring metadata. Exported a standalone `scorer.json` from the already-frozen
recipe (SHA-256 `2a6fe308730c27e907bb6aeb6f8d1026db419cb80a3304a3312cd3ca88c18964`).
The standard scoring CLI resumed the existing audit with 100 reused, 0 scored,
and the same run ID. Log: `/tmp/vrl2-exact-edit-cli-resume.log`. Published the
copyable command and explicit pixel-ranking limits in the reward baseline report.
No model work or new preference labels were introduced by this resume.

### 16:55 UTC — completion wording diagnosis and alpha-alias qualification

The real CPU input-contract probe completed 16 forwards on two training sources:
RGB/RGBA observations, original/explicit-stop task wording, both initial states.
Raw action-label token mass was at least 0.99999994; the output head is assigning
probability to the expected labels. Without the completion instruction, the
highest-probability action was edit in every case. With the same generic
instruction added to every task, it was edit for both unfinished states and stop
for both completed states, in both observation modes. At temperature 16, mean
stop probability for completed states changed from 0.283 to 0.589 in RGB and
0.317 to 0.551 in RGBA. These are two-source CPU diagnostics, not held-out GPU
validation or an RL improvement. Temperature-4 values in the summary are
analytic logit rescaling, not additional forwards. Artifacts:
`outputs/qwen_image_21/controller_input_cpu_20260922/`, script
`outputs/qwen_image_21/probe_controller_input_contract_cpu.py`.

Created `outputs/datasets/rgba_alpha_alias_20261001/` using the four existing
training targets and 20 fresh targets from seed 20261001 (4 monitor, 16 final).
Each target produces an opaque white-composited unfinished state and its exact
transparent completed state. Their controller-visible RGB composites are
pixel-identical; actual alpha differs. Both states receive identical task wording
and action menus, including the generic completion instruction. All target hashes
are unique across these splits and disjoint from previous controller evaluation
targets. IDs and oracle assets are not controller observations. Dataset builder:
`outputs/qwen_image_21/build_alpha_alias_tasks.py`; manifests, identities and limits
are retained in its protocol. This is an observation identifiability fixture, not
a claim of natural-image editing capability.

A further 16-forward CPU qualification on the first two alias training sources
has started (`controller_alias_cpu_20260922`, log
`/tmp/vrl2-controller-alpha-alias-cpu.log`). It compares both observation modes and
wordings without any update or held-out input. No new GPU training is scheduled
until this diagnostic establishes whether the proposed input actually exposes
the completion distinction. Existing frozen GPU queues are unchanged.

### 16:59 UTC — alpha-alias input qualification failed

All 48 new alias initial states passed the actual CPU RGBA-reference scorer:
unfinished states score 0 and completed states score 1. The real model's 16-forward
CPU probe then confirmed identical action probabilities for RGB aliases. More
importantly, adding alpha masks plus generic completion wording did not establish
correct stopping: both unfinished and completed states prefer stop. Mean stop
probabilities at temperature 16 were 0.5851 unfinished versus 0.5506 completed.
The original wording prefers editing in both. Thus the earlier easier-scene
wording improvement does not show alpha understanding. Full negative evidence:
`outputs/qwen_image_21/controller_alias_cpu_20260922/summary.json`; no new GPU
training was scheduled from this recipe.

A bounded follow-up now compares generic completion wording against an explicit
explanation of the alpha-mask completion condition, on all four training sources
only. The explanation is identical in both states and describes observable mask
semantics; it does not expose score, target image, state ID or oracle output.
Sixteen real CPU forwards, no model update, 900-second limit, frozen 18b890515.
Artifacts: `controller_alpha_semantics_cpu_20260922`; script
`outputs/qwen_image_21/probe_controller_alpha_semantics_cpu.py`, log
`/tmp/vrl2-controller-alpha-semantics-cpu.log`. All fresh monitor/final inputs remain
unqueried. This is training-set task-design diagnosis, not capability acceptance.

### 17:05 UTC — explicit alpha semantics also fails; isolate perception

The 16-forward four-source training diagnostic completed successfully as an
execution, but failed its behavioral qualification. Generic completion wording
prefers stop in all eight states (mean stop probability 0.5803 unfinished, 0.5428
completed at temperature 16). Explicit mask semantics instead prefers edit in
all eight (0.3721 unfinished, 0.3840 completed). Neither establishes conditional
stopping. All rows and the summary remain under
`outputs/qwen_image_21/controller_alpha_semantics_cpu_20260922/`.

Started a separate 16-forward classification diagnosis on the same four training
sources: identify a completely white mask versus a white circle on black, using
one alpha image or the existing four-view presentation. This removes the edit/stop
instruction and uses the same frozen initial controller. Correct labels are
computed from source pixels only for analysis, never included as model input.
No model update, oracle target input, GPU allocation or held-out inference.
Artifacts: `controller_alpha_perception_cpu_20260922`; script
`outputs/qwen_image_21/probe_controller_alpha_perception_cpu.py`, log
`/tmp/vrl2-controller-alpha-perception-cpu.log`. Existing GPU queues remain intact.

### 17:10 UTC — perception passes, final controller capability gate fails

The direct alpha classification diagnosis completed: 8/8 correct with a single
mask and 8/8 correct in the four-image layout, across all four training sources.
Actual CPU Qwen3-VL forwards, frozen initial policy, no updates or held-out inputs.
Together with the failed edit/stop probes, this narrows the fault to task/action
interpretation or policy learning rather than absent alpha pixels. It does not
prove correctness on natural images or a successful editing policy. Full rows and
summary: `outputs/qwen_image_21/controller_alpha_perception_cpu_20260922/`.

RGB controller final evaluation completed at 17:08:45 UTC: 64 paired comparisons
per checkpoint, 16 source groups. The independent watcher verified final media
hashes, matching settings and identical baseline media/scores across checkpoints.
Mean net return was 0.659480864 -> 0.693053224. Source-paired delta +0.033572359 has
a descriptive, unadjusted bootstrap interval [-0.017890100, 0.083477072], median 0;
7 sources improve, 5 regress, 4 tie. There is no robust demonstrated training gain.
Against fixed one-edit (0.805295713), all 16 source averages lose: mean gap
-0.112242490, interval [-0.145378984, -0.085687720]. The capability gate failed.

Condition analysis shows needs-edit return 0.528408 -> 0.658970, while already-done
return regresses 0.790554 -> 0.727136. Mean calls rise from 1.25 -> 1.84375 and
1.375 -> 1.78125 respectively. Initial stop probability falls to 0.04704 for
needs-edit and 0.07240 for already-done. More editing, not reliable conditional
stopping, explains the behavior. Artifacts:
`controller_rgba_exploration_analysis.json` and
`controller_rgba_exploration_conditions_final.json` under `outputs/qwen_image_21/`.
No recipe was retuned on these final results.

The frozen owned GPU queue advanced to `rgba_mixed_clipped32` at 17:08:45 UTC
(driver child PID 409844). It is loading the real training/rollout stack; the dense
arm and independent rescoring remain queued. Collection budgets are predeclared;
actual optimizer-step counts will be audited separately. All existing later GPU
jobs and three HTTP soaks remain active. No new GPU job was appended based on the
failed task-wording diagnostics.

### 17:18 UTC — rectangular native probe and local photo diagnostic fixtures

Extended the Qwen native/official edit probe with paired `--width` / `--height`.
Both paths receive the same explicit canvas and decoded channels/height/width are
validated together. Square `--resolution` behavior remains the default; reference
encoding has its independent resolution budget. The separate optional LoRA
backward smoke remains 256x256. No new modules, business constants, scheduler or
family adapter changes were needed; existing family interfaces remain uniform.

Seven focused CPU tests passed, then the actual tiny-transformer edit execution
test was strengthened to generate rectangular RGBA before resetting to square RGB
and passed again. Real large encoders/VAE remain fakes in that CPU test, as its
existing docstring states. CLI rejected missing height and non-multiple-of-32
height before model load. Changed-file Ruff passed. Logs:
`/tmp/vrl2-rectangular-probe-tests.log`, `/tmp/vrl2-rectangular-execution-test.log`.
Real-model rectangular native/official qualification is still pending.

Built `outputs/datasets/natural_edit_diagnostics_20260922/` from four existing local
photo originals and task text only: armchair material, sweater recoloring with
print preservation, farther-chair recoloring and left dining-chair recoloring.
Copied originals match their recorded hashes; no historical candidate images,
scores or labels were imported. Canvas sizes 320x480, 384x512 and 480x320 preserve
framing without cropping; originals are retained. Coarse manual allowed-edit
extents support outside-region change diagnostics only, not segmentation or a
semantic correctness oracle. These are existing diagnostic fixtures, not fresh
held-out sources. No human preferences or exact natural-image targets exist here.
Builder: `outputs/qwen_image_21/build_natural_edit_diagnostics.py`; protocol records
all source/mask identities and limitations. GPU work will be serialized after the
existing exact-edit queue, without competing with active training.

### 17:24 UTC — menu counterfactual rejected; rectangular photo queue preflighted

Completed 16 actual CPU forwards comparing legacy and conditional action menus on
four alpha-alias training sources. The menu counterfactual retained the same task
and images and performed no editor invocation. Both presentations prefer stop in
all eight initial states. Conditional-menu mean stop probabilities were 0.5926
unfinished and 0.5841 completed at temperature 16. Therefore no display-description
API or new GPU recipe was added as a supposed capability fix. Evidence:
`outputs/qwen_image_21/controller_action_menu_cpu_20260922/`; script
`outputs/qwen_image_21/probe_controller_action_menu_cpu.py`, log
`/tmp/vrl2-controller-action-menu-cpu.log`. Await the already-predeclared alpha
training experiment instead of retuning against final evaluation results.

The rectangular photo queue uses frozen f1b486e08 at
`/tmp/vrl2-natural-edit-20260922`. Visual preflight of all four actual source photos
found that two initial coarse extents excluded small intended garment/chair parts.
Stopped only the owned waiting driver PID 420750 after verifying no child or GPU
job existed. Its state is retained as `superseded-before-GPU`; no experiment was
restarted because of an observation timeout. Revised fixture v2 broadens the
sweater extent and correctly encloses the farther chair, while retaining explicit
coarse-region limitations. No generated outcomes were used to adjust these masks.

Active replacement: PID 423332, script
`outputs/qwen_image_21/run_natural_edit_after_exact_v2.py`, log
`/tmp/vrl2-natural-edit-queue-v2.log`, state
`outputs/qwen_image_21/natural_edit_queue_20260922_attempt2/`. It waits for exact-edit
owner PID 372275 and checks its process-start identity and successful completion.
Eight jobs are predeclared: portrait/landscape official parity, then three canvas
groups at each of two seeds. Each uses native 20-step RGB generation, reference
budget 512, pinned source and input hashes, and a free-GPU ownership check. Gates:
zero latent replay error, exact decoded dimensions, official mean pixel error
<=1 on the 0..255 scale and fraction with error >8 <=0.01. Failure stops the queue.

The v2 manifests passed the actual prompt loader: source images are conditioning,
edit masks stay reward metadata. Four unchanged controls were independently scored
through the standard scorer: locality 1 and both inside/outside change 0. Run ID
`60db5029127edd4da98d294ead9ca0c66ae79b1f4ce012281917e3d64a1f3b0e`.
Final queued scoring will cover eight real edits plus these four controls; semantic
judging/human review remain separate. Dataset:
`outputs/datasets/natural_edit_diagnostics_20260922_v2/`. Source photos, masks and
all manifests including unchanged controls are pinned. No old preference labels
or model outputs were imported. The original fixture and its control scores remain
historical preflight artifacts, not current evaluation inputs.

### 17:26 UTC — first mixed-opacity arm completed with 28 actual updates

The clipped reward arm completed at 17:25:02 UTC and the dense arm started next.
Verified checkpoint-32 SHA-256
`8f2583dab7bbce452c0ac2f2890bdf10c09f59aef0e9854f011df5b0c00157f2`
before loading its owned local state. Progress records 32 completed collection
epochs and 28 optimizer updates; all 256 parameters with Adam state have step 28.
The admission ledger contains 64 groups: 43 kept, 21 dropped; 256 collected samples
and 172 selected. Selection and optimizer application remain separately reported.
Audit artifact: `outputs/qwen_image_21/rgba_mixed_clipped32/completed_budget_audit.json`.
The scheduled paired budget watcher will independently report both arms after dense
training completes. Native generation/rescoring and any capability conclusion are
still pending. No extra updates were added to compensate after seeing rejections.

At 17:24 UTC all three owned HTTP drivers and service children were live, with no
failure artifact: 3,000 + 2,772 + 1,276 successful requests (7,048 total). Each
service had 11 descriptors and RSS about 552 MB. These are running CPU-oracle
soaks with controlled restarts, not a 24-hour or GPU-capacity acceptance result.

### 17:35 UTC — bind actual controller prompt content to checkpoint identity

Found a concrete restore-contract gap: the Qwen controller bound model/processor
files, temperature and observation mode, but not the fixed task/choice/alpha
prompt text used to construct new observations. Extracted the unchanged text to
`vrl/rollouts/visual_controller_prompt.json`. The Qwen factory's base identity v2
includes the validated exact template and Pillow version; it checks the template
before/after loading and supplies the captured value to the controller. Each
controller holds an immutable snapshot, so editing the asset cannot silently
change a loaded policy. A fresh controller with changed template content rejects
a mismatched checkpoint before applying parameters. Old base-identity-v1 runs
remain reproducible from their recorded frozen checkouts; the current factory does
not invent missing prompt provenance. Diffusion checkpoint contracts are unchanged.

Architecture: the change is a prompt configuration/identity boundary, not a new
action-menu API. The private validation method is shared by identity construction
and controller initialization. Composition, image preparation and lifecycle stay
in the existing controller; no model/trainer split, taxonomy table, scheduler or
cross-family interface cleanup was introduced. Static prompt data belongs in a
named packaged asset, while protocol schema keys remain explicit code boundaries.

Seven focused CPU tests passed after the final change, including unchanged restore,
changed-template rejection before parameter mutation, loaded-policy snapshot
stability, mutation during loading, real tiny-VL replay/gradients and existing
controller CLI resume. Log: `/tmp/vrl2-controller-prompt-contract-tests2.log`.
Changed-file Ruff and diff whitespace checks passed. A built wheel contains the
exact JSON asset and loaded it successfully from the wheel outside the worktree
(`/tmp/vrl2-controller-prompt-wheel/`, build log
`/tmp/vrl2-controller-prompt-wheel.log`). Frozen inexact sync for the test/lint,
cosmos and reward-service environment rebuilt only the local project; running
experiment dependencies were not replaced.

Actual local Qwen processor comparison against frozen 18b890515 produced identical
inputs and action token IDs in all 16 comparisons: eight training states, RGB and
RGBA modes. This executes real image/token preprocessing, not a VL model forward
or an RL update. Artifact:
`outputs/qwen_image_21/controller_prompt_input_parity_20260922.json`; script
`outputs/qwen_image_21/check_prompt_extraction_inputs.py`, log
`/tmp/vrl2-controller-prompt-input-parity.log`. Existing GPU queues use their frozen
sources and are unaffected. No held-out input or new prompt wording was used.

### 17:43 UTC — full CPU regression suite 7 passed

After the controller prompt-identity fix, full CPU regression exited zero:
4,841 passed, 17 skipped, 155 deselected, 72 warnings in 346.66 seconds. Command:
`HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/pytest -q -m 'not gpu and not distributed and not optional and not slow_test'`.
Log: `/tmp/vrl2-production-cpu-suite7.log`. This covers current source cf5cae0e2,
including exact masked target scoring, rectangular native-probe support and prompt
snapshot ownership; the later sprint edit is documentation only. Excluded GPU,
distributed, optional and slow lanes are not claimed as covered by this result.
The separate real-model experiments remain serialized on their recorded frozen
checkouts. Dense mixed-opacity training was confirmed live at optimizer step 30;
its completed budget and native quality results were not yet available.

### 17:44 UTC — dense arm complete; native evaluation started

Dense training completed at 17:42:43 UTC. The independent budget watcher finished
successfully and checked both checkpoint digests, 32 consecutive collection metric
rows, resume cursors and actual optimizer state. Dense checkpoint-32 SHA-256:
`1a3b2189c3eae2b7cda71251afad4925e638700a7e67357a5619788ca8121099`.
All 256 parameters with Adam state have step 32; all 32 metric rows have nonzero
gradient norm. Dense kept all 64 groups / 256 samples. Clipped kept 43 groups /
172 samples and made 28 updates; its zero-gradient collection epochs are 3, 7, 17
and 21. No post hoc budget equalization was performed. Full report:
`outputs/qwen_image_21/rgba_mixed_training_budget.json`.

The owned GPU queue has advanced to `rgba_mixed_eval_base_314159`, the first of
six native generation jobs (base/clipped/dense x two independent seeds, 100 sources
each). Initial outputs have the expected RGBA shape and zero native replay error.
All outputs must finish and be independently rescored before judging image gains.
This is a fixed-collection-budget reward comparison, not equal optimizer compute.

### 17:53 UTC — source-balanced reward comparisons by task category

Added optional `paired --stratify-by METADATA_KEY` to the independent reward
analysis CLI and `compare_paired_outputs`. The overall comparison stays intact;
each category receives the same source-balanced statistics, coverage, failed/missing
observations and directed intervals. This addresses a concrete diagnostic gap:
opacity subgroups in the running experiment previously reported only the clipped
terminal match, which can hide partial alpha/color progress or subgroup regressions.
Categories must be explicit strings, integers or booleans; missing metadata fails.
JSON types remain distinct, and paired task identity comparisons now preserve
those types rather than allowing Python's True == 1 equivalence. Strata may share
sources and intervals remain descriptive/unadjusted, not independent confirmations.

Five focused tests passed, covering repeated-source weighting, a declining stratum
hidden by the global mean, error/missing coverage, lower-is-better direction and
typed categories. Log: `/tmp/vrl2-stratified-reward-tests2.log`. The actual CLI
loaded four persisted natural-photo unchanged controls and retained all four source
strata with zero self-comparison deltas. This was explicitly a same-snapshot
plumbing check, not an independent model evaluation. Artifact:
`outputs/reward_evaluation/natural_control_strata_selfcheck_20260922.json`.

Also compared the default new API against frozen f1b486e08 on the existing real
100-source native RGBA confirmation outputs. Entire reports, including comparison
ID and bootstrap results, match exactly. Artifact:
`outputs/reward_evaluation/stratification_default_compatibility_20260922.json`.
No scoring model was loaded and no images or labels were generated by these checks.
Changed-file Ruff and diff checks passed. The implementation reuses the existing
comparison recursively; no new thin module, taxonomy or trainer coupling was added.
Full suite 7 predates this analysis-only feature; its targeted checks are reported
separately instead of claiming the full suite covered a later change.

### 17:56 UTC — freeze additional opacity-stratified axis analysis

Started an independent CPU watcher using frozen 1aa5f8644 at
`/tmp/vrl2-reward-strata-20260922`: PID 457190, script
`outputs/qwen_image_21/analyze_mixed_strata.py`, log
`/tmp/vrl2-mixed-strata-analysis.log`, protocol/state
`outputs/qwen_image_21/rgba_mixed_strata_20260922/`. It binds the existing scorer
PID 256149 to its process-start identity and waits for complete original scoring.
No scorer, generation setting, training recipe or GPU scheduling changed.

Before quality results were available, declared per-opacity reports for the same
five already-declared global axes: original match, dense match, alpha L1,
foreground-normalized premultiplied L1 and background alpha. Each pair compares
base/clipped, base/dense or clipped/dense, with two seeds and all five opacity
categories. It checks 100 source groups / 200 expected pairs globally and 20 / 40
per category, preserves missing scores, and requires every global report to match
the original frozen analysis exactly. Output will be
`outputs/qwen_image_21/rgba_mixed_stratified_axes.json`. These remain development
diagnostics with unadjusted intervals, not new confirmation experiments.

The first three native generation jobs completed successfully (300 images); the
second base-seed job is active. Completion of all 600 outputs and independent
scoring remain required before drawing a capability conclusion.

### 18:06 UTC — mixed-opacity results expose a transfer tradeoff

All 12 queued jobs completed at 18:03:34 UTC. The independent scorer and frozen
stratified analysis completed successfully: 600 generated outputs, 100 independent
source groups per arm, two native generation seeds and 20 sources per opacity.
All five global axes exactly matched the original frozen analysis. Evidence:
`outputs/qwen_image_21/rgba_mixed_experiment_summary.json` and
`outputs/qwen_image_21/rgba_mixed_stratified_axes.json`.

Original RGBA match: base 0.407737, clipped-trained 0.436086, dense-trained
0.458338. Source-bootstrap intervals for improvements over base are
[0.011322, 0.046950] and [0.015974, 0.080055]. Dense versus clipped improves the
mean by 0.022252, but its interval [-0.005035, 0.048110] includes zero.
Dense training improves the 40%/60% foreground-opacity strata substantially:
0.05928 to 0.27069 and 0.35707 to 0.53170. However, fully opaque sources regress
from 0.89175 to 0.76132, delta interval [-0.214548, -0.070793]. The dense reward's
own native-evaluation mean is nearly unchanged (0.404675 to 0.403872).
Background alpha and foreground-normalized color error improve globally, while
alpha L1's improvement interval includes zero. These are development diagnostics,
with unadjusted intervals and unequal realized optimizer counts (28 versus 32).
They do not establish a universally better reward or production replacement.
Fresh confirmation must preserve the opacity breakdown and both training arms.

The four-view alpha likelihood/gradient probe completed with exact replay,
1,916,928 trainable parameters, gradient norm 0.155560 and maximum parameter
change approximately 1e-5. The recorded action log probability did not change at
this update's numerical resolution; this is gradient plumbing evidence, not a
successful learned policy. The queued real GPU SIGKILL recovery experiment has
started and its uninterrupted control completed; interrupted/resumed comparison
is still pending.

### 18:08 UTC — producer-declared candidate digest

The standalone media manifest now accepts an optional lowercase SHA-256. A
mismatch rejects before scorer construction or output-directory creation, and
scoring rechecks around inference. Stress-manifest loading also respects the
producer digest, while transformed candidates do not inherit the source's digest.
Adding a matching digest preserves legacy evaluation run identity and cache reuse.
The Qwen generation probe records each PNG digest immediately after writing, plus
the optional official comparison PNG digest. Existing frozen experiments retain
their original source and do not retroactively gain generation-time provenance.

Validation: eight evaluation/stress tests passed, including cached resume after
adding a producer digest, replacement rejection without model construction or new
output creation, and changed-source rejection before stress output creation.
Changed-file Ruff checks passed. Evidence: `/tmp/vrl2-producer-digest-tests.log`.
This adds fields at existing manifest/report boundaries; it introduces no new
module, taxonomy or helper layer, and leaves generation numerics unchanged.

### 18:12 UTC — real GPU interruption recovery passes; fresh confirmation queued

The actual Qwen GPU recovery experiment completed successfully. Starting from
checkpoint-16, the control continued to step 18. The interrupted arm received
SIGKILL at its owned training leader PID 467198 after checkpoint-17 published;
the supervisor succeeded on attempt two. Recursive comparisons across model,
trainer (including optimizer), progress and RNG found no mismatch: 1,027 tensors,
one array and 1,942 scalars. Both final checkpoints report next/global step 18.
Evidence: `outputs/qwen_image_21/real_recovery_20260922/{injection,comparison,status}.json`
and `/tmp/vrl2-real-recovery-queue.log`. This is one real local-GPU interruption at
a complete checkpoint; it does not establish every failure timing or multi-node
recovery. The predeclared alpha-controller 512-episode training has now started.

Frozen a confirmation dataset at
`outputs/datasets/rgba_mixed_confirmation_20261010/`: 100 scenes, 20 per alpha
64/102/153/204/255, new generation seeds 161803 and 141421. All 100 oracle outputs
score exactly one on original and dense match. New sources have zero exact-byte
overlap against 2,284 prior unique source hashes; new targets have zero overlap
against 2,280 prior unique target hashes. This is distribution-matched synthetic
confirmation, not natural matting or independent training-seed replication.
Every scene is reserved for confirmation, including fixture files whose generator
uses historical train/eval filenames. No training or checkpoint selection uses it.

The owned queue PID 472374, session 25260, waits for natural-photo queue PID 423332
with its process-start identity pinned. It uses frozen 8f12aa811 at
`/tmp/vrl2-mixed-confirmation-20260922`, stops on failure, checks free GPU and
input/checkpoint hashes, then runs base/clipped/dense across both draw seeds.
Every generated candidate must pass native replay, shape, prompt/reference and
producer-digest checks before independent CPU scoring. All five declared axes and
five opacity strata will be reported for all three comparisons. No scalar-only
winner selection is permitted by the protocol.

Scripts: `outputs/qwen_image_21/{build_mixed_confirmation,run_mixed_confirmation_after_natural}.py`.
Protocol/state: `outputs/qwen_image_21/mixed_confirmation_queue_20260922/`.
Builder log: `/tmp/vrl2-build-mixed-confirmation.log`; queue log:
`/tmp/vrl2-mixed-confirmation-queue.log`. Final analysis will be
`outputs/qwen_image_21/rgba_mixed_confirmation_analysis.json`.
Any further GPU experiment must serialize after PID 472374, the new queue tail.

### 18:17 UTC — separate opacity correctness from combined RGBA score gains

Post-hoc pixel analysis of all 600 development outputs verified candidate and
target digests before reading pixels. Target-core alpha (pixels with target alpha
at least 95% of specified opacity) reveals that dense-trained outputs remain
nearly opaque: mean 0.92791 for requested 0.4 and 0.88587 for requested 0.6.
The corresponding base means are 0.99548 and 0.97888. Core straight-RGB error
falls 0.24782 to 0.13146 and 0.19918 to 0.14484. Thus improved combined scores
do not prove accurate requested transparency. For opaque targets, core alpha
falls 0.96489 to 0.85821 and RGB error rises 0.02154 to 0.10298. These are
correlated diagnostics, not a causal decomposition of the learned update.
Fixed first-source examples from the 40% and 100% strata were also inspected;
no image was selected based on its rank or used to tune the frozen experiments.
Evidence: `outputs/qwen_image_21/rgba_mixed_pixel_diagnosis.json`, script
`diagnose_mixed_opacity_pixels.py`, log `/tmp/vrl2-mixed-pixel-diagnosis.log`.

Commit e05de6707 adds the independent diagnostic `foreground_alpha_l1` to the
existing RGBA scorer: sum(abs(candidate alpha - target alpha) * target alpha)
divided by target alpha mass. Its [0,1] range is lower-is-better; transparent
padding cannot dilute it. Original training scalars and all nine existing axes
are unchanged. This is an extra measurement within the existing model boundary,
not a new environment or aggregation rule. No helper file or vocabulary was added.

Six relevant tests passed, including real reward-component transport and a case
where adding transparent padding reduces whole-canvas alpha error sixteenfold
but leaves foreground error at 0.6. Changed-file Ruff passed. Log:
`/tmp/vrl2-foreground-alpha-tests.log`.

Actual independent CPU rescoring completed under explicit v3 recipe/model/rubric
identities in fresh `outputs/reward_evaluation/rgba_mixed_eval_*_foreground_v3/`
directories. All 600 input records and all nine historical score axes match their
v2 values exactly; no historical record was overwritten. Report:
`outputs/qwen_image_21/rgba_mixed_foreground_alpha_analysis.json`; log:
`/tmp/vrl2-foreground-alpha-rescore.log`. The additional axis's mean error is
0.38833 base versus 0.38753 dense, with directed-improvement interval
[-0.02296, 0.02090]. At requested 40% it improves 0.59093 to 0.53729, still very
large; opaque-target error worsens 0.04153 to 0.14992. This substantiates the
tradeoff instead of declaring transparency solved from the combined mean.

This new endpoint is explicitly post-hoc. The already-frozen confirmation retains
its original five axes and source commit; no training or confirmation protocol was
changed after inspection. Alpha-controller child PID 473084 remains live on the
GPU and is writing completed episode records; no restart was attempted.

### 18:22 UTC — queue independent learned scoring for natural-photo edits

Prepared a serialized real EditReward diagnostic after the fresh mixed-opacity
confirmation queue. Driver PID 479910/session 33419 binds predecessor PID 472374
and process start 161372203. It uses frozen 488793075 at
`/tmp/vrl2-photo-reward-20260922`, with the existing separate Transformers 4.57
EditReward environment. The v7 service CLI imports and help command succeeded in
that environment without loading a model. Upstream EditReward remains clean at
77a93aaa461fe9187e0ff841b59ecc0d0620bb7f. Model/base revisions are pinned in the
copied configuration; offline mode prevents downloading new weights.

Once all existing generator jobs finish and the GPU is free, the queue will
launch its own temporary service on port 18320. The original parked service PID
123746/port 18315 remains untouched. The new service must advertise wire v7 and
the expected model identity. It scores all eight native photo edits and four
unchanged controls against the exact inputs already measured by the locality
scorer, verifying candidate and auxiliary-file digests first. It then reports
learned EditReward rankings alongside locality, outside RGB error and inside
changed fraction, retaining all raw axes. It does not turn learned predictions
into human labels or combine the opposing metrics into a promoted scalar.

There is one actual scoring pass; repeated HTTP cache hits will not be reported
as independent uncertainty observations. Only the created child is shut down in
cleanup, with clean process exit required for success. Any error stops the queue
without an automatic retry. Source/target copies remain local, and no paid APIs
are involved. The four photos are known diagnostic sources, not an independent
holdout or evidence of RL improvement.

Script: `outputs/qwen_image_21/run_photo_reward_after_confirmation.py`.
Protocol/state/config: `outputs/qwen_image_21/photo_reward_queue_20260922/`.
Log: `/tmp/vrl2-photo-reward-queue.log`; expected report: that state's
`analysis.json`. Future GPU jobs must serialize after PID 479910, the new tail.
At this checkpoint the queue is confirmed live and waiting, not already scored.

The alpha controller has written 36 completed episode records and its GPU child
remains live. All three service soak children were confirmed live with no failure
artifact: original 357 cycles/3,570 requests; auxiliary 309/3,399; v7 instance
173/1,903. Each has 11 descriptors and approximately 552 MB RSS. These running
checks are neither final soak acceptance nor GPU-serving throughput claims.

### 18:27 UTC — audit real controller traces and correct completion counting

Read-only snapshot audit passed for all 512 completed RGB-controller episodes
(933 decisions, 770 edit calls, 16 published updates) and 55 successful alpha
controller episodes (92 decisions, 64 edit calls, one published update). One
additional alpha trace was still running and was explicitly excluded. Earlier
updates that called all existing episode files "completed" were too strong:
those counts came from filenames and may include an in-flight trace. Use trace
status and published update records for completion evidence going forward.

The audit uses the existing training-admission validator to check task/media
lineage, action vocabulary, policy identity, terminal reward, causal returns,
budget and termination. It also verifies every replay-file SHA-256 and recorded
selected-action probability, checks categorical normalization, and independently
checks the undiscounted identity terminal score minus actual edit costs. All
published metrics match complete 32-episode groups and exact likelihood replay.
This checks real persisted data; it is not a new forward pass, independent proof
of the admission validator, or evidence of policy quality.

The first audit attempt failed before publishing a report because the older RGB
settings lack the later observation-mode field. Its frozen 2050e6d6f preparation
code explicitly uses two RGB white composites. The corrected audit records
"legacy-rgb-no-mode-field" rather than pretending that the field existed; the
alpha settings explicitly record rgba. No experiment or model was restarted.

Evidence: `outputs/qwen_image_21/controller_episode_snapshot_audit_20260922.json`,
script `audit_controller_episode_snapshots.py`, log
`/tmp/vrl2-controller-episode-audit.log`. These are snapshot counts while alpha
training continues. No source implementation change was needed by this audit.

### 18:35 UTC — full CPU suite 8 passes after reward diagnostic additions

The frozen-lock dry run reported no environment changes using
`uv sync --dry-run --frozen --inexact --group test --group lint --extra cosmos --extra reward-service`.
Then full CPU regression at a99655326 completed with exit zero:
**4,844 passed, 17 skipped, 155 deselected, 72 warnings in 409.72 seconds**.
Command: `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/pytest -q -m 'not gpu and not distributed and not optional and not slow_test'`.
Log: `/tmp/vrl2-production-cpu-suite8.log`.

This run includes the new metadata-stratified comparisons, producer-declared
media digests, and foreground-alpha diagnostic added after suite 7. It does not
replace GPU/distributed/optional/slow acceptance. No source edits were made while
this suite ran, and the user's outstanding sprint/untracked files remain intact.
The independently frozen alpha GPU training process stayed live and continued
publishing successful episodes during the test. All predeclared generation and
reward queues remain scheduled; no experiment was restarted for the CPU suite.

### 19:20 UTC — alpha training midpoint integrity, with live service soaks

The frozen alpha-observation controller reached 8/16 real optimizer updates and
256 episodes admitted to those updates. Every checkpoint-1 through checkpoint-8
was read and matched its declared SHA-256, rgba observation mode and exact
32-episode cursor increment. All eight metrics report an actual optimizer step
and zero likelihood replay error. The process remains live at PID 473084 and is
collecting the next batch (259 successful traces plus one in flight at snapshot).
No parameters, prompts, tasks, sampler settings or evaluation protocol were
changed during this run. Policy capability remains pending the predeclared
monitor/final baseline comparisons.

Snapshot artifact: `outputs/qwen_image_21/controller_alpha_midpoint_audit_20260922.json`.
It also records the three live service children and their current summaries:
original 4,150 successful requests / seven restarts; auxiliary 4,037 / seven;
v7 instance 2,541 / four. No failure artifact was present. Their starts differ,
so these counts are not a 24-hour soak or an aggregate throughput benchmark.

The intervening goal turns used verified waits on live session 72503 and checked
real episode progression. No stalled or timed-out observation was treated as a
reason to restart a healthy experiment. The requested minimum work duration has
not elapsed; earliest completion remains 2026-09-23 07:36:41 UTC, subject to the
full acceptance scope being proven.

### 19:38 UTC — minimum-duration midpoint; tenth alpha update verified

The goal has run for just over 12 hours since 07:36:41 UTC. The required minimum
24-hour duration is not complete. The frozen alpha controller has published
checkpoint-10 with episode_cursor 320 and next_update 10. Its file digest matches
the manifest; the tenth metrics record reports an actual optimizer step and zero
replay error. Six training updates and all four controller evaluation jobs remain
before this ablation can support a capability conclusion.

The original scope remains active: finish that evaluation, qualify RGB and
rectangular native editing, run fresh mixed-opacity confirmation, compare natural
photo semantic/locality scores, and finish the service soaks. Earlier narrow
opaque-RGBA gains and successful real crash recovery do not substitute for these
remaining requirements. No experiment was restarted or recipe changed during the
verified waits. The code remains at the full-suite-8 tested implementation plus
evidence-only documentation commits.

### 20:35 UTC — alpha training complete; paired evaluation started

The frozen alpha-observation ablation completed all 512 successful episodes and
16 actual optimizer updates. All 16 checkpoint digests, rgba observation modes,
32-episode progress increments and metric policy versions were checked together.
Every update reports zero likelihood replay error. The audit is retained at
`outputs/qwen_image_21/controller_alpha_completed_audit_20260922.json`.

The queue advanced without intervention to the initial-checkpoint monitor
evaluation; its real child process PID 584208 was observed running. Four paired
evaluation jobs remain. Training completion does not establish a capability gain
or repair the earlier RGB controller's failed fixed-policy comparison. Subsequent
RGB, rectangular-photo, mixed-opacity confirmation and learned-reward queues
remain serialized. The minimum 24-hour goal duration has not elapsed.

### 20:47 UTC — alpha monitor improves over initialization but fails baseline gate

Both monitor evaluations completed eight task/seed pairs across two sources.
Controller net return increased from 0.4689246 to 0.6199011, still below fixed
one-edit 0.8039999 and random 0.6227084. All four baseline methods' final media
digests and scores match exactly across checkpoints. Condition diagnostics checked
controller artifact hashes, trace returns and task/seed pairing, and are retained
in `outputs/qwen_image_21/controller_alpha_exploration_conditions_monitor.json`.

Needs-edit return improved 0.19595 to 0.52108 while already-done return regressed
0.74189 to 0.71872. Already-done mean edit calls increased 1.5 to 1.75; initial
stop probability fell 0.3261 to 0.1388. This small monitor does not establish a
reliable stopping policy. The predeclared final evaluation started at PID 593768;
its result remains pending. No prompt, checkpoint or reward was selected or
changed in response to these monitor outcomes.

### 21:08 UTC — user-directed scope correction; useful text-region reward shipped

The user challenged repeated progress polling and requested broader production
reward/agent editing work, explicitly including sequential editing and dense text
such as manga. Subsequent work prioritizes usable reward and editing interfaces;
the existing frozen GPU queue continues, but new synthetic training is not added
merely to occupy the minimum duration. The expanded, requirement-based work plan
is `docs/sprints/planned/SPRINT_production_visual_editing_and_text.md`.

Source inspection found the legacy image OCR substring shortcut unsuitable for
region-bound dialogue. A separate `text_regions` model and standard reward adapter
now accept an explicit canvas and text boxes, retain case/punctuation/extra words,
and report equal-region similarity, worst-region similarity, exact fraction and
empty-recognition fraction with structured transcripts. Existing OCR semantics
remain unchanged. The adapter preserves the common Ray/HTTP and CPU placement
boundary instead of adding another scoring service.

Focused verification: 59 tests passed, one existing test skipped; Ruff passed on
touched Python files, with a frozen-lock dry run reporting no environment changes.
A real cached CPU PaddleOCR audit used a four-region English page, with correct,
swapped, extra-word and missing-region variants. Correct similarity/exact fraction
were 1/1; swapped 0.59184/0.5; extra words 0.97072/0.75; missing 0.75/0.75 with
worst-region similarity 0 and empty fraction 0.25. The standard independent
`rescore_media` path rescored all four; every axis and structured diagnostic
matched the direct real OCR execution. Evidence and recipes are under
`outputs/reward_evaluation/text_regions_audit_20260922/`; standalone run ID is
`21769648d99218a5b6212f063f6f18a90b677bb2aa0eb7e42f939e3abdde2676`.

The rendered fixture tests the verifier, not Qwen-generated manga, preference
quality, multilingual recognition or a trained planning policy. Reading order,
speaker attribution, outside-region text and aesthetics remain separate gaps.
Sequential task specifications and frozen-calibration-to-training integration are
still pending; they are not marked complete by this new component.

### 21:12 UTC — text specifications enter visual tasks and replay contracts

Visual tasks now carry an optional immutable `text_layout`, with a shared schema
boundary in `vrl/rewards/text_spec.py`. Region tuples and their nested values are
frozen, caller-owned input dictionaries are detached, and empty specifications
preserve legacy task identities. The judge and exported evaluation manifests
forward the specification; controller episode reconstruction restores it so a
changed transcript fails the observation/replay identity check. Required dialogue
still needs to be present in editor/controller instructions, rather than assuming
that verifier-only metadata is visible to the policy.

Region similarity and exact-match axes now survive ordinary reward aggregation
and visual scoring. A single-frame C,T,H,W collector payload is accepted as an
image; multiple frames remain rejected. Fourteen focused episode, adapter, replay,
credit and text tests passed, including a full fixture episode through the judge
and rejection of modified transcript evidence. Ruff and diff checks passed.

Real cached CPU OCR also ran through `RuntimeVisualJudge` over the rendered states
missing -> correct -> extra -> correct. Aggregate scores match the earlier
independent-file execution exactly, while the first region's exact-match axis
tracks 1 -> 1 -> 0 -> 1. Artifact:
`outputs/reward_evaluation/text_regions_audit_20260922/visual_judge_v2.json`.
This establishes transport/specification integrity, not generated sequential
editing capability. Staged task execution, preservation reporting and the frozen
calibration bridge remain active work.

### 21:18 UTC — independent sequential preservation report verified

`analyze_scores sequence` now consumes an existing scoring snapshot and an
explicit ordered sample/requirement specification. It reports first satisfaction,
adjacent measured regressions/improvements, final status and coverage, including
requirements activated at later steps. Missing axes and failed/missing scores
remain unknown. Prompt, metadata and auxiliary identities must stay fixed across
states; changing the expected answer is rejected rather than called preservation.
The report hashes criteria and observed results. No threshold enters training.

Six targeted sequence/diagnostics tests passed, including repaired-final-state
regressions, lower-is-better criteria, staged requirements, unknown intervals and
changed task rejection. Real cached CPU OCR rescored four rendered states through
the standalone scorer under a new per-region-axis recipe, then the public CLI
reported panel-3 first satisfied at step 1 and panel-1 regressed at step 2 and
repaired at step 3. Report:
`outputs/reward_evaluation/text_regions_audit_20260922/sequence_report.json`, ID
`2b4ccde042cf4f7d6328e7eea0ec83c52729001ff437ee5ce306d82dfa638974`.

This is a verifier/tooling check over rendered fixture states, not evidence of
Qwen generating those steps. The report deliberately distinguishes declared
ordering from verified tool input/output lineage. Episode-to-rescoring export,
real staged editing and frozen-calibration integration remain pending.

### 21:24 UTC — actual Qwen episode export and independent state rescoring

The new `export_episode_media` command validates successful episode task/replay,
image-lineage, budget, termination and causal-return contracts before exporting
the original plus each actual edit. Media hashes, original reference/verifier
assets, text specifications, parent relationships and recorded policy identities
are retained. Stop does not duplicate a state. A fresh directory and final
provenance marker distinguish complete exports; files are rechecked before the
marker. Original media remains in place and must still match at rescoring time.

The existing validator moved from the trainer to the episode owner so this
independent tool does not construct/import the controller trainer. Training uses
the same function and retains its existing import surface. The thin CLI is a
public execution boundary; no algorithm, family adapter or alternate service was
introduced. Eleven focused export/episode/credit/sequence tests and Ruff passed.

Two existing real-model traces were exported: the two-edit natural armchair
episode (`outputs/reward_evaluation/armchair_episode_export_20260922`) and a
two-edit RGBA episode (`outputs/reward_evaluation/rgba_episode_export_20260922`).
The latter was independently CPU-rescored from its actual generated PNGs:
0 -> 0.8838185285 -> 0.8548785786. Its second edit reduces the original objective;
the exact-reference criterion is not satisfied. No new generation was performed.

A deliberately tight 1e-12 score-equality check exposed an actual transport
precision boundary: direct PNG scoring divides uint8 by 255 in float64, while the
historical visual judge first creates unit-range float32 tensors. Maximum scalar
difference was 2.6605321e-9. Reconstructing the historical float32 path reproduced
all three recorded episode scores exactly. Both results and the failed exact-file
equivalence assumption are preserved in `score_precision_audit.json`; no scorer or
historical experiment was changed to hide this difference. This is reward-input
precision, separate from diffusion/controller likelihood replay gates.

### 21:32 UTC — shared frozen reward arithmetic (not yet deployment admission)

Offline application/evaluation and `MultiReward` now share an immutable, digest-checked
`FrozenRewardCombination`. Runtime aggregation retains original axes and signed
per-axis contributions, rejects missing axes/nonfinite arithmetic/ambiguous component
weights, and does not fit batch statistics. Aggregation configuration is checked
before constructing scorer resources. Related calibration and composite tests: 48
passed; touched-file Ruff passed. Fixtures validate arithmetic, not preference quality.
The public YAML/deployment binding is still pending: supplying a recipe dictionary
alone does not prove the actual runtime scorer/preprocessing matches calibration.

### 21:42 UTC — qualified frozen rewards reach the training factory

Added explicit `reward.calibration` receipt pin and `calibrate_scores qualify`.
Qualification uses the existing HTTP model adapters on float32 image tensors,
checks raw axes against the exact offline service recipe with declared tolerances,
rechecks media/auxiliary hashes, and saves all measured pairs. Training validates
receipt/configuration identity before client construction; original scores and
signed contributions remain visible. Initial support is HTTP + explicit RGB/RGBA
float32 C,1,H,W images, not video/Ray/object-store boxes. The offline recipe may
use files: the receipt measures that representation change rather than asserting
bitwise equivalence. Model-version labels are still operator declarations.

Validation: 370 calibration/composite/config tests, 27 factory tests, and 49
calibration/deployment tests passed (overlapping sets, not additive); touched-file
Ruff passed. A real CPU RGBA HTTP test demonstrated strict-zero parity rejection,
explicit-tolerance success, online factory equality, altered config/artifact
rejection, input dtype enforcement, and changed auxiliary-file rejection.

Real saved Qwen episode audit (three states, no new generation):
`outputs/reward_evaluation/calibrated_deployment_20260922/report.json`.
The public qualification CLI and online factory both ran. Maximum raw drift was
3.424506456e-9 on rgba_dense_match and 2.399007798e-10 on foreground_alpha_l1;
explicit atol=1e-7, rtol=0. Factory batch-of-three exactly reproduced the separate
qualification calls' frozen totals: 0.0005276463, 0.8230553982, 0.7784966682.
Receipt 1171fd4cf99f69f0543c0d32150a90111353e360523be9be53c6a83ec0d06b21.
This manually specified objective validates wiring, NOT human preference fit or
RL improvement. Its temporary CPU service was closed; the receipt is audit
material, not a live production endpoint.

Three existing CPU soaks remain healthy at this check: base 5,550 successful
requests / 10 restarts; auxiliary 5,577 / 9; instance-binding 4,081 / 7, with all
seven stale instances rejected. Each live child remains around 552 MB and 11 FDs.
These are periodic correctness probes, not load/throughput measurements.

### 21:45 UTC — prevent alternate entrypoints from dropping calibration

Visual collection/controller-training sessions now load the same pinned deployment
before reward construction. A missing receipt fails before controller allocation.
The separate RGB checkpoint-evaluation policy rejects calibration explicitly until
its input contract is qualified; it previously projected only three reward keys,
which would silently drop the newly introduced field. Existing visual reward
session tests passed (2); evaluator regressions checked separately.

### 21:50 UTC — sequential edit baseline with explicit preservation deadlines

Added a fixed-order controller protocol adapter and `visual_sequence` executable.
It executes existing named edit actions on the preceding output, with a bounded
budget and the existing park/wake handoffs. It avoids loading a learned controller
for a scripted baseline. A separate plan states when final-task requirements
become due; reports use recorded per-state judge axes, preserve unknowns, and
export all media/lineage for independent rescoring. Terminal rewards remain paid
once. This is neither learned planning nor interrupted-episode recovery.

Thirteen focused episode/sequence/export/session/comparison tests passed. A
purposefully defective two-stage image tool loses its first achievement during
the second edit: the report catches that regression and its failed final criterion,
while causal credit remains exactly terminal score minus both tool costs.
Invalid action plans fail before allocation; failed tools leave no success report.
Also: the earlier RGB checkpoint-evaluator regression suite passed 21 tests.

### 21:55 UTC — text verifier fails its own narrow manga-layout oracle

Prepared a four-panel 768x1024 page with 81 English words and the sequence:
create artwork/dialogue → recolor a coat → correct north to south. Two seeds,
three edits each, plus one same-canvas official replay/parity probe are queued
behind the previous final GPU owner (photo reward PID 479910). New queue owner:
666055; frozen source 139d4cab7; state `manga_sequence_queue_20260922`.
No new training was queued.

Before generation, actual cached CPU OCR on known-correct rendered controls
failed exact matching: font sizes 20/22/24/26 yielded exact-region fractions
0.25/0.50/0.75/0.25. Incorrect OCR strings still had high line confidence.
A hypothesis that balloon outlines caused the error was tested by keeping the
same pixels and using inset text bounds: fractions 0.25/0.75/0.25/0.00; it did
not solve the problem. Reports, recognized strings and images are retained in
`outputs/datasets/manga_sequence_diagnostics_20260922/{ocr_control,font_audit,text_bounds_audit}.json`.
No transcripts or thresholds were changed to manufacture a pass.

The queued Qwen sequences remain useful diagnostic media, but OCR exact failures
cannot be attributed to the generator without image inspection. The queue
protocol explicitly disallows claiming OCR reward is admitted for RL. It will
retain all intermediate images, independent OCR reports and a contact sheet;
coat correctness, character identity and story quality remain unverified.

### 21:59 UTC — repair the independent reward import boundary

CPU suite 9 at source 139d4cab7 finished: 4,856 passed, 17 skipped, 155 deselected,
72 warnings, one failure in 399.54s. The failure correctly caught rollout-specific
`export_episode_media` importing episode admission from inside `vrl.rewards`.
Moved that complete export operation to `vrl.rollouts.episode_export`, updated all
callers, and kept model-free score sequence analysis in the reward layer. The CLI
remains a public adapter, and the exporter remains an operation/ownership boundary;
no generic adapter flattening or constant cleanup was needed. Fifteen focused
architecture/export/sequence tests now pass. Log `/tmp/vrl2-production-cpu-suite9.log`
retains the original failed full run; no claim that it was fully green.

### 22:03 UTC — stronger cached OCR passes positive and negative controls

Investigated an already cached PP-OCRv6 medium detector/recognizer on the exact
same pixels. Plain CPU inference with oneDNN disabled recognized all 16 regions
across the four font-size controls exactly. Added an explicit opt-in
`text_regions` backend (`ocr_backend=paddle_v6_medium`) requiring a local model
root plus all six graph/config/weight file SHA-256 pins. Files are checked before
and after model construction and pins travel in structured diagnostics. Default
v4 semantics and the existing frozen manga GPU queue are unchanged.

Actual standalone scoring with the implemented pinned backend covered ten images:
four correct renders plus wrong-word, swapped-bubble, missing-bubble, extra-word,
repeated-word and blank controls. All 16 positive regions passed exactly; all six
negative cases failed in their intended affected region. The wrong-word page
still scores 0.99519 on aggregate similarity, illustrating why aggregate text
similarity cannot replace region-level exact constraints for word correction.
Report: `outputs/reward_evaluation/text_regions_v6_audit_20260922/report.json`;
run ID 47ccbf4909d508cef051895db0e6e88d3487327df55a7974f6f2f0d7e8c0d4c2.
No new models were downloaded. This is controlled English OCR improvement, not
human preference calibration, generated-manga success, multilingual reliability
or RL improvement. Four text-region tests pass, including replacement of model
files before/during loading; touched-file Ruff passes.
