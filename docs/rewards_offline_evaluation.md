# Score existing media independently

> Scoring (`vrl.scripts.rewards.rescore_media`) is part of the framework. Analysis,
> calibration fitting, qualification receipts and review packets live in the
> separate `reward_lab` package, which depends on `vrl` and is never imported by it.

`vrl.scripts.rewards.rescore_media` scores a JSONL media manifest through the
existing local or HTTP reward scorer. It does not construct a trainer, generation
worker, optimizer, or Ray cluster. It preserves every score axis rather than
choosing a training reward. Calibration and candidate comparisons consume these
raw measurements separately.

## Local CPU example

Create a manifest; relative paths resolve beside the manifest:

```json
{"sample_id":"sample-1","prompt_id":"prompt-1","prompt":"A red chair","path":"chair.png","metadata":{"seed":101}}
```

Optional `assets` maps scorer metadata keys to reference/mask file paths. The
files are resolved and hashed, then their absolute paths are passed in metadata.
These keys must not also appear in `metadata`. All external file dependencies
must be declared as assets to participate in resume identity.

Optional `sha256` declares the expected media digest (64 lowercase hexadecimal
characters). Record it when producing the candidate, then copy it into the scoring
manifest. A mismatch fails before constructing a scorer or creating the output
directory, and scoring rechecks the digest around inference. The Qwen Image 2.1
edit probe records `sha256` for each generated PNG and `official_sha256` for its
optional comparison. Legacy manifests remain supported; their first scoring run
binds the bytes present at scoring time, without proving generation-time identity.
Adding a matching digest does not change an existing evaluation's run identity.

Create a standalone YAML (not a training preset):

```yaml
name: sharpness-audit
revision: image-sharpness-code-v1
preprocessing_revision: native-middle-frame-v1
rubric_revision: laplacian-energy-v1
worker_config:
  model_factory: vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel
  reward_model_version: image-sharpness-code-v1
  device: cpu
media_mode: file
batch_size: 2
```

```bash
python -m vrl.scripts.rewards.rescore_media \
  --manifest media.jsonl --config sharpness.yaml --output-dir outputs/sharpness-audit
```

This measures high-frequency edge energy. It is a pipeline check, not evidence of
semantic correctness or overall visual quality; noise can also score highly.

## HTTP scoring

Use an existing operator-owned reward service. Replace `worker_config` with:

```yaml
inference:
  kind: http
  endpoint: http://127.0.0.1:8300
  expected_model: deployed-reward-name
  expected_model_version: pinned-reward-version
  timeout_s: 1800
media_mode: tensor
```

`tensor` decodes RGB media into a channel-first float tensor and uses the existing
checked upload protocol. It uploads all decoded frames; use bounded clips/batch
sizes to control CPU memory. This mode does not preserve alpha. `file` leaves
original files intact and requires service-allowed shared filesystem paths.
Auxiliary assets in metadata always require shared paths for HTTP scoring;
they are not automatically uploaded.

The service also confines built-in file metadata (`reference_image`,
`reference_images`, `target_image`, `target_video`, `edit_mask`) to absolute paths
within its configured `artifact_roots`, including when the candidate is uploaded.
It hashes these files before and after scoring, and rechecks them on a cached
success. A changed target/reference/mask rejects that request instead of returning
an old score; use a new request identity for intentionally changed inputs. These
server-owned digests supplement the caller's declared asset provenance. They are
before/after checks, not a filesystem snapshot against concurrent adversarial
rewrites. Custom rewards consuming additional file metadata must extend the
`RewardInferenceArtifact.metadata_file_paths` schema contract. The HTTP payload
format is unchanged; restart an existing service to apply this server fix.

Some image models consume tensors via `artifact.as_media()` and need tensor
mode; file-aware video/image scorers can use file mode. Declare preprocessing
explicitly and compare equivalent preprocessing when evaluating transports.

## Evidence and resume

The output directory contains:

- `provenance.json`: full recipe, input paths, prompt/metadata, media/asset hashes,
  and the run fingerprint.
- `samples/<sample-id-hash>.json`: original sample identity, status and every
  named score, model version and scorer timings. Failed batches contain error
  records, never fabricated zero rewards.
- `summary.json`: published only after all samples complete successfully.

Add `--resume` to reuse successful records and retry failures. Changed media,
references, metadata, rubric or recipe refuse resume; use a new directory.
The declared revision fields are operator provenance: pin actual model weights,
code and preprocessing, and change revisions when those change. A revision
string is not an automatic attestation of model bytes.

On the local POSIX filesystem, an exclusive directory lock covers scoring and
shared locks cover complete snapshot reads. Concurrent writes (or reads during a
write) fail promptly. The kernel releases ownership even after SIGKILL, so a
committed partial run can resume without deleting lock files. The persistent
`.writer.lock` contains a protocol marker, not live ownership; do not remove it.
It also prevents older sentinel-only writers from modifying these directories.
Legacy unversioned lock files remain an explicit error because their ownership
cannot be inferred safely. Remote in-flight scoring may outlive a killed client;
resuming can repeat that computation, but the dead client cannot publish results.
This tool neither stops another process nor takes over a live writer's directory.
Filesystems without the required directory-lock semantics fail rather than provide
this guarantee. A kill before initial provenance publication may require a fresh
output directory; no completed samples exist to reuse at that stage.
HTTP deadlines/cancellation use the existing client; a blocking in-process model
cannot be forcibly interrupted by an asyncio timeout. Use an isolated service
when enforceable process responsiveness is required.

Historical training scores are never edited. Each candidate evaluation gets
its own directory. Versioned raw scores can later support calibrated aggregation
without rerunning compatible model inference.

## Health and candidate ranking comparison

```bash
python -m reward_lab.scripts.analyze_scores health outputs/sharpness-audit \
  --output outputs/reports/sharpness-health.json
python -m reward_lab.scripts.analyze_scores compare outputs/candidate-a outputs/candidate-b \
  --first-axis quality --second-axis overall \
  --output outputs/reports/candidate-rankings.json
```

Health reports preserve failures/missing records, per-axis coverage, scorer timings,
model-version counts, distributions, and per-prompt ranges. Zero spread means no
ranking signal for that axis, not proof that a task is too easy or too difficult.
Raw distribution statistics describe samples; prompt-balanced means/intervals treat
prompts as units. A single prompt receives no confidence interval.

Candidate reports require identical input records and complete scoring; they never
silently intersect away failures. They compare within-prompt pairs, give each prompt
equal weight, and retain reversed rankings and tie disagreements for review.
`--first-direction -1` and `--second-direction -1` support lower-is-better axes.
Tie tolerances are explicit in each score's units. Agreement is consistency between
scorers, not correctness or human preference accuracy.

## Frozen preference calibration

A preference JSONL names two scored sample IDs, their shared source group and an
explicit split:

```json
{"pair_id":"annotation-1","left":"sample-a","right":"sample-b","source_group":"original-scene-1","split":"calibration","preference":"left","dimension":"overall","tags":["local-edit"]}
```

Allowed preferences are `left`, `right`, `tie`, and `unsure`. Separate annotations
may retain different judgments under unique pair IDs. Calibration and holdout may
not share prompts, source groups, exact media, or auxiliary assets. This deliberately
conservative asset rule includes masks; split shared assets consistently. Different
edits/crops of one source must carry the same source group: hashes alone cannot
identify near-duplicates.

To collect these annotations, declare pairs in `review-pairs.jsonl` using the same
fields but **omit `preference`**. Choose pairs and source-separated splits before
looking at holdout results. Export a portable browser review:

```bash
python -m reward_lab.scripts.calibrate_scores prepare-review \
  --evaluation outputs/multiaxis-audit --pairs review-pairs.jsonl \
  --seed 42 --output outputs/preference-review
```

Open `outputs/preference-review/index.html`. Original media bytes are copied after
checking their recorded hashes; alpha is preserved and shown over a checkerboard.
Supported browser media are PNG/JPEG/WebP/GIF and MP4/WebM. Each pair shows the
prompt, requested dimension and original reference when available. Candidate order
and A/B assignment are randomized reproducibly. Scores, sample/model names and
split assignment are absent from the page. This is presentation blinding, not
access control: the separate `audit.json` is the content-bound identity map.
Verifier targets and masks are not displayed as candidate evidence.

Choose A, B, tie or cannot judge, then download `review-answers.json`. There are no
default judgments, and unanswered pairs are omitted. Browser storage is a convenience;
download answers to keep them independently of the browser. Convert explicit
answers back to the original sample orientation:

```bash
python -m reward_lab.scripts.calibrate_scores import-review \
  --review outputs/preference-review/audit.json --answers review-answers.json \
  --output preferences.jsonl
```

Import validates the review identity and rejects altered mappings or unknown choices.
It never fills unanswered pairs and refuses to overwrite an existing label file.
The packet is a collection tool, not human evidence until a person actually judges
it. A single-source demonstration cannot support source-separated calibration.
Multiple annotators need distinct pair IDs when their labels are combined.

```bash
python -m reward_lab.scripts.calibrate_scores fit \
  --evaluation outputs/multiaxis-audit --preferences preferences.jsonl \
  --axes alignment quality --l2 0.1 --tie-margin 0.1 \
  --output outputs/reports/frozen-combination.json
python -m reward_lab.scripts.calibrate_scores evaluate \
  --evaluation outputs/multiaxis-audit --preferences preferences.jsonl \
  --combination outputs/reports/frozen-combination.json \
  --output outputs/reports/preference-holdout.json
```

The fit standardizes unique calibration samples and learns an L2-regularized linear
combination with logistic preference loss. Each source group has equal total weight.
Ties target 0.5; unsure annotations are excluded. The saved artifact freezes axes,
scales, coefficients, tie margin, recipe digest, and calibration identities. This combines pointwise axes; it does not
implement a visual pairwise judge or guarantee calibrated probabilities.

Independent scorers can be joined without rerunning models:

```bash
python -m reward_lab.scripts.calibrate_scores fit \
  --component semantic=outputs/editreward-audit \
  --component locality=outputs/masked-edit-audit \
  --preferences preferences.jsonl \
  --axes semantic/editreward locality/locality \
  --output outputs/reports/frozen-combination.json
```

Use the same component aliases with `evaluate` on source-disjoint holdout runs.
Every scorer must cover exactly the same sample grid and input records (including
source/mask hashes); missing/error rows cause rejection, never intersection or zero
imputation. The joined view namespaces axes and retains original model versions and
timings. Its identity binds source run IDs and observed results. The frozen recipe
binds every scorer configuration, while allowing different holdout samples/run IDs.
The join is an in-memory analysis view, not a new model service or a scoring YAML.

Holdout uses exact three-way agreement, excludes unsure annotations, and bootstraps
source-group mean accuracy. Per-tag figures are descriptive annotation averages.
Changing holdout labels cannot alter fitted weights. Choose hyperparameters before
examining holdout; the tool cannot prevent a human from repeatedly tuning on reports.
No human labels are fabricated, and none are bundled with this implementation.
These commands do not change a training reward configuration automatically.

Apply a frozen combination to a new scoring snapshot without labels or model
inference:

```bash
python -m reward_lab.scripts.calibrate_scores apply \
  --evaluation outputs/reward_evaluation/new-candidates \
  --combination outputs/reports/frozen-combination.json \
  --output outputs/reports/candidate-combination-scores.json
```

The same repeated `--component NAME=DIRECTORY` inputs are supported when the fit
used joined scorers. Recipes must match the fit exactly. Coefficients and scales
stay frozen, including negative coefficients for measured costs; no new labels
are inferred. Each successful row reports its combined score, signed standardized
axis contributions, and original scorer evidence. Failed/missing rows stay
unscored; missing required axes, nonfinite arithmetic, and changed recipes fail.
The application digest binds observed results as well as the combination and run
identities. This derived JSON report does not overwrite raw scores, install a
training reward, or measure held-out preference accuracy. Use `evaluate` with
independent annotations for that final claim.

### Qualify a frozen combination for training

`qualify` re-scores the saved images through training's HTTP adapters, compares
each selected raw axis, and records a deployment receipt. The source evaluation
must use the same HTTP endpoint, model name/version and transport configuration.
For joined evaluations, each mapped axis must come from its corresponding service;
renaming a different measurement to a calibration axis is rejected.

Provide a reward-section YAML (without a surrounding `reward:` key) with unit
component weights and the intended `inference`/`kwargs`. Provide a JSON mapping
from frozen axes to runtime raw axes, for example
`{"semantic/overall": "editreward/overall", "sharpness": "image_sharpness/image_sharpness"}`.
These illustrative axes must actually exist in the selected source evaluations.

```bash
python -m reward_lab.scripts.calibrate_scores qualify \
  --component semantic=outputs/reward_evaluation/semantic \
  --component local=outputs/reward_evaluation/local \
  --combination outputs/reports/frozen-combination.json \
  --reward-config deployment/reward.yaml \
  --axis-mapping deployment/axis-mapping.json \
  --atol 0.000001 --rtol 0 \
  --output outputs/reports/reward-deployment.json
```

Choose tolerances deliberately for the actual measurements; the example is not
a universal precision recommendation. All source rows must succeed and all media
and auxiliary hashes must match. A failed comparison emits no successful receipt.
The comparison is raw-axis parity, separate from diffusion log-prob replay parity.

Add the returned `deployment_id` and file path to the same training reward section:

```yaml
reward:
  # Keep components, kwargs and inference identical to qualification.
  calibration:
    deployment_path: outputs/reports/reward-deployment.json
    deployment_id: <the 64-character deployment_id from that file>
```

The factory validates this pin and the resolved reward configuration before
constructing scorer clients. Frozen coefficients replace the usual component
weighted sum. `calibration/contribution/<axis>` and original axes remain available.
Editing the artifact after construction cannot alter the loaded objective.

Initial support is deliberately limited to HTTP services and explicit float32
RGB/RGBA `[C,1,H,W]` tensors in `[0,1]`, matching the chain judge.
Videos, boxed object-store outputs and other tensor contracts are rejected by
this qualified mode. Qualification measures one image per request; it does not
establish arbitrary batch-size invariance or future-input parity. Service version
labels remain the operator's declaration, not cryptographic proof of model bytes.
Passing this gate does not supply human labels, validate preference quality, or
show that optimizing the resulting objective improves visual quality. Ray scorer
binding and broader input contracts need their own qualification path.


## Training observations

Online training retains configured top-level rewards in the stable CSV views and
writes all observed axes to `reward_components.jsonl` beside them. Each record names
its epoch; nested axes use `component/axis`. Missing axes are absent, not replaced
with zero. Resume aligns this sidecar with the checkpoint, including removal of an
incomplete final append. These are per-update mean observations, not per-sample
labels or proof of reward validity; preserve reward archives for later rescoring.

## Structured verifier evidence

`RewardInferenceResult.diagnostics` carries finite JSON objects alongside numeric
scores. It is copied at the result boundary and persists in offline sample records,
per-reward debug JSONL, and HTTP responses. It never becomes a numeric reward axis
or a loss term. Models may implement `score_results(artifacts)` or return structured
results from their existing scoring hook. Runtime-owned identity, deployed version
and elapsed inference timing are still checked; a model cannot substitute another
artifact's result or an incompatible revision.

GenEval now exports its `why` and input specification from the same detector pass.
Its existing numeric `score_batch` API remains available, and the bundled reward
preset enables a debug sidecar. Health reports count evidence-bearing rows and
plain `why` strings; joined evaluations retain diagnostics under scorer names.
A detector's explanation is evidence about its decision, not proof that it saw the
image correctly or a causal diagnosis of the generator's failure.

The HTTP wire version is **7**. Version 6 introduced structured diagnostics;
version 7 adds a service-instance identity to the handshake. Upgrade/restart
clients and services together; older peers fail explicitly at the version
handshake. Existing persisted evaluations without diagnostics still read with
an empty object.

Each service instance generates an ID returned by `GET /info`. The client binds
scoring, wake, park and cancellation requests to that validated ID using
`X-VRL-Service-Instance`. Missing or stale IDs fail with HTTP 409 and the
non-retryable `service_identity_changed` code before artifact decoding or model
work. This also applies when a replacement uses the same model name and version:
its request ownership and accelerator isolation have not been validated by the
old client. Create and preflight a new client before resuming scheduling; do not
silently refresh an active client's cached memory-placement assumptions. An
ambiguous request on the old instance remains unresolved if cancellation reaches
a replacement, so shared artifacts must remain retained. The instance ID binds
request ownership; it is not authentication or a durable cross-restart lease.

## Reproducible perturbation stress audits

Build diagnostic variants of existing image outputs, then use the same standalone
scorers and transports as ordinary evaluation:

```bash
python -m reward_lab.scripts.stress_media \
  --manifest outputs/candidates/media.jsonl \
  --output-dir outputs/reward_stress/candidates --seed 42
python -m vrl.scripts.rewards.rescore_media \
  --manifest outputs/reward_stress/candidates/media.jsonl \
  --config path/to/scorer.yaml --output-dir outputs/reward_stress/scores
python -m reward_lab.scripts.analyze_scores stress \
  outputs/reward_stress/scores --output outputs/reward_stress/report.json
```

The recipe creates a baseline, RGB blur, seeded additive RGB noise, checkerboard
RGB, black RGB, fully opaque alpha and empty alpha. All files are normalized to
RGBA PNG on the original canvas. RGB perturbations preserve alpha. RGB-only
scorers may ignore alpha entirely; that is an observable limitation, not a reason
to fabricate a penalty. Animated and unnormalized EXIF-oriented inputs are rejected.
Prompts, auxiliary references/masks/targets and metadata remain attached to each
candidate. Source/content identities, seeds and perturbation parameters are saved.
Output directories must be fresh; the normal scoring stage supports exact resume.

The report pairs each variant only with its own baseline under the same task and
asset identities. It retains failed/missing pairs without assigning zero deltas.
Positive deltas mean numerically larger scores; error axes can have the opposite
quality direction. A score increase under noise is a case to inspect, not automatic
proof of reward hacking. These perturbations are not human annotations and never
enter preference calibration as invented labels.

The small perturbation recipe belongs in the independent audit CLI; analysis lives
in the existing reward diagnostics module. The trainer, reward scalar aggregation,
model adapter shapes and generation family APIs remain unchanged.

## Compare generated outputs after a short RL run

Use `paired` when the **images differ** between a baseline and a trained generator
but the scorer and task inputs stay fixed. This differs from `compare`, which
compares reward rankings on the same images.

```bash
python -m reward_lab.scripts.analyze_scores paired \
  --baseline outputs/reward_evaluation/base-seed1 \
  --candidate outputs/reward_evaluation/trained-seed1 \
  --baseline outputs/reward_evaluation/base-seed2 \
  --candidate outputs/reward_evaluation/trained-seed2 \
  --axis rgba_match --output outputs/reports/paired-generator-scores.json
```

Baseline/candidate arguments pair in their respective input order. Each pair
requires identical sample IDs, prompts, auxiliary asset paths/digests and metadata;
output media paths/digests may differ. Every run uses the same scoring recipe.
Record effective generation seeds and sampling settings in manifest metadata when
you want the comparison to check them; the scorer cannot infer missing generation
provenance from pixels. Independently verify model/checkpoint and sampler contracts.

Repeated observations are averaged within sources before computing summary scores
and a source-level bootstrap. Groups use `metadata.source_group`, otherwise the
reference-image digest, otherwise `prompt_id`. One source cannot be assigned to
multiple groups. Explicitly group crops, related prompts and near-duplicates;
file identity alone does not establish semantic independence. Use `--direction -1`
for a lower-is-better cost: reported raw baseline/candidate values stay unchanged,
while the paired difference is signed toward improvement.

Failures and missing rows retain their statuses and have no invented scores.
The report marks incomplete comparisons and states both expected and measured
coverage; numerical summaries are conditional on successfully paired observations.
The comparison ID binds the observations and source grouping. These are frozen
score changes, not automatic proof of human preference or production readiness.

Add `--stratify-by foreground_alpha` to inspect a categorical metadata field that
exists in every sample. The report retains its overall source-balanced comparison
and adds full per-stratum comparisons, including coverage, raw values, directed
deltas and source-level intervals. For example, use `--axis alpha_l1 --direction -1`
with that option to inspect alpha error separately at each foreground opacity.
This can reveal regressions or partial progress hidden by an overall mean or a
clipped terminal score. No model loading or rescoring is needed.

Stratum values must be strings, integers or booleans; their JSON types remain
distinct. Missing fields fail explicitly instead of silently dropping samples.
Choose meaningful categorical bins before analysis rather than using continuous
measurements as categories. Failed and missing scores remain in each stratum's
coverage, and repeated samples are still averaged within sources. Strata may
share sources, so their intervals are descriptive, correlated and unadjusted for
multiple comparisons; they are not independent confirmation experiments.

## Sequential edit preservation

Export a completed edit-chain run (see `edit_chains.md`) instead of assembling
media paths by hand:

```bash
python -m agentic.scripts.export_chain_media \
  --run outputs/chain-eval/chains/page-12/run/run.json \
  --output-dir outputs/chain-export
python -m vrl.scripts.rewards.rescore_media \
  --manifest outputs/chain-export/media.jsonl --config scorer.yaml \
  --output-dir outputs/chain-rescore
```

The export directory contains a copy of the run record, a content-bound
`media.jsonl` (the source, then every step's output, each naming the source as
its reference and carrying the chain's reward assets), and `provenance.json`
with ordered sample IDs and parent relationships. Media stay at their original
paths. Only a successful run exports; provenance is written last as the
completion marker, so do not consume partial exports.

This validates recorded lineage, not an independent attestation that the generator
internals used the reference. It does not rerun the policy or prove likelihood
parity. Keep those model-specific checks separate.

After independently scoring all states of an editing sequence under one fixed
recipe and task, report when requirements are achieved or broken:

```bash
python -m reward_lab.scripts.analyze_scores sequence outputs/sequence-scores \
  --spec sequence.json --output outputs/sequence-report.json
```

Example `sequence.json`:

```json
{"sequence_id":"dialogue-edit","samples":["initial","added","damaged","repaired"],"requirements":[{"name":"first-bubble","axis":"region/panel-1/exact","direction":1,"threshold":1.0,"active_from":0},{"name":"protected-artwork","axis":"rgb_outside_l1","direction":-1,"threshold":0.02,"active_from":1}]}
```

The region exact criterion has a literal OCR interpretation. The pixel threshold
above is illustrative, not a validated default; choose tolerances using the task's
actual reconstruction/noise behavior before examining results. Use namespaced
axes when auditing a joined scoring view. The current CLI reads a single scoring
directory; the Python `sequence_report` function also accepts an explicitly joined
evaluation from `join_evaluations`.

Every state needs its own sample ID, even if the image bytes repeat. Step zero is
the initial state. `active_from` specifies when a requirement becomes due, allowing
later steps to add requirements. All images are still scored against the same
final task/reference/text specification; changed prompt, metadata or auxiliary
identity is rejected. This prevents replacing an old transcript target from being
misreported as retaining it. It does not yet model legitimate replacement of a
previous requirement; use a separately declared task for that case.

Reports retain each raw value, pass/fail/unknown/not-active state, adjacent measured
regressions and improvements, first satisfaction, final status and missing
coverage. A final repaired page does not erase intermediate regressions. Failed
scoring or missing axes remain unknown, never zero; no transition is inferred
across an unknown interval. Criteria and observations participate in the report
digest. This report neither changes reward weights nor proves that the generator
used one state as the next edit's input: actual run lineage must be
verified separately. It measures declared score criteria, not human preference.

## Repeated scoring variation

Score the same manifest under the same frozen configuration into separate fresh
output directories, then compare those independent executions without inference:

```bash
python -m reward_lab.scripts.analyze_scores repeat \
  outputs/reward-repeat-0 outputs/reward-repeat-1 outputs/reward-repeat-2 \
  --output outputs/reward-repeatability.json
```

The command requires at least two distinct directories, identical scorer configs,
and exact matching sample grids and input identities. Identical run IDs are valid:
these IDs bind recipe and inputs, not a unique execution. A resumed directory
reuses completed scores and is not a new noise measurement. Fresh scoring uses
fresh inference request IDs; copying score files does not establish independent
execution, and the report cannot detect a deliberate copy into another directory.

For each axis/sample the report preserves scores and success/error/missing-axis
states across executions, range, and sample standard deviation when at least two
scores exist. It averages samples within their declared metadata `source_group`
(or `prompt_id`) before averaging source groups. Samples with fewer observations
remain explicit and do not receive an artificial zero range; source groups with
no eligible samples are listed. Status counts also remain available when every
scoring attempt fails and no axis can be summarized.

Observed repeatability does not validate semantic quality or provide a calibrated
uncertainty estimate. Keep preprocessing, rubric and model revision fixed; use
ranking comparison separately to assess whether measured variation changes the
ordering of actual candidates. Retain the number of repetitions and failures when
interpreting small or apparently zero variation.
