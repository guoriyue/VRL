# Score existing media independently

`vrl.scripts.rewards.score_manifest` scores a JSONL media manifest through the
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
python -m vrl.scripts.rewards.score_manifest \
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

A writer lock prevents concurrent modification. If a process dies, verify it
and its remote request have stopped before manually removing `.writer.lock`.
This tool never stops another process or silently takes over its output.
HTTP deadlines/cancellation use the existing client; a blocking in-process model
cannot be forcibly interrupted by an asyncio timeout. Use an isolated service
when enforceable process responsiveness is required.

Historical training scores are never edited. Each candidate evaluation gets
its own directory. Versioned raw scores can later support calibrated aggregation
without rerunning compatible model inference.

## Health and candidate ranking comparison

```bash
python -m vrl.scripts.rewards.analyze_scores health outputs/sharpness-audit \
  --output outputs/reports/sharpness-health.json
python -m vrl.scripts.rewards.analyze_scores compare outputs/candidate-a outputs/candidate-b \
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

```bash
python -m vrl.scripts.rewards.calibrate_scores fit \
  --evaluation outputs/multiaxis-audit --preferences preferences.jsonl \
  --axes alignment quality --l2 0.1 --tie-margin 0.1 \
  --output outputs/reports/frozen-combination.json
python -m vrl.scripts.rewards.calibrate_scores evaluate \
  --evaluation outputs/multiaxis-audit --preferences preferences.jsonl \
  --combination outputs/reports/frozen-combination.json \
  --output outputs/reports/preference-holdout.json
```

The fit standardizes unique calibration samples and learns an L2-regularized linear
combination with logistic preference loss. Each source group has equal total weight.
Ties target 0.5; unsure annotations are excluded. The saved artifact freezes axes,
scales, coefficients, tie margin, recipe digest, and calibration identities. Axes must
currently come from one scoring evaluation. This combines pointwise axes; it does not
implement a visual pairwise judge or guarantee calibrated probabilities.

Holdout uses exact three-way agreement, excludes unsure annotations, and bootstraps
source-group mean accuracy. Per-tag figures are descriptive annotation averages.
Changing holdout labels cannot alter fitted weights. Choose hyperparameters before
examining holdout; the tool cannot prevent a human from repeatedly tuning on reports.
No human labels are fabricated, and none are bundled with this implementation.
These commands do not change a training reward configuration automatically.
