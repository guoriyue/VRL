# Scoring existing media independently of training

`vrl.scripts.rewards.rescore_media` scores a JSONL media manifest with the same
scorer transports as training and writes a resumable, content-hashed run -- the
`Evaluation` every other reward command reads. No trainer, generator or Ray.

## Manifest

One row per sample; relative paths resolve beside the manifest:

```json
{"sample_id":"p0012-s0","prompt_id":"p0012","prompt":"remove the bicycle","path":"images/p0012-s0.png",
 "assets":{"reference_image":"sources/p0012.jpg"},"metadata":{"task":"removal","hint":false}}
```

- `prompt_id` groups samples of one prompt (health, spread and paired reports use it).
- `assets` are the scorer's auxiliary files (`reference_image` becomes
  `reference_images`, plus `edit_mask`, `target_image`, ...); they are hashed and
  passed as absolute paths. Do not repeat them in `metadata`.
- `sha256` is optional producer provenance; the run records its own digest.

## Scorer config (standalone YAML, not a training preset)

Local model:
```yaml
name: editreward-local
revision: editreward-code-v1
preprocessing_revision: native-middle-frame-v1
rubric_revision: editreward-bt-v1
worker_config:
  model_factory: vrl.rewards.models.editreward:EditRewardModel
  reward_model_version: 51b92ab5246295637c4ab3bd71e54a26f0a5189d
  device: cuda:0
media_mode: file
batch_size: 4
```
Operator-run HTTP service (the deployment training will use):
```yaml
inference:
  kind: http
  endpoint: http://127.0.0.1:18316
  expected_model: editreward-qwen25-7b
  expected_model_version: 51b92ab5246295637c4ab3bd71e54a26f0a5189d
  timeout_s: 1800
media_mode: tensor
```
`tensor` uploads decoded RGB frames (alpha dropped); `file` needs paths inside the
service's `artifact_roots`. Auxiliary assets are never uploaded: for HTTP they must
be shared paths under `artifact_roots`.

## Run

```bash
python -m vrl.scripts.rewards.rescore_media --manifest media.jsonl --config scorer.yaml --output-dir outputs/reward_evaluation/<reward>/<set>
```
Output: `provenance.json` (recipe, inputs, hashes), `samples/<hash>.json` (one per
sample, errors kept as errors, never zero scores), `summary.json` (only when all
succeeded). `--resume` retries failures; changed media, references, metadata or
recipe refuse to resume -- use a new directory. One scorer per directory; a
repeat run for gate 1 is a second directory with the same config.

## Reading a run

`python -m reward health --evaluation DIR` -- status counts, per-axis distributions,
prompt-balanced means with bootstrap CI, zero-spread prompts, timings.
`python -m reward compare` / `paired` -- ranking agreement between two scorers, or
a candidate vs baseline per prompt with `--stratify-by <metadata field>`.
`python -m reward repeat DIR1 DIR2` -- repeated-scoring variation (gate 1).
Full reference: `docs/rewards_offline_evaluation.md`.
