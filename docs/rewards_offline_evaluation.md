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
