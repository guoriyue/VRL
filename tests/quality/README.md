# Inference Quality Preflight Tests

This directory owns inference-quality checks. Nothing under `vrl/` imports this
package, the assets are not included in the wheel, and `vrl-train` does not
pause, resume, or make policy decisions from these tests.

Run the fast structural and synthetic corruption suite on CPU:

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/quality -q
```

Before an expensive training launch, an evidence producer should run the fixed
native/reference path, the exact production rollout path, and replay for one
resolved experiment config. Then run the opt-in artifact test:

```bash
pytest tests/quality/test_real_inference_preflight.py -q \
  --quality-config experiment/sana/online_grpo_aesthetic \
  --quality-evidence /absolute/path/to/evidence.json
```

For checkpoint evidence, also pass the exact `checkpoint.pt`:

```bash
pytest tests/quality/test_real_inference_preflight.py -q \
  --quality-config experiment/sana/online_grpo_aesthetic \
  --quality-evidence /absolute/path/to/evidence.json \
  --quality-checkpoint /absolute/path/to/checkpoint.pt
```

The test checks:

- approved primary model path and immutable revision;
- source/lockfile, installed inference dependencies, resolved config, protocol,
  scorer, and optional checkpoint identity;
- native and production media decoding, shape, and strict/guarded native
  similarity;
- conditioned-generation sensitivity;
- replay maximum error, prompt-alignment direction, AR segment order, and every
  modality-required corruption direction;
- artifact and evidence hashes in the emitted diagnostic report.

Visual quality is deliberately NOT scored by pixel statistics (noise scores
high on range/edge/motion style metrics). Instead the report lists every
native and production image/video with its prompt, path, and sha256 — the
launch workflow must have a human open and review those files.

The checked-in family profiles are deliberate test oracles, not runtime state.
A preset revision change must update the corresponding fixture in the same
review, otherwise the coverage test fails.

## Important limitation

This test evaluates artifacts; it does not itself implement 23 official native
inference stacks. The evidence producer remains responsible for actually
running native, production, replay, and the independent scorer. Hand-written
scores are not a proof of correct inference. A launch workflow should execute
the producer and this pytest command immediately before `vrl-train`; production
training code must not cache or trust an old PASS.
