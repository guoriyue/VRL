# Sprint 2: Test-Owned Few-Shot Rollout Preview

Status: **SANA VERTICAL SLICE COMPLETE**

## Decision

The image-quality gate design is rejected. Generic scorers, pixel statistics,
synthetic corruptions, prompt shuffles, and manifest thresholds cannot prove
that an arbitrary generated image is acceptable. A report that validates its
own scores and hashes is not an independent quality signal.

The long-term asset is a small preview under `tests/quality`: load one real RL
experiment YAML, use its configured dataset prompts and production rollout
executor, and write up to four individual images for a human to inspect. It is
not a `vrl/quality` package, training gate, pause/resume mechanism, supervisor
verdict, or VRL CLI.

## Behavior

The opt-in pytest entrypoint:

```bash
uv run --no-sync pytest tests/quality/test_rollout_preview.py -q \
  --rollout-preview-config experiment/sana/online_grpo_aesthetic \
  --rollout-preview-dir /tmp/sana-rollout-preview
```

does exactly the following:

1. composes and validates the requested experiment YAML;
2. resolves the family through `FAMILY_REGISTRY`;
3. reads up to the first four real training examples from `data.manifest`;
4. resolves the production rollout model and precision;
5. builds requests through `GenerationRequestBuilder` with one sample per
   prompt and deterministic test-fixture seeds;
6. executes the registered production executor's `plan()` and
   `forward_plan()` path;
7. writes one to four individual PNGs and a small `preview.json` that records the
   prompts, seeds, sampling values, checkpoint identity, and precision.

It does not produce a contact sheet or an automatic PASS/FAIL result.

## Family scope

There is no preview producer registry and no per-family adapter. The production
`FAMILY_REGISTRY` is the only support source. Every registered `t2i` family uses
the same config-driven path. Unsupported runtime dependencies or broken family
builders fail normally instead of being represented by a profile YAML.

Video and reference-conditioned previews are a separate future slice because
they require media-specific persistence and reference artifact handling. They
must reuse the same registry/request/executor path when added; they must not
introduce empty adapters.

## What was removed

- caller-provided quality scores and thresholds;
- native-versus-production comparison with mismatched sampling protocols;
- SANA-specific BF16 comparison and fixed prompt table;
- corruption and prompt-shuffle controls;
- independent replay bundled into human image review;
- artifact hashes, schema versions, status states, and report credentials;
- contact-sheet generation;
- producer coverage tables and family profile YAMLs.

SANA's real per-step denoise parity test remains, but it lives with SANA model
tests because it verifies wrapper math rather than visual quality.

## Architecture hygiene

The module-level preview count and base seed stay ALL_CAPS because they are
stable test fixtures. No business vocabulary, family taxonomy, thresholds, or
prompt bank is hardcoded in the workflow.

The small functions that remain have concrete boundaries:

- the pytest function is the opt-in framework adapter;
- `build_preview_request` adapts a typed training example to the production
  request protocol;
- `write_preview_image` enforces the one-output media/identity boundary;
- registry-owned `ModelFamilyEntry.executor_kwargs(root)` projects executor
  config for both the Ray launcher path and preview tests, so neither reimplements it.

Keeping the registry and executor shapes uniform across image families is more
valuable than reducing a few lines. Video support, checkpoint restore, reward
scoring, replay verification, and training orchestration are explicit
non-goals for this slice.

## Verification

```bash
CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest tests/quality tests/utils/test_media.py -q
uv run --no-sync python -m vrl.config.lint
```

The real SANA check must additionally produce one or more openable PNG files
and a `preview.json` whose resolved precision matches the experiment YAML.
Visual acceptance is recorded by the human reviewing those individual images,
not by the test.
