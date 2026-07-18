# SPRINT: Producer-backed inference quality tests

Status: **PLANNED; structural validator landed, evidence producers missing
(2026-07-18)**.

## Outcome

Before an expensive training launch, an opt-in test must prove that the exact
resolved experiment can produce valid outputs through three independent paths:

1. the official native/reference inference path;
2. the production rollout path used by the repository;
3. an independent replay reconstructed from persisted trajectory artifacts.

The proof remains owned by `tests/quality/`. Production code must not import the
test package, pause or resume training around a quality phase, cache a PASS, or
make runtime policy decisions from test state. An operator or launch workflow
runs the test immediately before `vrl-train` and treats a skip as **no proof**,
not as a passing gate.

This sprint proves inference correctness and detects collapse. It does not
promise that training improves quality.

## Current repository truth

Already landed:

- `tests/quality/protocols/families/` covers all canonical entries derived from
  `FAMILY_REGISTRY` without a second hand-maintained family vocabulary.
- Protocols classify the trainable policy through the orthogonal axes in
  `PolicySemantics`: temporal organization, step kind, action distribution, and
  trajectory layout.
- CPU tests validate config/model identity, artifact hashes, media decoding,
  shape, condition sensitivity, replay tolerance, alignment direction, segment
  order, and required corruption names.
- `tests/quality/test_sana_real_inference.py` proves the correct SANA
  FP16/no-autocast production path against an independent denoise loop and
  demonstrates that the rejected BF16 outer-autocast path diverges.
- `tests/quality/test_protocols.py` prevents `vrl/` from importing either
  `tests.quality` or a production `vrl.quality` package.

Still missing:

- No general producer actually runs the native, production, replay, and scorer
  paths for a resolved experiment.
- `tests/quality/evidence.py` currently accepts manifest scalars such as
  `replay_max_abs_error`, matched/shuffled alignment, and corruption scores.
  Threshold validation cannot prove that those numbers came from the declared
  artifacts or scorer.
- A checked-in profile is a test oracle, not evidence that its model has ever
  passed real inference.
- CI is CPU-only; opt-in real-checkpoint tests may skip when their model/GPU is
  unavailable.

## Required architecture

### Test-owned producers

Add producer support under `tests/quality/producers/`. A producer may import
production runtime code and the upstream/native library, but the dependency is
one-way: production code never imports the producer.

Each producer invocation receives one resolved experiment config and writes raw
records plus artifacts into a caller-selected output directory. It must:

1. resolve one immutable model identity and revision;
2. start the official native/reference path in a fresh process;
3. start the exact production rollout path in a separate fresh process;
4. persist the trajectory inputs needed for replay;
5. rebuild replay without reusing rollout memory objects;
6. execute the pinned independent scorer and every registered corruption;
7. compute all reported metrics from those records;
8. write hashes for inputs, outputs, source, dependency lock, resolved config,
   scorer, protocol, optional checkpoint, and environment identity.

The evidence validator must recompute summary values from raw per-sample records.
It must not accept a caller-provided scalar as proof when the referenced
artifacts contain enough information to derive that value.

### Family-specific native boundaries

Native adapters remain family-specific where upstream protocols differ. This is
a justified thin-file boundary: the independent implementation must not call the
production executor it is meant to check. Shared artifact, process, hashing,
replay, and scorer machinery belongs in common producer support.

One adapter may cover multiple registry entries only when model identity,
sampling protocol, conditioning, and trainable policy semantics are genuinely
the same. Aliases do not create new coverage; distinct checkpoint/task variants
do.

### Artifact lifecycle

Producer outputs are one-shot validation artifacts. Store them outside the
import graph, with names such as `*_preflight`, and retain only the report,
contact sheet, and minimal provenance needed to explain the launch decision.
Do not check large generated media or scratch trajectories into the repository.

## Positive and negative proofs

Every real producer test must contain at least one valid case and one deliberate
breakage that proves the assertion can fail.

Common negative cases:

- source/config/model/checkpoint hash mismatch;
- replay tensor or scheduler-step perturbation;
- prompt or reference permutation;
- solid color, patch blocks, noise, blur, saturation, or confetti;
- identical outputs across different seeds.

Joint-denoise policies additionally cover wrong scheduler, timestep mapping,
dtype, autocast, frame count/order, freeze, repeat, flicker, and reverse.

Causal-token policies additionally cover invalid token range, premature EOS,
token repetition, prefix/schedule mutation, wrong decoder path, and segment
reordering. Continuous-token policies must also run their complete flow decoder;
a shortened smoke configuration is not a native-quality proof.

Reference-conditioned image/video/world policies must use real condition assets.
Zero tensors and synthetic placeholders cannot prove conditioning correctness.

## Human review boundary

Pixel statistics alone cannot certify visual quality. The report must enumerate
every native and production artifact with its prompt/condition, path, and hash,
and must produce a contact sheet or video index for human inspection. Human
review supplements machine assertions; it does not replace replay, identity, or
corruption checks.

## Implementation map

Add or extend only test-owned assets:

- `tests/quality/producers/` — subprocess, artifact, replay, scorer, and
  family-native producer support;
- `tests/quality/test_real_inference_preflight.py` — run or validate produced
  evidence for one resolved config;
- `tests/quality/evidence.py` — recompute metrics from raw records;
- `tests/quality/protocols/` — protocol/version fixtures;
- `tests/quality/README.md` — exact pre-launch command and skip semantics;
- focused real-checkpoint tests for family-specific independent paths.

Do not add:

- `vrl/quality/`;
- a quality field copied into `ModelFamilyEntry`;
- `vrl/scripts/train_worker.py` or a training phase orchestrator;
- trainer/Ray pause-resume state for quality testing;
- YAML switches such as `quality.enabled` or `skip_preflight`;
- a parallel `SUPPORTED_FAMILIES` constant.

## Completion gates

CPU-only gates:

1. Registry-derived profile coverage remains complete when an entry is added or
   renamed.
2. Production-to-test imports remain impossible.
3. Evidence summaries are recomputed and tampered scalar/artifact/hash cases
   fail.
4. Every modality corruption has a true and false regression test.
5. Unknown protocol fields and stale model/config revisions fail closed.

Real-checkpoint gates, per canonical entry before claiming it is validated:

1. Native/reference, production, and independent replay all execute.
2. Correct outputs pass; the registered wrong dtype/scheduler/condition/segment
   case fails.
3. The pinned scorer ranks matched/clean artifacts above shuffled/corrupted
   artifacts by the registered margin.
4. Artifacts are opened and reviewed; the report records that review without
   claiming it was automated.
5. The report binds the exact source and experiment identity used by the later
   training launch.

The first vertical slice is SANA because it already has an independently proven
correct and incorrect precision path. Next add one video joint-denoise and one
causal-token image entry, then expand across the registry. An unimplemented
producer means that family remains **unproven**, never implicitly passing.

## Architecture hygiene

Keep `FAMILY_REGISTRY` as the single deliberately isolated taxonomy/config
table. Derive profile coverage from it. Keep schema/version, protocol names,
checkpoint names, and test fixture constants as legitimate ALL_CAPS boundaries;
do not duplicate typed fields into hand-maintained validation sets.

Keep thin native adapters, subprocess entrypoints, and artifact codecs only
where they provide an independent protocol, process-ownership, or versioned wire
boundary. Do not flatten these boundaries for line-count reduction, and do not
create symmetric empty adapters for policy profiles without a real family.

## References

- `tests/quality/README.md`
- `tests/quality/evidence.py`
- `tests/quality/test_protocols.py`
- `tests/quality/test_real_inference_preflight.py`
- `tests/quality/test_sana_real_inference.py`
- `vrl/families/registry.py`
- `vrl/families/semantics.py`
- `docs/MODEL_TAXONOMY.md`
