# Performance baseline record

`BASELINE.jsonl` is an append-only history of what was measured, on what
hardware, at which commit. It is the shared output of the probe CLIs in
`vrl/scripts/perf/`.

## Why this exists

`vrl/scripts/perf/` holds ~19 probes. Each answered a real question, but each
printed its answer to stdout and forgot it. That failure mode is not
hypothetical here:

> The 2026-06 P2 note concluded "CUDA graph works but is worthless" from a
> single cosmos data point (44.6 vs 43.6 ms, called "no difference").
> — commit `f4edb3da`

Re-measuring across four families two months later showed the effect was
**neither zero nor uniform** (+3.5% on sd3_5, −88% on cosmos-predict2.5). The
old conclusion's direction survived; its reasoning did not. Writing the
correction meant reconstructing "same RTX 5090, same script" from memory,
because the original number carried no provenance.

One unlabelled number cost two months of a wrong belief. That is the cost this
file is designed to remove.

## Why provenance, not just numbers

A measurement is only comparable to another if you know what produced it. Each
record therefore carries:

| Field | Why it decides comparability |
|---|---|
| `gpu_name`, `gpu_count` | An RTX 5090 number says nothing about an H100 |
| `commit`, `dirty` | `dirty: true` means the number is **not** reproducible from the commit alone and must never be silently compared against a clean-tree run |
| `torch_version`, `cuda_version` | Kernel selection and compile behaviour move between releases |
| `timestamp`, `hostname` | Orders history; separates machines |
| `context` | Run shape (family, dtype, batch, layers, …) |
| `metrics` | Flat numbers only, so records stay diffable |

`dirty` is deliberately three-state: `false` clean, `true` dirty, `null`
unknown (git unavailable). An unknown state must never be reported as clean.

## Usage

Opt in per run. Probes wired so far:

```bash
# torch.compile A/B (the probe whose missing record caused the incident above)
python -m vrl.scripts.perf.compile_benchmark \
    --family sd3_5 --device cuda --record-baseline

# quantized rollout drift (fp8 / nvfp4)
python -m vrl.scripts.perf.quantized_rollout_drift_probe \
    --scheme fp8 --record-baseline
```

`--record-baseline` takes an optional path; bare, it appends to
`docs/perf/BASELINE.jsonl`. Without the flag a probe behaves exactly as before.

The drift probe records **before** its nvfp4 gate, so a failing run still leaves
its evidence behind — a failure is the datapoint most worth keeping.

From a new probe:

```python
from vrl.scripts.perf.common.baseline import BaselineRecord, append_baseline

append_baseline(BaselineRecord(
    probe="my_probe",
    metrics={"step_ms": 12.3},          # flat numbers only
    context={"family": "sd3_5"},        # structured run shape
))
```

## Rules

1. **Append, never edit.** A wrong old number is corrected by a *new* record
   plus a note, not by rewriting history — that is what made the 2026-06 case
   hard to audit.
2. **Measure competing arms in the same run.** The `f4edb3da` correction found
   the eager baseline drifts under GPU contention, so cross-run comparison of
   separately-measured arms is not trustworthy. One record holds all arms.
3. **Metrics are flat numbers.** Anything structured goes in `context`.
4. **Never compare across differing `gpu_name` / `dirty` without saying so.**

## What this is not

Not a profiler and not a second timing layer. Stage attribution lives in
`vrl/utils/profiling.py` with the vocabulary in `docs/PROFILE_PHASES.md`; this
file only records results. Recording is opt-in: a probe run without
`--record-baseline` behaves exactly as before.

## References

- `vrl/scripts/perf/common/baseline.py` — record + provenance
- `tests/scripts/perf/test_baseline.py` — append-only / dirty / malformed-line behaviour
- `docs/PROFILE_PHASES.md` — the phase-name contract
- commit `f4edb3da` — the incident that motivated this
