# Executable application tools

`vrl.scripts` contains user- and operator-facing commands that compose the core
`generation`, `rewards`, and `trainers` domains. It is not a second runtime layer.

| Path | Ownership | Lifecycle |
| --- | --- | --- |
| `train.py`, `supervise.py`, `common/` | Training composition and run supervision | Production |
| `families/` | Family-specific commands that cannot be expressed by the generic entrypoint | Production or reusable evaluation |
| `data/` | Dataset import and canonical manifest construction | Long-term asset generation |
| `denoise/` | Offline target-latent preparation | Long-term asset generation |
| `eval/` | Checkpoint evaluation, reward gates, and temporary proof probes | Mixed; see rules below |
| `generation/` | Generation-path correctness probes without Ray or a trainer | Long-term verification |
| `perf/` | Hardware, numerical, and scheduling measurements | Long-term diagnostics or temporary measurements |

Run installed training commands through `vrl-train`. Other commands expose their
CLI with `python -m vrl.scripts.<module> --help`; data preparation is routed through
`python -m vrl.scripts.data.setup --help`.

## Lifecycle rules

- Keep production entrypoints, external-process adapters, public CLI facades, and
  reusable diagnostics whose result can change after a model, dependency, runtime,
  or hardware change.
- A file named `*_probe.py` must state the unresolved gate it owns. Delete it once
  the answer is recorded and no active sprint, production consumer, config, or
  runbook needs a rerun. Tests alone do not extend a probe's lifecycle; delete or
  migrate them with the retired probe.
- Keep fixed prompts, seeds, thresholds, and schema names beside an evaluation only
  when they define its reproducibility contract.
- Shared modules under `common/` or with a leading underscore must have multiple
  callers or provide a deliberate lazy-import, framework, or protocol boundary.
- Generated datasets, reports, traces, and checkpoints belong outside this package;
  commands should document their canonical output or regeneration path.
