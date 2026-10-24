# SPRINT: PT26-05 — Selective lowering with full training acceptance

Status: **planned — P1, conditional: PT26-03 identifies a compile-sensitive region and PT26-00 is usable.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

`vrl/nn/optimization/regional_compile.py` already compiles repeated transformer blocks while preserving state namespaces. Existing compile/FSDP/activation-checkpointing support must be re-read in the [FSDP compile sprint](SPRINT_fsdp_replay_compile_unblock.md); earlier hard-conflict descriptions are historical. Selective ATen lowering is a separate experiment, not a flag already implemented here.

## Deliverables

- [ ] Pin Torch, compiler/backend and device revisions in an isolated experiment; verify the required lowering APIs exist in that exact version.
- [ ] Define the smallest experimentally justified region and explicit eager fallback. Compare eager, current root compile, current block-regional compile and selective lowering with the same recipe/shape/adapter state.
- [ ] Record graph breaks, recompiles, cold compile time, steady forward/backward/update time, peak/reserved memory and per-step numerical differences. Keep eager divergence diagnosis separate from compiled acceptance.
- [ ] Verify original state-dict names, hot weight updates, checkpoint roundtrip, activation-checkpoint recomputation/RNG and representative shape changes. Fail visibly if a requested candidate silently falls back.

## Acceptance and stop rule

Use PT26-00 for full gradient/update and held-out quality checks. Merge only if the region improves an identified numerical/performance problem and full-step gains exceed measurement noise after compilation cost is accounted for. Otherwise retain the existing compile paths and close the experiment with the pending decision documented.

## Boundaries

Preserve `nn.Module.compile` state identity and optimization-pass ownership. No compiler fork, broad operator replacement, unsupported private API in default recipes or wrapper added solely for benchmark convenience.

Reference: [PyTorch compiler documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html).
