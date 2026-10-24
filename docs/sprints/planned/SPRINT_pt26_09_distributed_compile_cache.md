# SPRINT: PT26-09 — Publish distributed cold-compile artifacts safely

Status: **planned — P2, conditional: repeated multi-rank cold compile is a measured operational cost.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

`vrl/scripts/train.py` already sets a persistent `TORCHINDUCTOR_CACHE_DIR`. This sprint concerns coordinated artifact construction and reuse across workers, not enabling the existing cache. Reuse PT26-00 to separate cold startup cost from warmed throughput.

## Deliverables

- [ ] Measure cold/warm startup by rank and quantify duplicate compile/autotune work for one actual recipe. Verify public artifact export/import support in the pinned Torch version.
- [ ] Define a manifest signature covering source/graph identity, Torch/compiler/toolkit, GPU architecture, precision/attention configuration, shape/stride and regional/root mode, LoRA topology/rank and required scalar guards. Do not assume shape guards capture every semantic dependency.
- [ ] Prototype single-writer publication to a temporary generation followed by validated manifest publication. Readers wait with a bounded policy or compile locally; they never consume partially written artifacts.
- [ ] Test incompatible signatures, corrupt/missing artifacts, writer death, simultaneous writers, changed shapes/LoRA topology and legitimate recompilation. Keep worker state dictionaries and policy weights out of shared compiler artifacts.

## Acceptance and stop rule

Compare fresh-cache, local-warm and coordinated-cache startup on the real worker topology. Require correct outputs/full update, no stale artifact acceptance or startup deadlock, and lower overall startup wall time at acceptable artifact/storage cost. Do not merge coordination if ordinary persistent local caching already removes the problem.

## Boundaries

Keep Inductor's own validation authoritative and existing persistent-cache defaults unchanged unless evidence requires a change. No network cache service, unconditional cross-device cache reuse or global shared mutable cache directory. Coordinate with the [FSDP compile](SPRINT_fsdp_replay_compile_unblock.md) and [upgrade](SPRINT_torch_vllm_ray_upgrade_readiness.md) owners.
