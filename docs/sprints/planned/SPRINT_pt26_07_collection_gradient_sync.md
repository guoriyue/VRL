# SPRINT: PT26-07 — Defer redundant gradient synchronization across collections

Status: **planned — P1, conditional: multiple collections feed one optimizer update and profiling shows exposed repeated communication.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

`vrl/trainers/online/trainer.py` accumulates replay/backward over collections into one update. `vrl/trainers/strategy.py` owns backend-specific backward; the FSDP strategy applies context-parallel loss scaling. A plain backward currently invokes the backend hooks. First establish the actual collective schedule; the presence of accumulation alone does not prove a bottleneck.

## Deliverables

- [ ] Fix effective global batch, group advantage normalization, optimizer-update count and microbatch order. Profile repeated synchronization and its exposed critical-path cost on the intended DDP/FSDP topology.
- [ ] Define synchronization boundaries through the existing strategy interface. Handle DDP no-sync around the required forward/backward region; independently verify the installed FSDP2 synchronization API and its reshard/memory behavior.
- [ ] Preserve CP gradient scaling, final synchronization, empty/dummy-rank collective ordering, uneven collection counts, streaming release and exception/cancellation behavior.
- [ ] Compare full gradients, clipping norm, optimizer state and next parameters with the unchanged baseline. Include an uneven-rank case that would expose a missed collective or deadlock.
- [ ] Measure retained unsharded gradients, peak allocated/reserved VRAM, total communication and complete update throughput; preserve a straightforward fallback.

## Acceptance and stop rule

Backend-specific distributed tests plus real NCCL GPU trials must pass; CPU/gloo tests alone cannot qualify the optimization. One collection must preserve baseline behavior. Require update equivalence at declared tolerance and a material full-step gain without violating memory headroom. Stop if communication is hidden already or retained-gradient memory exceeds the budget.

## Boundaries

Keep strategy classes and their uniform backward seam; they are necessary backend interfaces. Keep trainer normalization and optimizer ownership. Do not implement DDP mechanics directly in the common trainer or treat `no_sync` as an interchangeable FSDP solution. Use PT26-00 for comparisons; this does not alter rollout admission or policy staleness.
