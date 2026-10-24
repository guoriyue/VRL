# SPRINT: PT26-06 — Measure checkpoint stalls and stage immutable snapshots

Status: **planned — P1 measurement; asynchronous writing begins only if exposed checkpoint stalls justify it.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

`vrl/trainers/checkpointing.py` owns schema-v2 save/restore and uses synchronous state gathering, CPU snapshots, exports and file publication. DCP state-dict helpers do not mean sharded `dcp.save` or asynchronous checkpointing already exists. The current directory replacement sequence must be inspected for the remove/replace visibility gap before making any atomic-publication claim.

## Deliverables

- [ ] Measure all-rank gather, device-to-host snapshot, raw state, LoRA/EMA export, serialization, fsync and publication/coordination separately; report exposed training stall and CPU/host-memory overhead at the real save cadence.
- [ ] Audit model/optimizer/EMA/scheduler/scaler, step, RNG and data-order state and define a single checkpoint boundary. Confirm whether existing resume already reproduces the next update.
- [ ] If worthwhile, capture an immutable snapshot at that boundary and serialize in a bounded background writer. Specify snapshot ownership, maximum in-flight count, backpressure and shutdown/failure propagation before adding concurrency.
- [ ] Publish only a complete durable generation; preserve the previous readable checkpoint until the new one is valid. Add incomplete-save, writer-failure, concurrent reader and interrupted-publication coverage.
- [ ] Restore and reproduce the next update, not just load tensors. Test that later optimizer steps cannot mutate the staged snapshot.

## Acceptance and resources

CPU lifecycle/failure tests precede a real GPU save/resume trial. Require lower exposed checkpoint stall with acceptable D2H contention and host-memory peaks, complete error propagation, consistent all-rank boundaries and next-update equivalence. Small checkpoints with negligible stall do not justify a background writer.

## Boundaries

Keep checkpoint schema/model identity and current same-topology RNG contracts. Distributed sharded DCP and world-size-changing resume belong to the separate [multi-node plan](SPRINT_multinode_fsdp_hsdp_dcp.md); neither is automatically solved by async staging. Reuse `tests/trainers/test_checkpointing.py`, `test_checkpoint_integrity.py` and `test_checkpoint_schema_restore.py`.

Reference: [PyTorch asynchronous checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html).
