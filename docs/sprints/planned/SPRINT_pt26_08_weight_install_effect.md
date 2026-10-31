# SPRINT: PT26-08 — Verify weight installation effects before transport experiments

Status: **planned — P1 correctness qualification; alternative transports remain P2 and measurement-triggered.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

The CPU-snapshot/ObjectRef/policy-version/ACK protocol and actual Ray tiny-model tests already exist. `tests/generation/ray/test_weight_sync.py` covers shared puts and failure/cancellation scheduling. Transport ownership remains with the [weight-sync sprint](SPRINT_weight_sync_transport_seam.md).

Every install is now verified and fails closed. Each rank's ACK (`WeightInstallAck`, `vrl/generation/execution/types.py`) carries per-tensor digests (`vrl.utils.tensors.tensor_digests`, the encoding `trainable_state_digest` also uses) of the live parameters its forward reads, read back by payload name through `RuntimeModel.live_trainable_state`. `RayGenerationSession.update_weights` digests the snapshot it puts and requires every rank's ACK to carry the published version and equal digests; a mismatch terminalizes the runtime before the version is published. Versioned-slot workers activate the new slot at install so the read-back covers it. For a rollout-quantized linear whose master is synced (full fine-tune), the digest proves the master only; the requantized cache is not compared and is reported as `rollout.weight_sync.requantized_cache_unverified` plus a log line. The no-op and wrong-payload acceptance tests and a two-payload fixed-seed output test run on the real Ray tiny-SANA fleet on CPU.

## Deliverables

- [x] Extend existing trainer and real-Ray tests with two deliberately different valid payloads. Verify installed live trainable tensors, exact version association and a fixed-input forward/logprob that changes as expected.
- [ ] Where rollout quantization is enabled, verify the effective requantized weights/cache reflect the newly installed master state (the master is verified; the cache comparison remains, and needs a GPU for the NVFP4 kernels). Exercise stale/delayed ACK, partial failure, cancellation and source-buffer lifetime.
- [ ] Measure CPU snapshot, metadata/object-store transport, install/requantization, pause/drain and total exposed sync stall for LoRA and any relevant full-state workload.
- [ ] Only when a measured bottleneck meets the existing owner's trigger, prototype one alternative transport in isolation and feed its results back to that owner. Check available Ray RDT/NCCL APIs in the actual pinned environment rather than assuming master-doc APIs exist.

## Acceptance

A no-op install and a version-correct/wrong-payload install must both fail tests. Verify read-complete buffer ownership through partial failure and cancellation. A new transport must preserve admission/version semantics, pass the effect test and improve real end-to-end update throughput on the intended topology.

## Boundaries and deferred options

Keep immutable snapshots, monotonic versions, strict ACKs, one-put shared ObjectRefs and the existing transport facade until evidence justifies a change. RDT pre-registration, receive targets, bucketing and NCCL M2N are conditional candidates; measure metadata separately and account for send-buffer peaks. Sparse deltas need a full-parameter cross-cluster workload with measured low changed-byte density; tiny LoRA payloads do not justify a new wire format.

References: [Ray direct transport](https://docs.ray.io/en/master/core/direct-transport/index.html), [NCCL extensions](https://github.com/NVIDIA/nccl-extensions).
