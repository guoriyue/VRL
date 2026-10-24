# SPRINT: PT26-03 — Locate and reduce batch-dependent LoRA drift

Status: **planned — P0 diagnosis; kernel experiment begins only after a repeatable first divergent operator is isolated.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state and dependencies

Use the existing [parity sprint](SPRINT_rollout_replay_parity_root_cause.md) as the owning program and PT26-00 for backward/update acceptance. `full_sequence_probe.py` already reads recipes and different rollout/replay microbatches. Nonzero `--lora-b-std` is needed to expose adapter computation; an all-zero adapter can hide the issue. The old separate divergence probe and batch-invariant CLI switch are absent.

## Deliverables

- [ ] Establish a lag-zero baseline with identical installed weights, prompts, RNG, denoise/precision settings and attention backend. Record per-step logprobs, mismatch statistics and pre-update clipping for equal and unequal microbatches.
- [ ] Use eager module hooks to isolate the first divergent module, then a focused operator reproduction to identify shape/reduction/attention or input-cast causes. Compile runs compare outputs/performance without promising module-hook attribution.
- [ ] If a LoRA GEMM is responsible, test fixed tile/reduction behavior only at that operator. Compare targeted treatment against existing eager/root/regional compile paths and unchanged correction controls.
- [ ] Validate backward, trainable gradients and one optimizer update using PT26-00. Measure both local kernel cost and full collection/update throughput, then held-out quality for an adoption decision.
- [ ] Distinguish numerical drift, incorrect weight installation and policy staleness in reports; correction by TIS/RS is not proof of identical engines.

## Acceptance and stop rule

Reproduce drift with a nonzero adapter and fail a deliberately wrong implementation. Adoption requires passing explicit per-step, gradient/update and quality gates with acceptable end-to-end cost. Stop the kernel experiment if another operator is the first cause or the targeted treatment does not improve the diagnosed error. Equal hashes, finite gradients or a mean near zero are insufficient equivalence proofs.

## Boundaries

Inspect `vrl/algorithms/logprob_mismatch.py`, `vrl/scripts/generation/full_sequence_probe.py` and the identified operator; preserve existing replay gates, TIS/RS, family timestep semantics and adapter state names. Do not globally enable expensive batch-invariant kernels or extrapolate AR-LLM results to diffusion RL.

References: [TorchTitan](https://github.com/pytorch/torchtitan), [vexact](https://github.com/verl-project/vexact).
