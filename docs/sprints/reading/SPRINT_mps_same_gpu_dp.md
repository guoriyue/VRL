# READING: CUDA MPS and same-GPU replicas — superseded

Status: **superseded, 2026-07-12**.

This note originally proposed relaxing VRL's `0/1` rollout GPU guard, launching
multiple rollout workers on one GPU, and using CUDA MPS to fill apparent compute
headroom. Do not implement that proposal.

Subsequent production-shape measurements changed the decision:

- two SD3.5 rollout replicas improved aggregate throughput by only `1.03x` at
  the production chunk size;
- the archived heterogeneous DiT plus VAE effective-work result was also
  arithmetically overstated: the reported rates normalize to `1.03x`, not
  `1.14x`;
- the DiT critical path slowed by approximately `4.18x`; and
- low SM occupancy did not predict recoverable throughput.

MPS client priority is a driver hint, not an enforceable critical-path QoS
contract. A synthetic small-GEMM gain therefore does not justify an MPS product
path or a fractional Ray GPU escape hatch.

The accepted policy is documented in
`docs/sprints/done/SPRINT_gpu_saturation_and_colocation_decision.md`:

- shared trainer and rollout GPU: `strict_on_policy` with a full phase lease;
- disjoint trainer and rollout GPUs: `continuous` may provide real overlap;
- keep the rollout `gpus_per_worker` guard at `0` or `1`; and
- do not manage an MPS daemon from VRL launchers.

The original article attribution was never recovered as independently verifiable
project evidence. Retain this short pointer for provenance, but use the canonical
decision record and repository measurements for engineering decisions.
