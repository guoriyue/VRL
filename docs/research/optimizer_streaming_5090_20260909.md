# Disk AdamW: measured memory savings and I/O cost

Measured September 9, 2026 Pacific (September 10 UTC), on the shared RTX 5090.
Driver: NVIDIA 580.173.02; PyTorch 2.11.0+cu130.
This is an optimizer-only capacity probe, not a real-model training result.
[Raw report](optimizer_streaming_5090_20260909.json) includes per-step observations,
CUDA UUID, timestamps, runtime versions, source hashes, and the exact command arguments.
The implementation baseline is `8eb404569`; the probe source is identified by its
recorded SHA-256 because it was added in the measurement commit.

## Workload and result

64 FP32 tensors of 1,048,576 elements: 256 MiB of parameters, 256 MiB of constant
preallocated gradients, and about 512 MiB of Adam moments. One first step and three
steady steps per arm, with fresh processes and synchronized CUDA timings. Both arms
use native non-fused AdamW arithmetic. Disk state uses the production optimizer and
32 MiB parameter-aligned buckets on local `/tmp` (`/dev/nvme0n1p2`).

| Measurement | Resident AdamW | Disk AdamW |
|---|---:|---:|
| CUDA allocated before first step | 512 MiB | 512 MiB |
| Peak CUDA allocated | 1036 MiB | 548 MiB |
| CUDA allocated after final step | 1024 MiB | 512 MiB |
| First step | 45.1 ms | 590.1 ms |
| Median of three steady steps | 6.98 ms | 932.42 ms |
| Process peak RSS, including startup | 849.9 MiB | 910.1 MiB |
| Physical writes across four steps | 0 | 2,147,844,096 bytes |

Final parameter, first/second moment, and step-counter bytes matched exactly across
arms (SHA-256 `938bc74cf4d6b292921b8ec3ffce9cb11739696c54bb1071010014a227703056`).
The disk arm saved 488 MiB of peak allocated CUDA memory and was
133.6 times slower per optimizer step in this workload.
This ratio is **not** an end-to-end training slowdown estimate.

Logical reads in the disk window were 5,370,713,371 bytes; physical reads
were only 24,576 bytes. Filesystem cache served almost all reads.
Checksumming and serialization are included in timing; checkpoint export/content
hashing is excluded. Each step writes all moments, including fsync; the byte cap does
not include arithmetic temporaries or CPU copies. CUDA figures are this process's
PyTorch allocations, not total board memory or other users' allocations.

## Decision and remaining evidence

Keep disk state disabled by default. This demonstrates memory relief and a substantial
optimizer-step cost. It does not establish an acceptable budget for any training recipe.
No process eviction, GPU reservation, or global filesystem-cache flush was performed.
Sequential arms on shared hardware and three steady steps do not provide a performance
confidence interval. A prior exploratory run observed the same 488 MiB savings and
approximately 131 times optimizer-step slowdown; it is not pooled into this report.

Next acceptance requires a real full-parameter recipe's measured capacity gap,
forward/backward plus optimizer timing, host/checkpoint peak memory, convergence and
an explicit acceptable slowdown. FSDP and intra-parameter bucket splitting remain
unsupported. FP32 master residency is unchanged and is not included in this FP32-only
workload. This measurement does not close the six-item reproduction task.

## Reproduce

```sh
.venv/bin/python -m vrl.scripts.perf.optimizer_streaming_probe \
  --device cuda:0 --parameters 64 --elements 1048576 --steps 3 \
  --bucket-bytes 33554432 --directory /tmp/vrl-optimizer-probe \
  --report /tmp/optimizer-streaming-new-attempt.json
```

Reports are published without overwriting prior attempts. Each arm has a timeout;
failed children prevent publication, and numerical disagreement produces a negative
report and nonzero exit. The isolated probe is a performance/acceptance entrypoint;
it does not change trainer behavior, optimizer defaults, or parking boundaries.
