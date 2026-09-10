# Real full-parameter CPU weight transport acceptance

The v2 acceptance CLI completed the configured 64 MiB bucket transport with two
independent real SD3.5 medium CPU receivers on September 9, 2026 Pacific. This
extends earlier tiny CPU fixtures and the single-GPU LoRA direct-install probe
to the actual full-parameter tensor payload through the production sync owner.
It is not GPU-direct, multi-rank engine, forward-equivalence or training evidence.

Both attempts below ran from clean `07fa167a3`. They hid CUDA from the entire
private process tree, requested the repository's explicit CPU engine fleet and
limited OpenMP/MKL threads to two. No existing GPU workload was stopped or changed.
The selected BF16 model checkpoint is pinned to
`b940f670f0eda2d07fbb75229e779da1ad11eb80`.

## Completed bucket attempt

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m vrl.scripts.perf.weight_delivery_probe \
  --config experiment/sd3_5/online_grpo_ocr --workers 2 \
  --report outputs/repro/weight_delivery_sd3_5_fullparam_cpu_buckets.json \
  model.use_lora=false model.torch_compile.enable=false \
  distributed.resources.visible_devices=[] \
  distributed.resources.rollout.num_gpus=0 \
  distributed.resources.rollout.num_engines=2 \
  distributed.rollout.weight_sync_bucket_bytes=67108864 trainer.seed=17
```

The [raw report](weight_delivery_sd3_5_fullparam_cpu_20260909.json) records two
receivers, each verifying all 908 trainable parameter tensors totaling
4,486,343,040 bytes (about 4.18 GiB). These are actual full-model parameters, not
LoRA adapters or repeated synthetic tensors. Frozen buffers and encoder/VAE
parameters are outside the transformer trainable-state contract. Both receivers
first installed and verified deliberately different parameter values, then
accepted versions 1 and 2 only after exact content verification through the
configured staged transport. The CLI published after successful fleet cleanup
and exited zero.

| Measurement | Seconds |
|---|---:|
| CPU replay source build | 4.2447 |
| Immutable CPU snapshot export | 12.8518 |
| Receiver 0 poison construction/install/verification | 5.0961 |
| Receiver 1 poison construction/install/verification | 5.1048 |
| First full-fleet bucket sync/install/verification | 5.4903 |
| Repeated full-fleet bucket sync/install/verification | 5.2341 |

These are two acceptance observations, not steady-state throughput statistics.
The CPU source export includes copying the entire trainable state. Poison setup
also transmits the complete snapshot outside the bucketed timing intervals, so
this run does not prove an end-to-end memory ceiling. Each actual CPU rollout
pipeline also loads its frozen encoders and VAE even though this probe does not
execute them. Local log: `/tmp/vrl-weight-fullparam-cpu-buckets.log`.

## Default snapshot comparison did not complete

The comparison used the same command, model and two-receiver CPU topology, with
the bucket override omitted and output set to
`outputs/repro/weight_delivery_sd3_5_fullparam_cpu_snapshot.json`. Both actual
pipelines loaded. During production `rollout.weight_sync`, Ray's normal memory
monitor killed one of this probe's receivers at approximately 87.65 / 91.87 GiB
node memory (0.954008 versus its unchanged 0.95 threshold). The two probe receivers
were reported at 24.81 and 24.37 GiB each. Other workloads were also using host
memory. The error reported one in-use object of 4,486,670,214 serialized bytes.

The sync dispatcher raised `RayActorCallError`; the CLI exited 1 and did not
publish the requested report. Its actor processes were cleaned up. Full local
log: `/tmp/vrl-weight-fullparam-cpu-snapshot.log`. The memory monitor was not
relaxed or disabled, and the failed run was not retried with fewer receivers or
a smaller payload.

This is a real capacity failure of this attempt, not a controlled measurement
proving that buckets always use less RAM. Shared-host usage and allocator history
can affect outcomes, and peak receiver/object-store memory was not instrumented
for the successful bucket run. A completed default comparison, repeated warmed
measurements, and GPU/fabric performance still remain open. Neither transport's
production default changed based on these observations.

## Boundaries

The experiment reuses the actual replay exporter, real CPU rollout pipelines,
production engine/dispatcher/sync owner, staged assembly, model loader and exact
receiver verifier. No fake model or actor ACK replaces these paths. CPU engine
configuration is an existing supported resource boundary. No new transport
implementation, algorithm table or wrapper was introduced in this evidence
commit. Two independent single-rank receivers do not constitute a multi-rank
model-parallel engine or a two-GPU validation.
