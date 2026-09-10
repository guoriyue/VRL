# Real SD3.5 receiver weight acceptance

On September 9, 2026 Pacific, the isolated acceptance CLI completed successfully
on the shared RTX 5090 using the real SD3.5 medium checkpoint and the OCR GRPO
recipe. The source tree was clean at `30b75ae84`. The preserved
[raw report](weight_delivery_sd3_5_20260909.json) records the pinned model identity
and receiver results; it is not an environment-bound training completion receipt.

```bash
.venv/bin/python -m vrl.scripts.perf.weight_delivery_probe \
  --config experiment/sd3_5/online_grpo_ocr --workers 1 \
  --report outputs/repro/weight_delivery_sd3_5.json trainer.seed=17
```

The CLI built the actual replay source on CPU and loaded an actual rollout model
in a private Ray actor. No training checkpoint was restored. It exported 486
trainable LoRA tensors (95,551,488 bytes), installed deterministic different values
in the receiver, and verified that poisoned installation. It then installed and
read back the source snapshot twice, accepting versions 1 and 2 only after exact
name, shape, dtype and byte comparisons. Publication followed successful fleet
cleanup; the process exited zero. The local log is
`/tmp/vrl-weight-delivery-sd3.log`.

| Measurement | Seconds |
|---|---:|
| Source build | 3.06534 |
| CPU snapshot export | 0.02456 |
| Poison construction, install and verification | 0.10156 |
| First target install and verification | 0.06641 |
| Repeated target install and verification | 0.06309 |

These are one-run observations, not transport throughput benchmarks; install and
readback timing is combined. The command's seed override does not establish
repeatable source initialization: this CLI does not invoke the training entrypoint's
process RNG initializer. Exact comparison uses the snapshot actually exported.

This supplies real-model, single-worker, single-rank in-place trainable-byte
acceptance. It does not validate every synchronization in earlier training runs,
frozen base bytes, generation/logprob equivalence, retained version activation,
converted or quantized models, sharded engines, or a multi-GPU deployment. The
separate real training attempt still fails its first-update replay parity gate.

No runtime code, recipe, precision setting or acceptance threshold changed in
this evidence commit. Existing RPC adapters, model-owned readback and immutable
snapshot boundaries remain necessary. No new helper, contract or vocabulary table
was added. Multi-worker/rank and forward equivalence acceptance remain open.

A subsequent CLI repair initializes the source RNG through `OnlineRunConfig`
and records its seed/deterministic setting. That does not retroactively seed
this historical run or change its byte-acceptance result; see the
[follow-up validation](../sprints/planned/SPRINT_miles_weight_delivery_verification.md).
