# Sequence-parallel hardware acceptance audit

Status: P6 measurements exist; original numerical acceptance remains open.
The parent hardware goal must not count P6 as completed based on a relaxed
run verdict. This audit used existing artifacts only and launched no GPU job.

The authoritative BF16 image comparison at
`outputs/sp_acceptance/compare_retry.json` fails the original 0.02 tolerance:
max error 0.48235297203063965, mean 0.007060134783387184, mismatch fraction
0.07002290338277817. Single-forward FP32 evidence does not cover the complete
denoise trajectory and decoded output. Do not change the comparator threshold
or claim BF16 equivalence from a different precision/path.

Parsed all five rows in each online arm's `metrics.full_precision.csv`:

| Epoch | 2x1 replay max error | 1x2 replay max error |
| --- | ---: | ---: |
| 0 | 0 | 0.008816692978143692 |
| 1 | 0 | 0.01062200590968132 |
| 2 | 0 | 0.006722986698150635 |
| 3 | 0 | 0.005476169288158417 |
| 4 | 0 | 0.004317425191402435 |

The 1x2 resolved configuration explicitly sets the parity limit to 0.02;
its `run_verdict.json` reports success under that configuration. Epoch 1
exceeds the original 0.01 criterion. The earlier retained
`outputs/sp_online/1x2.attempt1.log` separately records the initial original-gate
failure. These are distinct attempts, not a single 0.0088-0.0111 error range.

2x1 pre-update clipping is zero throughout. The 1x2 range is
0.3923611111111111-0.4461805555555556 before optimizer updates. Similar reward
means on changing samples cannot establish equivalent learning or satisfy an
anomaly-free online comparison. Timing observations remain useful diagnostic
evidence, not a corrected-numerics throughput benchmark.

Updated `docs/sprints/SPRINT_engine_worker_vocabulary.md` to distinguish
implementation completion, executed measurements and failed original gates.
Historical results are preserved. No runtime, tolerance, output artifact or
disabled long queue was changed. GPUs remained unclaimed and empty.
