# SPRINT：Weight delivery：版本 ACK 之外验证真实参数内容

状态：**implementing；默认训练路径不增加全量 checksum 开销。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 问题与现状

论文 §4.4 的启发是先污染接收端，再验证首次同步，避免同源初始化掩盖漏写。
`vrl/generation/ray/weight_sync.py` 验证每 rank 的安装版本；
`vrl/generation/execution/worker.py::update_weights` 调用实际模型安装方法。
`vrl/trainers/weight_sync.py` 提供不可变 CPU snapshot。
版本正确是必要条件，但不是每个参数写对的充分条件。

复用 `tests/trainers/test_weight_sync.py`、
`tests/generation/ray/test_weight_sync.py` 及 family 的 trainable-state loader。
先审计现有 name/shape completeness 检查，不能另造平行状态 schema。
`vrl/trainers/diagnostics.py::trainable_state_digest` 已支持本地/DTensor 的
driver 状态摘要；优先复用其格式与可比较范围，不发明第二个哈希表示。
driver 步前/步后摘要不能替代接收端安装结果，两端模型布局不同也不能直接比摘要。

## 实施步骤

1. 建立 opt-in startup/acceptance probe，取得 trainer 真实导出与 worker 真实安装后视图；
   family/precision owner 决定比较逻辑，排除项必须列出原因与精确名字。
2. 在隔离的验收进程中，把即将被同步的接收端 trainable 参数置为确定性不同值，
   同步后逐项验证 name、shape、dtype 和内容。禁止污染 live training fleet。
3. 非量化同构路径逐位比较；merge/requantization 路径把 trainer snapshot
   经同一已验证转换后与接收端比较，同时测前向/logprob；禁止一个万能 atol。
4. 将 probe 接入真实一 worker、两 worker 和 version-slot 测试。
   检查全部 rank，不能只看 primary；缺失 slot、混版本和部分成功仍走现有 quarantine。
5. 输出同步 bytes、snapshot/export/install/verify 时间及内容差异报告。
   默认 runtime 不启用逐 tensor 回读，硬件验收/升级/新 family 才运行 probe。

## 故障注入与 DoD

- 漏掉一个 adapter、交换两个同形 tensor、错 base、只更新一个 rank、
  返回正确版本但不写参数：均不能通过。
- 两次安装同一状态得到同一推理结果；slot 淘汰不影响仍在用的版本。
- LoRA merge/requantization 的等价目标明确；冻结 base 不被误当需同步的参数。
- probe 报错不能允许新版本进入 admission；清理后未修改 trainer 参数。
- 真实 Ray worker 实验通过，不以模拟 ACK 的单测替代。

## 应改／应留／非目标

增加真实 verifier 和验收入口；保留原 snapshot、同步锁、all-rank ACK 与失败边界。
薄 Ray worker 方法是 RPC adapter，应保留；checkpoint/protocol 名称是边界常量。
不在此 sprint 添加 NCCL/RDT/delta transport；不重写 state loader 或默认 LoRA 同步。


## 2026-09-09: Opt-in live receiver readback

`RayGenerationWeightSync(..., verify_content=True)` now propagates an acceptance
request to every engine rank. The real Ray worker adapter forwards it to
`GenerationWorkerCore.update_weights`, which checks actual installed parameters
before committing its version ACK. A no-op install raises through the existing
terminal Ray actor boundary; that worker does not advance its policy version.

Native diffusion and token bases expose readback in their existing module/key
namespace. Multi-root verification uses the family's trainable roots, so selected
Wan experts retain their ownership. The comparison reuses strict name/shape/dtype
validation and compares materialized bytes, preserving bf16/float64, NaN payloads
and signed zero. No second digest format or universal numerical tolerance was
introduced. Frozen parameters are excluded by the existing trainable-state
contract, not by a new family-name table.

This is currently a constructor/API-level acceptance option, not a YAML flag or a
production-wide default. It performs synchronous full tensor readback only when
explicitly requested. Retained version slots, DTensor/sharded values, quantized
representations and meta tensors are rejected rather than credited with a live
parameter comparison. Slot activation, merge/requantization forward checks,
base-checkpoint mismatch acceptance, timing reports and a real-model/GPU CLI
remain required before this sprint is complete. Do not enable live-fleet poison
injection: the tests create isolated CPU parameter modules initialized to -99.

The real-process acceptance test uses the production Ray adapter, worker core and
loader/readback path with a small CPU model. It checks two independent engines and
one two-rank engine, including a deliberately skipped second-rank install. These
are actual Ray processes and tensor copies, not scripted version ACKs, but they
are not a multi-GPU model-parallel numerical validation. Default synchronization,
all-rank failure propagation and RPC adapters remain unchanged; the thin methods
are necessary family/RPC boundaries. Shared helpers remove duplicate namespace
validation without introducing contract dataclasses or algorithm constants.

Validation: 67 model readback, worker-slot, Ray weight-sync and architecture tests
passed (one existing dependency warning). The CPU bitwise tests cover poisoned
parameters, reordered values, bf16/float64, signed zero/NaN, omitted expert roots,
and attempts to include frozen parameters. The Ray tests retain the sender
snapshot and verify failed receivers keep their previous version.

## 2026-09-09: Active version-slot acceptance readback

`GenerationWorkerCore.verify_active_weights(sender_snapshot, policy_version)` and
its Ray RPC adapter now audit already active parameters. The diffusion model owns
`verify_active_trainable_state`: it requires the requested active slot ID and then
compares actual tensors against the independently supplied sender snapshot. It
never activates a slot, reloads weights, or reads its expected values from the
receiver's retained payload. This catches a corrupted slot and an activation that
claims the right version without copying the corresponding parameters.

The separate audit is intentional: non-draining installation only retains a slot;
its ACK cannot claim to have verified live parameters that a future request will
activate. `update_weights(..., verify_content=True)` still rejects that path.
An isolated acceptance runner must execute the desired request and then audit all
ranks before accepting the experiment. The new RPC does not advance the latest
submit version; it can successfully audit an older request after a newer slot was
installed. Default execution does not acquire per-request readback overhead.

Tests use actual model-base slot installation/activation and worker request
execution. The Ray case executes a small CPU executor in two isolated processes:
both serve v1 after retaining v2, then one activates v2 while the other stays on
v1. The second rank's v2 audit fails without repairing the state. Its forward
output is a test fixture; this proves request-version activation/readback wiring,
not production model numerical output or multi-GPU equivalence.

Keep the thin model-owner and RPC methods: they enforce the active-slot and
process boundaries. No new slot payload format, global model taxonomy, automatic
poisoning, or training flag was introduced. A complete acceptance CLI, full model
identity checks, converted/quantized/sharded verification, timing reports and GPU
forward/logprob validation remain outstanding.

Validation: 96 model-base, worker-slot, Ray sync and architecture tests passed.
Five dependency warnings and a SWIG deprecation notice were emitted. Fault tests
verify that wrong-slot claims and corrupt retained payloads do not pass readback,
and that failed audits leave parameter contents unrepaired.

## 2026-09-09: Isolated in-place acceptance CLI

The executable entrypoint is now:

```bash
python -m vrl.scripts.perf.weight_delivery_probe \
  --config experiment/sd3_5/online_grpo_ocr \
  --workers 1 \
  --report outputs/weight_acceptance/sd3_5.json
```

Optional `--checkpoint PATH` restores a training checkpoint through the existing
strict model-restore protocol before exporting the source snapshot. With no
checkpoint, the probe exports the freshly built replay model; it does not claim
to have tested trained weights. Ordered config overrides follow the options.
Use explicit checkpoint/model pins and the intended precision settings for an
actual acceptance run. The source build requests CPU, so this needs enough host
memory for the real replay model and its exported snapshot.

The CLI projects the real replay/rollout builds through `resolve_online_run` and
`ray_launch_inputs`, exports through the existing trainable-state getter and
immutable CPU snapshot, and starts a private local Ray actor group. It refuses
an already initialized process cluster. It neither attaches to existing workers
nor adds a poison method to production worker actors. Each isolated receiver
first installs and verifies an elementwise-different payload, then installs and
verifies the real snapshot twice. All actors must return before a report is
published. Actor cleanup and private-cluster shutdown precede atomic,
non-overwriting publication using the shared evidence publisher.

The report contains source model identity, optional checkpoint path, source/export
cost, and each worker's tensor count, logical payload bytes, and combined
install/readback timings. It does not measure network throughput or separate the
install and verification costs. `--workers N` creates N single-rank replicas;
GPU execution needs N visible GPUs. Source/target dtype or unsupported-loader
mismatches fail instead of receiving an implicit cast or numerical tolerance.
Versioned slots and multi-rank engine configs are explicitly rejected by this
entrypoint; their separate active-state APIs are not advertised as this CLI's
coverage. Forward/logprob equivalence and converted/sharded acceptance remain open.

Validation: 60 CLI, evidence and architecture tests passed, with one dependency
warning. The CLI test uses a clearly labelled tiny CPU source but actual export,
Ray actor-group launch, tensor serialization, poisoned receiver installs and
readback. A second receiver that passes poisoning but skips the target install
prevents report publication, as does cleanup failure. The sender parameters remain
unchanged; an existing report cannot be overwritten. `--help` was also executed.
No full-model GPU acceptance run was performed in this commit.

The CLI file owns a real executable workflow, the subclass isolates destructive
acceptance behavior, and the shared publisher removes duplicate atomic IO. These
are necessary boundaries; no additional model/algorithm registry was introduced.

## 2026-09-09: Real SD3.5 single-worker acceptance

The real checkpoint probe completed on RTX 5090 at clean revision `30b75ae84`.
One private Ray receiver accepted all 486 trainable tensors (95,551,488 bytes)
after a verified poisoned install and two verified source installs.
See the [command, timings, raw report and limitations](../../research/weight_delivery_sd3_5_20260909.md).
This closes the real single-worker in-place byte check only. Multi-GPU/rank,
converted/sharded and numerical forward acceptance remain open; the separate
training parity failure is not resolved by this result.

## 2026-09-09: Honor the configured source initialization seed

The acceptance CLI now invokes the existing `OnlineRunConfig` process RNG
initializer after preflight and before materializing its CPU replay source.
Previously `trainer.seed=17` was accepted but did not seed newly initialized LoRA
weights in this entrypoint. The byte comparison itself was still valid against
the actual exported snapshot, but repeated source construction was not controlled
by the requested seed. Optional checkpoint restore still follows construction.

New reports include `source_initialization.seed` and
`source_initialization.deterministic`. These describe source-process setup, not
remote worker numerical determinism or forward equivalence. Historical reports
are not rewritten or credited with seeded initialization.

The existing RNG owner is reused; source construction, poisoning, all-receiver
readback, cleanup-before-publication and refusal to overwrite reports remain
unchanged. The isolated worker subclass is still a necessary destructive-test
boundary. No new RNG helper, wrapper, registry or contract dataclass was added.

Validation: six CLI tests passed. The successful real-Ray CPU fixture constructs
a fresh actual linear source for each invocation, changes caller RNG state, and
checks equal source parameters for repeated seed 17 and different parameters for
seed 18. Each invocation still poisons and verifies two actual Ray receivers.
Failure-injection coverage retains missed installation and cleanup failure,
source nonmutation and no report overwrite. Ruff passed for the two touched
Python files. This is a tiny CPU source test, not a repeated real SD3 GPU run.

## 2026-09-09: Exercise the configured production transport

The CLI now sends both target installs through `RayGenerationWeightSync`, using
`resolved.generation.worker.weight_sync_bucket_bytes` exactly as the production
launcher does. Previously the actor installed the received snapshot twice inside
one RPC, so selecting bucket transport did not actually exercise it in the CLI.
Receiver poisoning remains an isolated local operation before these two installs.
The sync owner requires content verification and the version ACK from every
receiver before the CLI can publish success.

Reports now use `vrl.weight-delivery-acceptance/v2`. `transport.kind` distinguishes
`snapshot` from `staged_buckets`; `transport.bucket_bytes` records the resolved
ceiling, and `transport.sync_verify_wall_s.first/repeat` include coordinator-side
transport, receiver installation and verification. The old per-receiver target
installation timing fields were removed rather than relabeled as transport timing.
Per-receiver poisoning time, byte/tensor count and accepted version remain.
Historical v1 evidence stays valid within its original direct-install scope.

Example for bucket acceptance:

```bash
python -m vrl.scripts.perf.weight_delivery_probe \
  --config experiment/sd3_5/online_grpo_ocr --workers 1 \
  --report outputs/weight_acceptance/sd3_5_bucket.json \
  distributed.rollout.weight_sync_bucket_bytes=67108864 trainer.seed=17
```

Poison preparation still sends the complete snapshot once, before the measured
target transfers. This CLI therefore does not establish a hard whole-run
object-store/RSS ceiling or bucket transport peak memory. The two timing samples
are acceptance observations, not a warmed throughput or full-parameter benchmark.
The CLI still rejects multi-rank engines and retained-slot configs; real-model
GPU acceptance of this v2 path remains open.

The existing engine, dispatcher and sync owner supply the real transport and
failure boundaries. The local async function only bridges the synchronous CLI
to those async production calls; the worker subclass only isolates poisoning.
Default production transport, model loaders, all-rank verification, source RNG,
checkpoint restoration and cleanup-before-publication remain unchanged. No new
transport flag, registry, contract dataclass or algorithm vocabulary was added.

Validation: ten CLI checks passed with real Ray processes and tiny CPU source
parameters. Both transports cover two receivers, repeatable source initialization,
a skipped target install, cleanup failure and refusal to overwrite reports.
A bucket-only dropped-chunk injection proves that the CLI actually takes the
staged path and rejects incomplete assembly without publishing a report. Ruff
passed for the two touched Python files. No GPU workload was started for these
checks.
