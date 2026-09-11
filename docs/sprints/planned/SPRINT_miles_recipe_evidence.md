# SPRINT：Recipe evidence：把冒烟、完整曲线和确定性回归分开

状态：**implementing；启动身份记录已接入 online，曲线与确定性验收仍待执行。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 问题与现状

论文 §6–8 把证明对象限定为具体 recipe、拓扑和环境。VRL 的
`tests/e2e/test_real_checkpoint_rl.py` 已验证真实 checkpoint 的有限步更新、
loss/grad 有限性和 logprob parity；`tests/architecture/test_real_cover_labels.py`
已有 fake/real 对应关系。它们不能单独证明完整学习曲线或夜间稳定复现。

当前必须读取的生产消费者是 `vrl/scripts/train.py`、
`vrl/trainers/online/trainer.py`、`vrl/trainers/checkpointing.py` 和
`vrl/scripts/eval/image_checkpoint_eval.py`。先确认具体 trainer 路径；
不能把老 sprint 的历史路径作为修改依据。

## 目标与产物

每个被宣传为“验证过”的 recipe 都能追溯到一次具体运行：
recipe/config hash、代码和模型 revision、硬件/驱动/torch/kernel、
种子、拓扑、精度、步数、checkpoint、固定评估协议、原始指标及运行日期。
使用实验输出中的一个 manifest 和人可读索引；不向 family registry 加“verified”布尔值。

证据区分完整训练曲线、确定性全尺寸回归、声明过缩小差异的 proxy、仅 smoke。
硬件没有覆盖就明确留空，不从同名 GPU 架构或相邻模型推断。

## 实施步骤

1. 盘点 e2e CASES、experiment presets、已有 curve artifacts；输出每项证据及缺口，
   不把未运行测试写成通过。沿用 real-cover 机制连接 proxy 与真实实验。
2. 选 SD3.5 最小真实 recipe 做参考，固定数据/reward/model 来源及指标集合。
   在相同环境重复运行至少两次，区分采样随机性、GPU 非确定性、外部 reward 波动。
3. 确定性只对已验证的本地 backend/reward 组合启用；遇到非确定性算子直接报告，
   不靠放宽阈值让“bit-exact”成立。外部 API reward 另做统计比较。
4. 保存不可静默覆盖的参考指标与环境身份。确定性指标逐位比较；
   统计性指标预先确定重复次数、置信区间和最小关心效应。
5. 接入现有 GPU CI lane，CPU CI 只验证 manifest、证据文件和引用完整性。
   更新基线必须展示旧/新结果和原因，不允许失败时自动重录。

## 验收与失败注入

- 删除证据文件、改 config/hash、替换模型 revision、伪造 proxy 为 full，必须失败。
- 相同环境可重放；环境变化标记为不可直接比较，不偷偷使用旧 golden。
- 真实多步曲线与 held-out 指标可访问；确定性未实现时仍诚实标记 smoke/curve-only。
- 性能数字单列 warm-up、采样次数和量测边界。不能用 reward 上涨一个 seed 宣布学习收益。

## 应改／应留／非目标

改变证据记录及检查；保留现有训练入口、e2e helper、真实接口和 real-cover 标签。
CASE 常量是测试 fixture，保留；不创建模型支持名单或验证状态解释器。
不因论文使用 nightly 就虚构可用 GPU runner，不重写 CI 调度平台。


## 2026-09-09：启动证据记录已实施

`vrl/trainers/evidence.py` 与 online runner 已接线：每次启动/resume 原子发布独立
`run_evidence/<launch_id>.json`，记录 resolved config/hash、model identity、configured
manifest/report 内容 hash、Git/dirty-diff、软件版本、GPU/driver 和数值运行开关。
不添加 family verified 名单，不给仅启动过的 recipe 打成功或确定性标签。

配置/证据输出和原 CSV preflight 共用 `run_on_primary_rank`：它是实际跨 rank 的失败传播边界，
避免 primary 写盘失败时其他 rank 继续进入训练 collective。协议名、文件目录与环境 key
是边界常量，保留；没有新增 per-algorithm vocabulary 或第二套 model identity。

验证：64 项 evidence、online lifecycle/metrics 和 architecture 测试通过，其中包含真实
双进程 Gloo 的 primary IO 成功/失败传播、启动证据失败后的清理，以及配置解析/hash、
manifest 内容变化、resume 不覆盖和发布冲突测试。一个警告来自 PyTorch TF32 旧接口提示。

使用及证据范围见 [说明](../../training_examples/RUN_EVIDENCE.md)。
尚待：运行结束后的 metrics/checkpoint/eval 证据封存与校验、历史 recipe 盘点、真实重复短曲线、
确定性/统计性比较和 GPU CI 接线。当前快照不是完整训练复现报告。

## 2026-09-09: Final-loop artifact integrity

The online runner now publishes a separate, non-overwriting artifact receipt
following final checkpoint export. It binds the exact launch record, metrics,
and complete final checkpoint tree. The verifier checks content without loading
checkpoint pickle payloads, and rejects missing roles, redirected references,
changed launch/config/model identity, and checkpoint tree additions/removals.
Primary IO errors use the existing cross-rank propagation boundary and still
enter lifecycle cleanup. No additional contract class or algorithm vocabulary
was introduced. The shared atomic publisher removes duplicated publication logic;
protocol constants and existing lifecycle boundaries remain necessary.

This is an integrity observation before cleanup, not a run verdict or verification
grade. It does not preserve pre-resume metrics/checkpoint bytes: archive the output
directory before resuming. Final verdict and held-out evaluation binding, historical
recipe inventory, actual repeated training curves, deterministic/statistical
comparison, and GPU CI integration remain outstanding. Large checkpoints incur an
additional streaming disk read when the receipt is written.

Validation: 78 evidence, online lifecycle/metrics, and architecture tests passed.
Coverage includes real file/tree drift and archive relocation, no-overwrite
publication, final-checkpoint-before-seal ordering, and cleanup on sealing failure.
One warning is the existing PyTorch TF32 API deprecation. These are CPU/integration
checks, not real GPU learning or reproducibility evidence.

## 2026-09-09: Attempt-scoped completion verification

Supervised child launches now receive a fresh shared attempt ID, inherited by all
torchrun workers. Launch evidence, rank verdicts and the supervisor aggregate carry
this ID. Live collection rejects stale or untagged verdicts; the supervisor also
records the actual exit code after joining the child. `verify_run_completion`
requires intact artifacts, a matching successful attempt, zero observed exit,
and exactly one successful verdict from every distributed rank. A stale success,
missing/duplicate rank, rank from another attempt, or success JSON followed by a
nonzero process exit cannot certify completion.

Existing signal cleanup, restart policy, per-rank verdict file ownership, and
aggregate-after-join behavior stay in place. The new constant names an environment
protocol boundary, not a business vocabulary. No extra lifecycle contract class
or wrapper module was introduced. Custom supervised verdict writers must include
the inherited attempt ID, preferably through the existing `write_run_verdict` API.
Standalone/historical runs without shared identity and observed supervisor exit
remain unproven by this completion verifier; no ID is inferred from old filenames.
Final verdict bytes are still separate from the pre-cleanup artifact receipt.

Validation: 119 evidence, supervisor, training-signal and architecture tests passed.
The supervisor suite runs actual subprocess/process-group and torchrun retry
cases. Added checks cover independent attempt IDs, unchanged parent environment,
stale rank rejection, and post-success nonzero exit. CPU evidence fixtures cover
all-rank outcome association and rejection of legacy unassociated successes;
these tests do not establish GPU learning curves or numerical determinism.

## 2026-09-09: Existing image evaluation association

The native image checkpoint evaluator now exposes a verification-only CLI mode
that checks its independently resolved plan against the saved report and original
PNG grid, then associates it with a completed supervised training attempt by model
identity and actual final checkpoint payload digest. The result records protocol
and full evaluation-tree identities. It never generates or scores in verification
mode. `EvaluationArchive.verify_report` is the shared archive integrity boundary;
`verify_training_evaluation` remains a cross-type guard rather than another class.
Required report filenames are schema boundaries. Existing evaluation schemas,
paired statistics, reward ownership and immutable report publication are retained.

This covers native full-sequence denoise image reports only. It does not claim
held-out independence, extend support to video/token evaluators, or turn fixture
scores into a real training curve. Real repeated runs, recipe inventory and GPU CI
remain outstanding. Current hardware inspection found one RTX 5090 with 19,604 MiB
occupied; no GPU learning/reproducibility result was produced by this change.

Validation: 75 evidence, native image evaluation and architecture tests passed.
Failure injection covers independently valid but mismatched model/checkpoint
artifacts, edited scores/images, changed expected protocol, stale verdicts and
nested completion-marker injection. The verification-only CLI test forbids
regeneration and rescoring. These are CPU fixtures, not real learning evidence.

## 2026-09-09: Repository evidence inventory

The dated [inventory](../../training_examples/RECIPE_EVIDENCE_INVENTORY.md) and JSON
snapshot enumerate all 75 experiment YAMLs, 12 real-checkpoint cases (including
all overrides), and 35 existing real-cover labels. 73 presets compose without
extra inputs; two Anima templates intentionally require composition. None of the
configured output directories inspected contain metrics, launch records, complete
checkpoint metadata, or completed evaluation markers. Five presets reference
missing data exports. This scope excludes arbitrary external historical archives.

Eleven real-checkpoint cases substitute reward; two also substitute replay
rollouts. The SD3.5 qualitative note's old 8-sample recipe differs from today's
16-sample, compile-enabled preset. These are explicit evidence gaps, not inferred
recipe failures or a reason to rewrite historical notes. CASES and real-cover
constants remain valid test fixture/protocol boundaries; no production registry
was added to store this dated inventory.

The SD3.5 model cache and OCR manifests are locally present, but Paddle/PaddleOCR
are absent from the active environment and the single GPU remains shared. The
existing smoke memory guard checks total capacity, not free capacity. No real
training was launched. Validation: all 12 case configurations parsed in the
existing CPU preflight test; 15 unrelated tests were deselected. The JSON was also
checked for unique preset paths and consistency with the observed artifact counts.

## 2026-09-09: Preserve metrics before display rounding

The online writer now emits `metrics.full_precision.csv` beside its display CSV,
using the same field schema and resume alignment. Finite aggregates round-trip
without loss, including signed zero. Artifact receipts bind the additional file
when present; historical receipts are still readable but cannot supply missing
precision. The display output and trainer calculations are unchanged. This closes
a prerequisite for numerical regression: prior six/four-decimal formatting could
hide differences. It does not establish repeated real-recipe determinism.

Validation: 166 metrics, evidence, online lifecycle and precision-bridge tests
passed. Added checks distinguish values collapsed by display rounding, verify
scalar bit round-trips, exercise actual online writes/resume, and reject altered
full-precision artifacts. The remaining comparison protocol and real runs are
still required.

## 2026-09-10: Exact cross-run comparison retired

The strict same-revision scalar comparison function, CLI, and dedicated fixtures
were removed. TrainingRunTrace, full-precision metrics, and artifact integrity
checks remain. Historical progress below describes earlier implementations;
there is no current requirement to match complete run configurations or metric
values exactly across runs.

## 2026-09-09: Seed trainer initialization and expose strict numerical settings

The online entrypoint now applies its existing run-owned seed policy before model
construction, using the same initialization seed on every rank. It resets trainer
RNGs to a rank-derived stream after construction and before checkpoint RNG restore.
The prompt Generator and existing checkpoint RNG format remain unchanged. Fresh
initialization is now governed by `trainer.seed`, rather than an ambient global RNG.

`trainer.deterministic=true` explicitly requests strict Torch deterministic algorithms,
deterministic cuDNN and disabled cuDNN benchmarking. Configuration resolution does
not itself mutate process state. The cuBLAS environment name and accepted values are
backend protocol boundaries, not algorithm vocabulary. Offline DPO's existing field
consumption guard rejects the unsupported setting; no duplicate rule was needed.
Default numerical switches and precision/compile/attention ownership are preserved.

Validation: 357 config, online lifecycle and run-policy tests passed. Coverage
includes reaching actual replay model construction under different ambient RNGs,
repeatable Python/NumPy/Torch initialization, rank stream separation, checkpoint RNG
restoration, late workspace rejection, and exact repeated small CUDA network updates.
The CUDA check is synthetic and in-process; it is not independent supervised runs,
a diffusion curve, or full-recipe determinism. Remote rollout/reward randomness and
actual repeated recipe acceptance remain open.

## 2026-09-09: First real SD3.5/OCR launch and precision-recording fix

Installed the repository-pinned PaddleOCR 3.7.0/PaddlePaddle 3.2.1 extra into the
active environment without replacing existing packages. Real OCR preflight scored
one configured manifest row using cached PP-OCRv4 models and synthetic media.
That validates dependency/data plumbing, not image quality or learning.

A supervised two-epoch attempt retained the preset's 512 resolution, batch geometry,
compile policy and OCR reward, with explicit trainer/sampling seeds of 17 and strict
trainer determinism. The real replay model loaded, but launch evidence failed before
the training loop: reading legacy `allow_tf32` conflicted with the production model's
new `fp32_precision` API. The supervisor observed exit 1 and exhausted its one-attempt
budget; no training curve was produced. Original local output:
`outputs/repro/sd3_5_ocr_strict_a`; log: `/tmp/vrl-sd3-ocr-strict-a.log`.

Evidence recording now reuses `vrl.models.precision.float32_precision_state`, the
existing new/legacy compatibility boundary. It records effective string-valued
matmul/cuDNN modes instead of reading legacy TF32 booleans. No numerical policy or
fallback implementation is changed. An actual fresh-process regression applies the
production precision setup before capturing runtime identity; this reproduces the
failing ordering without contaminating other tests' global backend settings.

## 2026-09-09: Real rollout dependency and parking failures

After the precision-recording fix (`08efb112c`), attempt
`outputs/repro/sd3_5_ocr_strict_b` published launch evidence but failed rollout
startup because vLLM/CuMemAllocator was absent. Installed vLLM 0.21.0 without its
Torch dependency closure and the seven entrypoint dependencies documented in
`pyproject.toml`; existing packages were not replaced. The real CUDA one-shot
CuMem build/sleep/wake test passed, including exact tensor restoration.

Attempt `outputs/repro/sd3_5_ocr_strict_c` used the same two-epoch, seed-17 protocol.
Its launch `1fb3248fa648483da9f9fafd2cfed6b0` records a clean checkout at
`08efb112cb5d4ae67065d78acb0a664018252b52`, strict trainer flags, effective TF32
backend modes, installed package versions and both OCR manifest hashes. The
real rollout pipeline loaded and generated 16-sample batches; the first logged
batch took 5.339 seconds. Real OCR scoring ran with the cached PP-OCRv4 models.

The second physical parking validation then failed. CuMem reported releasing
15.74 GiB each time. The baseline was 1,289,748,480 bytes; first parked usage was
1,505,755,136 bytes (within the 268,435,456-byte allowance), while the second was
1,790,967,808 bytes (501,219,328 bytes above baseline). The supervisor observed
exit 1 and stopped. There are no completed epoch rows or final artifact receipt;
this is not a training curve or a repeatability result. Full local log:
`/tmp/vrl-sd3-ocr-strict-c.log`.

Source inspection found that `vrl/utils/cuda_memory.py::gpu_used_bytes` uses
`torch.cuda.mem_get_info`, which measures the whole device. On this shared card,
that cannot attribute a change to the parking worker alone. The failure therefore
does not yet distinguish retained worker allocations, process-lifetime CUDA state,
or other processes' memory changes. The 256 MiB allowance has not been relaxed.
Next diagnosis must compare per-process physical memory and Torch allocations
against the device-wide observation, then validate the ownership boundary with
concurrent allocations. This is an open investigation, not a proven worker leak.

## 2026-09-09: Correct the memory ownership measurement

The [independent CUDA reproduction](../../research/parking_process_ownership_20260909.md)
confirmed that another process's allocation can make the old device-wide parking
check fail after the owner has fully released its CuMem physical pages. Generation
and local reward parking now query process-attributed NVML memory on the CUDA GPU
UUID. Missing/ambiguous accounting fails closed. The residual allowance and phase
ownership contracts are unchanged. Validation: 131 tests passed, one skipped,
including real CuMem sleep/wake with a concurrent foreign allocation. This fixes
an attribution defect; full SD3.5/OCR acceptance remains pending a retry.

## 2026-09-09: Real run passes parking, reaches numerical parity gate

Retry D at clean `a583d830c` passed repeated process-scoped CuMem parking with
identical 660 MiB residual usage, 162 MiB above its 498 MiB baseline. It then failed
before the first optimizer update because rollout/replay log-probability difference
was `0.02667667716741562`, above the unchanged `0.01` threshold. The debug event
records finite values and zero trainer/global step. This is a numerical issue to
investigate through actual backend precision and batch geometry, not an optimizer
update failure or a completed learning curve. Neither threshold has been relaxed.

## 2026-09-09: Matmul reduction modes in launch evidence

Launch runtime identity now records FP16 and BF16 reduced-precision reduction
switches alongside FP32 math modes. The real SD3.5 batch experiment showed that
the BF16 switch changes numerical results despite identical dtype and TF32
settings. The existing whole-runtime comparison rejects different recorded
switches; capture does not mutate them. This describes the trainer process only,
not unobserved remote worker state. Historical receipts are not backfilled.

The fields live in the existing runtime snapshot; no new helper, configuration
policy, module table or numerical default is introduced. Validation uses a fresh
process after production IEEE setup, toggles both real PyTorch switches, checks
recorded values and verifies that collection leaves backend state unchanged.
A comparison fixture rejects otherwise matching runs with differing BF16 modes.

The launch environment allowlist also records
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS` when explicitly set. The SD3.5 experiments
showed different compiled backward logprob errors with this switch, so omitting
it could make different startup environments look identical. This records the
startup variable, not arbitrary subsequent Python mutations of Inductor config
or the settings inside remote actors. The existing whole-runtime comparison
rejects a recorded environment difference. This is an environment-name boundary,
not an expansion to unrestricted environment capture or a new compiler policy.
