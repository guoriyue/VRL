# SPRINT PROGRAM：训练段效率 —— rollout 段之外剩下的那 36%

状态：**active（2026-08-17）**。基线 main @ `abb8e4da`。
本文是索引 + 排序，不含实施细节；执行项各自独立可做。

> **P1 已实测否决（2026-08-17）**，见
> [`done/SPRINT_train_step_sync_audit.md`](../done/SPRINT_train_step_sync_audit.md)。
> 收益 0.2%（噪声内），不实施。**§0 那个「训练段 64% busy」的问题仍未解释** ——
> P1 只排除了一个嫌疑人，见该文 §4 的剩余嫌疑人清单。

## 0. 这个 program 为什么存在

本仓的性能工作**几乎全部打在 rollout 段**，而且做得很彻底：compile 已落地默认
开（1.25–1.37×）、QKV 融合实测低 ROI 已否决、CUDA graph 2026-08-16 复测四家族
后以「收益 family 相反 + 安全前提未满足」明确关闭、FP8 用户暂缓。

同一份 trace 里还写着另一句，但**没有任何 sprint 接过去**：

> cosmos **训练段** GPU 实际只有 ~64% 在做事（1.02M kernel/step，elementwise 47%）
>
> `docs/sprints/info/SPRINT_cross_model_performance.md` §0

以及那句方法论结论：

> **挤性能的对象是 launch 开销和传输/序列化，不是 kernel 本身。**

这个 program 把这句结论应用到它自己还没被应用的地方。**并且每一项先过
KILL-RISK 门再实施** —— P1 就是被自己的门挡下来的。

## 1. 执行项（P1 已否决，剩 P2 / P3）

按「证据强度 × 风险倒序」排。互相独立，可并行。

### ~~P1 — 训练步同步审计~~ —— **实测否决，不实施**

移至 [`done/SPRINT_train_step_sync_audit.md`](../done/SPRINT_train_step_sync_audit.md)。

计数是对的（每次迭代确实 16 次 D2H，且确实无循环内消费者），**代价不对**：
那 16 次同步合计 75 µs，而所在迭代是 138 ms —— 占 0.05%，端到端交替 A/B
中位数差 0.2%，在噪声以下。同步只在「CPU 与 GPU 时间可比」时才贵，真实
DiT replay 比那个 regime 重两个数量级。

**留下的教训**：不要再从「哪里有 `.item()`」找训练段的 36% 空转；实测证明内层
循环是 GPU-bound 的，空转不在那里。下一步应该先用 nsys 定位区间。

### P2 — [prompt embedding 缓存](SPRINT_prompt_encode_cache.md)

**做什么**：text encoder 证明冻结（`requires_grad_(False)`、不在
`trainable_modules`、不在 `policy_cores`、weight sync 不推它），但
`_forward_chunk` 每个 chunk 重新编码。`sample_batch_size=1` 时同一 prompt 被
编码 G 次。缓存它。

**为什么排第二**：收益机制最干净（常量函数缓存，逐位等价，零 drift），
baseline 用现成 telemetry（`stage_durations["encode"]` 已在生产里）。唯一真实
集成风险是 parking 的 256 MiB 残留预算，已在 sprint 的 P3 点名。

严格说 P2 打的是 rollout 段，但它属于「重复固定成本」这一类，和 P1 是同一个
病因，所以放在同一个 program 里。

### P3 — [rollout 侧 merge LoRA → dense](SPRINT_rollout_lora_merge.md)

**做什么**：`SPRINT_gemm_utilization.md` 杠杆表里唯一列出但没写的一条。
merge 后 rollout 前向拿到全参的 GEMM 形状，训练侧仍只存 LoRA 状态。
范围收在 disaggregated worker，colocated 明确排除。

**为什么排最后**：收益最大（LoRA 家族 ~47% elementwise），但也是三项里唯一
**会引入 replay drift** 的（`(W+BA)x` vs `Wx+B(A x)` 在 bf16 下不逐位相同），
必须过 parity 红线，且有 colocated 的边界要守。

## 2. 共用的测量口径

三项都用同一套，不各自造轮子：

- **kernel / launch 计数**：nsys，`vrl/scripts/perf/nsys_gpu_busy.py`
- **同口径 A/B**：`vrl/scripts/perf/compile_benchmark.py`（CUDA graph 复测用的
  就是它，四家族两条腿的数字可直接对照）
- **训练段 MFU**：`vrl/scripts/perf/backward_mfu_probe.py`
- **阶段耗时**：生产内建的 `stage_durations` + `profile_range`

**比较对象的纪律**（抄 2026-08-16 复测的教训）：eager baseline 在 GPU 竞争下会
漂，要比就比两个**生产实际会跑的臂**。

## 3. 共用的红线

任何一项都不得越过：

- **rollout-vs-replay logprob parity 均差 ≤ 0.01**（`trainer.py`）。过不了就
  停，把数字记进对应 sprint 的执行记录，**不放宽阈值**。
- **默认行为零变化**。三项都默认关或默认等价，开关走配置。

## 4. 明确不在本 program 内

- CUDA graph —— 2026-08-16 已用四家族实测关闭，理由是「收益 family 相反且最大
  受益家族反受其害 + parking 缺失效钩子」。**不要重开，除非 parking 先长出
  graph 失效钩子**。
- FP8 / FP4 —— 用户 2026-06-14 暂缓。
- QKV 融合 —— 已实测低 ROI，profiler 里留度量，runtime 不加。
- 权重传输 NCCL 直传 —— `parked/SPRINT_weight_sync_transport_seam.md`，
  触发条件是「第一个全参大模型多卡负载」。
- chunk 间分相重叠 —— `planned/SPRINT_rollout_finalize_overlap_ga.md`。

## 5. 相关

- 起点：`docs/sprints/info/SPRINT_cross_model_performance.md`
- rollout 段已收口的同类工作：`docs/sprints/done/SPRINT_gemm_utilization.md`
- 挂载 seam：`docs/sprints/SPRINT_plug_and_play_optimization_layer.md`
