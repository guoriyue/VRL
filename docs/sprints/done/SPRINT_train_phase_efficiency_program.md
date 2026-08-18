# SPRINT PROGRAM：训练段效率 —— rollout 段之外剩下的那 36%

状态：**done（2026-08-17）**。三项全部收口：**两否决、一落地**。
基线 main @ `abb8e4da`。

| 项 | 结果 | 收益 |
|---|---|---|
| P1 训练步同步审计 | **实测否决** | 0.2%（噪声内） |
| P2 prompt embedding 缓存 | **实测否决** | 0.4%（仓库自己早测过） |
| P3 rollout 侧 merge LoRA | **已落地（opt-in）** | **5–12%**，扛得住 compile |

**这个 program 最大的产出是三次 KILL-RISK 门本身**：两个看起来合理的提案
（数得出 16 次同步、证明得了 encoder 冻结）在实测下都只值零点几个百分点，
而真正有货的那条是仓库自己杠杆表里列了却一直没写的。
**先测再写**这条纪律省下的，是两次对 `trainer.py` 和 rollout 执行路径的高风险重构。

> **§0 那个「训练段 64% busy」已经查清：那个数字是过期的。**
> 它测于 2026-06-11，而直接针对它的 compile（launch 数砍 2.6–2.9×、训练 1.25×）
> 于 **2026-06-15** 落地 —— 四天后。五个候选根因已逐个实测排除，
> 全部记录在 [`info/SPRINT_train_phase_gap_hunt.md`](../info/SPRINT_train_phase_gap_hunt.md)。
> 那份存档还测出一条与杠杆表**方向相反**的事实：全参在**训练段**比 LoRA 慢 26%，
> 所以 P1.5 的收益是 rollout-only 的。

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

## 1. 执行项（全部收口）

按「证据强度 × 风险倒序」排。互相独立，可并行。

### ~~P1 — 训练步同步审计~~ —— **实测否决，不实施**

移至 [`done/SPRINT_train_step_sync_audit.md`](SPRINT_train_step_sync_audit.md)。

计数是对的（每次迭代确实 16 次 D2H，且确实无循环内消费者），**代价不对**：
那 16 次同步合计 75 µs，而所在迭代是 138 ms —— 占 0.05%，端到端交替 A/B
中位数差 0.2%，在噪声以下。同步只在「CPU 与 GPU 时间可比」时才贵，真实
DiT replay 比那个 regime 重两个数量级。

**留下的教训**：不要再从「哪里有 `.item()`」找训练段的 36% 空转；实测证明内层
循环是 GPU-bound 的，空转不在那里。下一步应该先用 nsys 定位区间。

### ~~P2 — prompt embedding 缓存~~ —— **实测否决，不实施**

移至 [`done/SPRINT_prompt_encode_cache.md`](SPRINT_prompt_encode_cache.md)。

前提全对（encoder 确实冻结、确实跨 chunk 重复编码），收益不对：encode 只占
denoise 的 0.63%，而且本仓的 b8/b16 对照**早就直接测过这个干预本身**（chunk 数
减半 = encode 次数减半）→ 0.4%。结构性原因：encode 每 chunk 一次，denoise
每 chunk `steps × CFG` 次。video 家族的比值只会更小。

**重开条件**：极少步数的蒸馏家族（causvid）。见该文 §3。

### P3 — [rollout 侧 merge LoRA → dense](SPRINT_rollout_lora_merge.md) —— **已落地（opt-in）**

唯一过门的一项。实测 eager 14.0% / compiled 11.9%（seq 1024），
走真实 pass 端到端 9.8%（seq 2048）。**收益扛得住 compile** —— inductor 融得掉
scaling，消不掉那两个额外 GEMM。

计划里三处假设被执行中的证据推翻（colocated 不必排除；PEFT merge/unmerge 往返在
bf16 下 1000 轮后毁掉基权重；versioned weight sync 是计划没预见的冲突），
详见该文 §4。

**默认关。** 开启前必须先过真实模型的 logprob parity 红线，见该文 §5。

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
