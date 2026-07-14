# SPRINT: 启动时 chunk-size 探测（vLLM profile-run 形状）—— samples_per_chunk: auto

> **Historical correction (2026-07-13).** The startup real-execution probe remains
> current. Later cleanup removed the standalone `estimate_chunk_cost` helper and
> the test-only `ChunkMemoryReading.to_metrics` bridge. Memory shadow rows remain
> log-only provenance and are no longer attached to `GenerationOutput.extra`.

状态：**DONE（2026-07-07）——真机 gate 已过**。代码全部落地（§7）；CPU 侧 13 个新单测 +
全量 580 过、ruff 干净；真机探测在 RTX 5090 上通过（§8 实测记录）。

## 8. 真机 gate 实测（2026-07-07，RTX 5090 32GB，SD3.5-medium 512² / 10 步 / CFG）

生产同路径驱动（真实 launch contract → `GenerationWorkerCore.load_policy` 真权重 →
`probe_chunk_size`），配方 `sd3_5/online_grpo_ocr`（`n_samples_per_prompt=16` 为上限）：

```text
n= 1 [  warmup] peak=16.40GiB wall=7.2s   （载重后首跑，吃掉 autotune/懒初始化）
n= 1 [ fit-low] peak=16.40GiB wall=0.2s
n= 4 [fit-high] peak=16.43GiB wall=9.8s   （见 caveat②）
n=16 [ confirm] peak=17.24GiB wall=1.7s
VERDICT: samples_per_chunk = 16    探测总耗时 ~19s（< 1 分钟 gate ✓）
```

- **判决合理性**：512² 图像 chunk 显存极便宜（confirm 实测 ~130MB/样本），上限被
  `samples_per_prompt=16` 封顶而非显存封顶；历史上该配方 sbs=8/16 也确实跑得动，
  与"探出的 n 不输人肉档案"gate 一致。峰值曲线两轮重跑逐字节一致（显存记账在 host
  侧发射时发生，天然同步，不受异步计时影响）。
- **caveat① 小 n 拟合欠分辨**：(1,4) 两点斜率 13MB/样本 vs confirm 实测 130MB/样本——
  固定开销远大于每样本增量时两点拟合读不准斜率。**设计如预期兜住**：拟合只产生候选，
  confirm 真跑负责裁决；对显存贵的 video 形状（GB/样本级）斜率分辨率不是问题。
- **caveat② 首跑计时污染**：n=4 的 9.8s 疑似该 batch 形状的一次性 cudnn/cublas 自调优
  （已修一个真 bug：trial wall 原先在 `synchronize()` 之前测，异步 kernel 会记错账）。
  拐点规则的污染方向分析：fit-high 被抬高 → 增长收益虚高 → 继续长 → 仍被 confirm 的
  显存判决兜底；confirm 被抬高 → 保守回落。**最坏情形 = 保守，永不越界**。后续若要
  精确拐点，改为每个计时点跑两遍取第二遍（成本翻倍，图像形状无所谓，video 再权衡）。

## 7. 落地记录（2026-07-07）

- **删除**：`estimate_chunk_bytes` / `admissible_chunk_samples` / `ChunkMemoryProfile`（公式估算
  三件套，2026-07-02 落、2026-07-07 按本 sprint 判决移除）；`build_chunk_memory_shadow` 降级为
  纯读数行（漂移监控），日志行改为 `chunk memory: ...`（无预测列）。
- **新增**：`AffinePeakFit`（chunk_placement.py，两点拟合 + `max_samples_within` 预算除法）；
  `DiffusionDenoiseConfig.execute_steps`（buffer 全量分配、denoise 循环截短）；
  `GenerationWorkerCore.probe_chunk_size`（warmup + 双测点 + 确认跑/二分 + 拐点，OOM 是预期内
  判决；budget = fraction×total 合同值）+ `RayGenerationWorker` 透传；
  `RayGenerationRuntime.generate` 的 "auto" 解析（首次 generate 探测、facade 缓存、请求重写、
  lease 重建不重探）；planner 对漏网 "auto" 显式报错；schema 注释登记三态。

承接 [[SPRINT_memory_plan_full]] Phase 2（字节准入）的方向修正：
**放弃公式估算，改为 vLLM 式"启动时真跑探测"**。上下文：手调 `samples_per_chunk` 实测过 1→4 =
1.4x（[[project_p1_sbs_confirmed]]），但每换模型/分辨率都要人肉重调；公式估算（同请求拟合的
`estimate_chunk_bytes`）被判不成立——同请求内 n 恒同导致 err% 恒 0（自己验证自己）、单点拟合把
n 缩放项错记进截距、allocated 峰值对 reserved/碎片乐观。需求侧确认是仿射的（latents/激活/
trajectory buffer 全部 `a + b·n`），非线性只住在 allocator 层（segment 取整、碎片、per-batch
kernel workspace），只咬预算边界的最后几个百分点——所以：**线性负责把搜索降到 2+1 次试跑，
最后一次真跑负责替我们对 allocator 层签字**。

## 0. 一句话

`sampling.samples_per_chunk` 新增 `auto`：rollout worker 启动时（权重已载、首个真实请求之前）
用截短步数的真实 chunk 做 2 测点仿射拟合 + 1 次边界确认跑 + 拐点计时，得出本次 run 的 n，
写回现有 `samples_per_chunk` 链条——之后整条 planner/executor 路径一个字不改。

## 1. vLLM 对照（已读源码，`~/Desktop/vllm` @ 60e48454d）

vLLM 的完整机制，四步，全自动：

1. **启动快照**：初始 free/total、`gpu_memory_utilization × total` = requested 预算；载权重记
   `weights_memory`（`vllm/utils/mem_utils.py:109-305` `MemorySnapshot`/`memory_profiling`）。
2. **最坏形状真跑一遍**（不外推）：`profile_run()` → `_dummy_run(self.max_num_tokens)`——按调度器
   今后被允许的最大 token 数跑一次 dummy forward，实测激活峰值 + non-torch 增量
   （`vllm/v1/worker/gpu_worker.py:439-449`、`gpu_model_runner.py:6239-6312`）。
3. **一道减法**：`available_kv = requested − (weights + torch_peak_increase + non_torch_increase)`
   （`gpu_worker.py:466-500`）。
4. **一次除法 + 运行时数配额**：`num_blocks = available // page_size // num_layers`
   （`vllm/v1/core/kv_cache_utils.py:987`）；运行时准入只数空 block，字节数学只在启动发生一次。

**接线（回答"how vllm connect this"）**：`EngineCore.__init__`（`vllm/v1/engine/core.py:133`）→
`_initialize_kv_caches` → `determine_available_memory()` → `cache_config.num_gpu_blocks = ...`
（core.py:306）→ `model_executor.initialize_from_config(...)`（core.py:321）。**同一次进程启动内、
serving 第一个请求之前完成，用户零操作**；手动逃生门 = `kv_cache_memory_bytes` 配置直接跳过
profiling（`gpu_worker.py:419-437`）。

**为什么 vLLM 敢除法我们要确认跑**：它的除法对象（KV block）是分页结构、严格线性，且 KV 池
预分配成一整块张量，碎片无处可咬；我们的 workspace 没法预分配成一块 → 边界必须真跑一次。

## 2. 结构映射

| vLLM | 我们 |
|---|---|
| `max_num_tokens` 固定，KV 池弹性 | 弹性变量就是计算 batch（chunk n）本身 |
| `profile_run` dummy forward | truncated-steps 真实 chunk probe（worker 侧） |
| `gpu_memory_utilization` | `colocate_with_trainer.memory_fraction`（合同预算） |
| `num_blocks = available // page_size` | 仿射拟合出候选 n（2 测点） |
| （无需确认，池是一块张量） | **n_max 上确认跑一次**（allocator 层签字） |
| `kv_cache_memory_bytes` 手动跳过 | `samples_per_chunk: <int>` 手动跳过 |
| 运行时数 block | 运行时固定 n；OOM-split 留守当安全网 |

## 3. 设计

### 3.1 配置面

`sampling.samples_per_chunk: auto | <int> | null`（`vrl/config/schema.py:268`）：
- `<int>`：现状，probe 关（= vLLM 手动覆盖）。
- `auto`：启动探测。
- `null`：现状默认（planner 回落 `samples_per_prompt`，`chunk_placement.py:101`）。

### 3.2 probe 机制（worker 侧）

时机 = worker `load_policy()` 完成后、首个真实请求前（对应 vLLM "engine init 内、serving 前"）。

1. **预算 = 合同值，不是瞬时 free**：`budget = memory_fraction × total`（独占 = 1.0 × total −
   非本进程占用的部分按启动快照扣除）。**坑（必须这么做的原因）**：colocated 场景 probe 时
   trainer 还没把权重/优化器搬回 GPU，`mem_get_info` 的瞬时 free 会高估可用量，按合同扣就不会。
2. **truncated-steps 试跑**：真实 `forward_chunk_plan` 路径 + probe 模式——trajectory buffer 按
   配置的真实 `num_steps` 全量预分配（buffer 是 n×steps×latent 的大头，必须按真实大小算），但
   denoise 循环只执行 2 步，然后照常 decode 一次。峰值在首两步 + decode 就全部到位；几秒/试点，
   视频也一样。产物全部丢弃。
3. **2 测点仿射拟合**：n=1、n=min(4, 上限) 两次试跑 → `slope = (peak(4)−peak(1))/3`、
   `intercept = peak(1) − slope`（两点拟合天然修掉"单点把 n 缩放项错记进截距"的 v0 病）→
   候选 `n_fit = (budget × (1−margin) − intercept) // slope`，margin 默认 3-5%。
4. **边界确认跑**：在 `n_cand = min(n_fit, samples_per_prompt, n_knee)` 上真跑一次；过 → 用；
   OOM → 退一档再确认（最多几次，每次几秒）。
5. **拐点 n_knee**：每次试跑计时，`ms/sample` 改善 < 阈值（~5%）就不再往上要——塞到收益平掉
   为止，不为最后 5% 白冒显存风险（实测 batch≥4 后基本平，[[project_rollout_bound_class_probe]]）。
6. probe 期间 OOM 是预期内：捕获（复用 `_is_oom_error` 语义）+ `empty_cache` + 降档重试；
   n=1 都装不下 → fail loud（和现状一样会崩，只是提前到 startup、报错可读）。

### 3.3 n 怎么传进真跑（回答"how is the n passed"）

现有链条**不新建管道**：config `sampling.samples_per_chunk`（schema.py:268）→
`CollectorRequestBuilder._sampling()` 打进每个 `request.sampling`
（`vrl/rollouts/collector/requests.py:63`）→ planner 读它切 chunk
（`vrl/generation/execution/chunk_placement.py:101`、`planner.py _chunk_size`）。

`auto` 模式（**实现时修正**，2026-07-07）：lease 模式的 worker 是首次 `generate()` 才懒加载的
（`runtime.py _ensure_runtime`），"runtime init 时探测"没有时机——所以探测挂在**首次 generate**：
`RayGenerationRuntime.generate` 看到 `sampling["samples_per_chunk"] == "auto"` 就对所有 worker 并发
发 `probe_chunk_size()` RPC、取 min、**缓存在 runtime facade 上**（facade 跨 lease 拆建存活），
然后重写本次及后续请求的 sampling 再下发。不写回 collector（跨层写回被否，collector 持续发
"auto"，facade 负责解析）；planner 对漏网的 "auto" 显式报错（非 Ray runtime 不支持）。
多 worker 取 min（同构卡时相等）。

### 3.4 生命周期

- **每次训练 launch 探一次**（vLLM 每次 engine start 也重探；成本几秒）。
- lease kill→relaunch 周期（colocated 时分复用）**不重探**：同 shape 同预算，n 是 run 级常量。
- weight-sync 不影响（权重字节不变）；换分辨率 = 新 launch 自然重探。
- [[SPRINT_memory_plan_full]] 已落的 shadow 读数留守，角色 = 漂移监控：实测 peak 与 probe 结论
  偏差超阈值打 warning（自动 re-probe 是后续 sprint，不在本期）。

### 3.5 RL 正确性

probe 在任何训练请求之前、`torch.no_grad` 的现有 rollout 路径、假 prompt/随机种子、产物丢弃、
不触碰 policy_version / trainable state / RNG 状态（probe 用独立 Generator）→ 对 old_log_prob
零影响。唯一残留 = allocator 缓存的 segment（反而是暖身，正面）。

## 4. 落地清单

```text
vrl/generation/execution/worker.py      probe_chunk_size()（与 update_weights 同层）
vrl/generation/diffusion/executor.py    truncated-steps probe 模式（denoise 循环提前退出开关，
                                        复用现有 forward_chunk_plan / 两相位测量）
vrl/generation/ray/runtime.py           runtime init 调 probe RPC + 结果写回 sampling 源
vrl/config/schema.py                    samples_per_chunk 接受 "auto"（校验三态）
vrl/generation/execution/chunk_placement.py  删 estimate_chunk_bytes/admissible_chunk_samples/
                                        err% 三件套；ChunkMemoryReading + shadow 原始读数留守
tests/                                  概率拟合纯函数单测（两点拟合、margin、降档）、
                                        schema 三态校验、worker probe RPC 契约（CPU fake）
```

## 5. 验收 gate

- `samples_per_chunk: auto` 的 recipe 在 5090 上不手调直接跑：probe 日志给出
  `n_fit / n_knee / n_final` 与各试点 peak/耗时；正式 run 零或个位数 OOM-split。
- 固定 shape 局部最大性：系统选的 n 跑通，`n+1` 在确认跑中 OOM 或被 margin 拒——不允许长期
  保守停在明显可放大的小 chunk（继承 memory-plan L2 的 gate）。
- 探出的 n 与人肉最优（cosmos sbs=4 档案）一致或更优；probe 总耗时 < 1 分钟。
- 手填 int 的所有现有 recipe 行为逐字节不变（probe 代码零介入）。

## 6. 非目标

- 不做运行中在线准入/动态改 n（OS 式"能塞就再跑一个"已论证不适合 compute-bound GPU：并发
  = 更小 kernel，不是更大 batch；OOM 不是缺页，失败不便宜）。
- 不做跨形状外推表 / per-family 常量表（就是要避免的标定债）。
- 不动 `estimate_chunk_cost`（LPT 调度提示，与显存无关）。
- 不做 probe 结果跨 launch 持久化缓存（几秒的事，缓存引入失效判断复杂度）。
- AR 家族不在本期（chunk 显存形状不同，等 diffusion 路径验证后套用）。
