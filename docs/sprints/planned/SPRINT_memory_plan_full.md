# SPRINT: Whole memory plan (current repo)

状态：plan。本篇是面向当前仓库实际状态的**完整内存计划**——从已落地的机制
统一，到尚未实现的内存系统。每个 Phase 独立成 sprint、单卡可验。

> 本篇取代并吸收旧蓝图 `SPRINT_generation_memory_system.md`（已删除，2026-06-17）。
> 旧蓝图的框架证据已并入下方"框架借鉴"表；其"现状"清单曾把**已撤销**的 trainer 侧
> frozen offload 误列为现有 generation 机制——本篇按 Phase 0 的实际代码状态修正，
> 不再把一个不存在的机制当系统前提。

## 0. 诊断：有机制，无系统；而且机制只统一了一半

### 现有机制清单（各自正确、互不知晓）

```text
静态策略   model.memory.vae_decode policy（tiling/slicing，已统一 5 family，见下）
          trajectory storage policy / uint8 视频 wire / reward artifact release
相位交接   Ray release lifecycle（拓扑派生）、driver model offload、
          empty_cuda_cache 在相位边缘
反应降级   chunk OOM split（裂半重试，vrl/generation/ray/executor.py）——
          目前 dispatch 前唯一的"预防"机制，且是事后反应
观测       per-chunk peak_memory_mb 遥测、host memory snapshot（只记录、从不裁决）
```

（注意：此清单**不含** frozen offload——它是 trainer 侧伪需求，2026-06-13 已撤销，见
Phase 0。旧蓝图把它列为现有 generation 机制是误判。）

### 已统一（generation 侧）

`model.memory.vae_decode` → 模型声明 `generation_memory_targets()` → runtime builder
调一次 `apply_generation_memory_policy`。家族零散布，单一来源 `MODEL_MEMORY_SECTIONS`。

### trainer 侧 —— 不需要 frozen offload（2026-06-13 结论）

最初判断 trainer 侧的 `model.memory.frozen_offload` 是"机制未统一"的不对称，要
像 VAE 那样收进"声明 target → 单一 policy"。后来发现这是伪命题：**trainer 根本
不加载这些模块。** 每个 family 在 trainer 侧用的是 `*ReplayModel`（`build_replay_bundle`
优先于 `build_bundle`），构造时只装 transformer + scheduler，text encoder / VAE /
pipeline 全部声明为 `generation_only_modules`（absent）。

所以"先全加载、再 park 到 CPU"从来没有真实消费者——`apply_trainer_memory_policy`
在所有 family 上都返回空 move 列表（空转）。正确边界是"不加载"（ReplayModel），
不是"加载后下放"。trainer 侧无需任何 `model.memory` section。

### 系统三件套全缺

```text
预算   没有代码声明"这张卡/这个相位可用多少字节"
       （ResolvedDistributedResources 无任何字节字段，只有 rollout_gpu_memory_fraction
        这种 GPU 占比、非字节预算；resolve_distributed_resources docstring 明示
        "intentionally does static ownership checks only; memory pressure is still a
        runtime concern"，vrl/ray/resources.py:166-171）
核算   peak_memory_mb 遥测只记录，从不裁决
准入   batch/chunk 是数量旋钮非字节旋钮——sbs=8 在 512p=23GB、704p=OOM，
       旋钮语义随分辨率漂移；dispatch 前无人回答"这个 chunk 装得下吗"
       （planner._chunk_size 把 sample_batch_size 解析成 max_samples_per_chunk，
        vrl/generation/execution/planner.py:239-252）
```

---

## Phase 0：trainer 侧 frozen offload —— ❌ reverted 2026-06-13（伪需求，已落地）

2026-06-12 曾把 frozen offload 收进与 VAE 对称的"声明 target → 单一 policy"框架
（`trainer_frozen_targets()` + `apply_trainer_memory_policy` + `frozen_module.py`）。
2026-06-13 整套删除：见上节诊断——trainer 用 ReplayModel，这些模块从未加载，policy
全程空转，不是有效省显存方案。

**代码状态已确认完成**（2026-06-17 核对）：

```text
vrl/trainers/frozen_module.py                   已删（文件不存在）
trainer_frozen_targets / apply_trainer_memory_policy  零引用（rg 无命中）
MODEL_MEMORY_SECTIONS                            = ("vae_decode",)（vrl/models/interfaces/runtime.py:24）
configs/.../sd3_5/medium.yaml 的 frozen_offload  已删
```

保留：所有 ReplayModel / `build_replay_bundle` / `minimal_replay_bundle_metadata`
（`generation_only_modules` 是"不加载"的长期证据）、`model.memory.vae_decode`
（generation 侧真实消费者）。测试断言 replay bundle 声明 generation-only 模块
absent（`test_minimal_replay_runtime_wiring.py`）。

教训：判断"机制未统一"前，先确认两侧是否真有同一个被管理对象。generation 侧
VAE 在 trainer 与 rollout 都存在；trainer 侧 text encoder/VAE 只在 rollout 存在，
强行"统一"会把一个根本不存在的对象的管理逻辑也铺过去。

---

## Phase 1（L1）：预算 + 核算 —— ❌ 2026-06-17 实现后整体撤销（不值得）

蓝图：resources 派生 per-role 预算分数（colocated = `gpu_memory_fraction`，独占 = 1.0），
worker 发回 chunk reserved 峰值 + 卡显存，driver executor 把分数 × 卡显存换成 MB 上限逐
chunk 比对、超预算 `logger.warning`，再经 collector → `RolloutStats` → `metrics.csv` 聚合。

**为什么撤销（承重负结果，别重建）**：预算线 = `fraction × 卡显存` = **就是那个 cap**
（budget_fraction 取的就是 `gpu_memory_fraction`），而 cap 是 `set_per_process_memory_fraction`，
它硬卡 **reserved**：reserved 想超过 `fraction × total` 时 allocator **直接 OOM**。所以
`max_memory_reserved ≤ 预算` 恒成立，`over_budget = (peak > budget)` **结构上不可能为真**
——告警无法在 crash 之前触发：chunk 要么在预算下（不响），要么想超就崩了（也看不到响）。
独占场景（fraction=1.0）同理：预算 = 整卡，撞物理上限就 OOM，告警永不响。

即：**warn-only L1 与"直接让它崩"等价**——告警是死的，结构化 `over_budget`/`memory_budget_mb`
也只有它自己的测试读、生产链路（曾接到 `metrics.csv`）也只是把恒 false 的布尔搬运一遍。
0.45 这个 cap 本身还是手写的人肉数（`distributed.resources.rollout.memory_fraction`，配
`gpu_pool=trainer` 时生效），即 L1 要替代的
东西。要让告警有意义必须用 **soft 阈值**（如 `0.85 × cap`，在崩前留提前量），但那又引入一个
拍脑袋系数；判断收益不抵复杂度，整套撤销。

**若将来重做**：(1) 预算线必须 **低于** allocator cap（soft margin），否则告警恒不触发；
(2) 真正有用的是 **peak/cap 比值**（绿色 run 上的余量可见度），不是 `over_budget` 布尔；
(3) 自动收益在 L2（按字节预算切 chunk），而非 L1 的观测。

---

## Phase 2（L2）：字节准入 + 执行器阶梯（核心增量）

目标先写清楚：memory estimation 的最终目的不是"把 chunk 切得越多越碎"，也不是
保守省显存；目标是在当前 topology/lifecycle 给出的可用显存内，**投放尽可能多的有效
rollout 工作量**。落到执行层，就是选择最大安全的 `chunk_samples` / `max_inflight`
/ decode microbatch：能塞 8 就不要保守跑 2，塞不下 8 才自动降到 4/2/1。`sample_batch_size`
因此降级为"用户愿意给的上限"，byte admission 负责把它变成"当前形状装得下的最大值"。

最接近的外部形状是 vLLM 的 profiling-admission，不是纯手算 estimator：启动/初始化时
做一次 dummy/profile run，用实测拆出 weight、peak activation、non-torch、CUDA Graph
等不可给 KV cache/请求池使用的内存，再把剩余预算分给可变工作区。vLLM 文档里的核心形状是：

```python
# Execute a forward pass with dummy inputs to profile the memory usage
# of the model.
with memory_profiling(...) as profile_result:
    self.model_runner.profile_run()
```

我们不照搬 KV cache，但照搬这个原则：**先 profile 出不可动基线，再让 admission 使用剩余
预算装最多的 rollout chunk**。纯公式只做初值；真正上线前必须被本机 smoke/profile 校准。

机制归管理：tiling/slicing/decode 微批今天是人肉 bool（配置输入），目标是决策
输出——延迟↔峰值的交换值不值得做只有持预算的一方知道。按阶梯逐级升级
（vLLM 驱逐→停 admit→抢占 同构）：

```text
预算装得下整图 decode → 不 tiling（不白付延迟）
装不下               → 开 tiling/slicing（最便宜执行器）
仍装不下             → decode 微批、缩 chunk（字节准入）
成本模型估错         → OOM split（安全网，出现率=模型质量指标）
```

实现：

```text
地基   estimate_chunk_cost（为 LPT 建）已在 chunk_placement.py:66-77
       —— standalone 纯函数，今天只当 LPT 调度优先级，从不当准入闸；不要把它改成
       显存语义，新增并列 estimator
测量   worker/runtime 建好后跑代表性 profile chunk，记录：
       model_baseline_bytes、profile_activation_peak_bytes、non_torch_bytes、
       cuda_graph_or_compile_overhead_bytes、free/total/cap bytes
估算   长出 estimate_chunk_bytes(request, chunk, capability, profile)
       —— latent_h×w×frames × 步数缓冲 × CFG 倍率 + trajectory buffers
          + VAE decode 峰值；纯公式只给 profile 外形变化做插值/外推
标定   用 cross-model smoke 的 peak_memory_mb 实测回填系数（现以静态 yaml
       数字死在 6 个配置里——把它从"人肉抄配置"变成"成本模型系数"）
       注意：denoise 激活=持续高原、VAE decode=短时尖峰，可能需两个系数；
       成本模型须读 policy metadata 知道 tiling 是否已应用（闭环已埋）
准入   planner 按剩余预算选最大安全工作量：
       chunk_samples = largest n <= sbs where estimate_chunk_bytes(n) <= budget
       若支持并发 dispatch，同样用 bytes 约束 max_inflight；tiling 配置三态化
       auto|true|false（auto=系统决定）
```

验收 gate：
- shadow mode 先落日志/metrics：`estimated_chunk_bytes`、`actual_peak_memory_mb`、
  `selected_chunk_samples`、`budget_bytes`、`model_baseline_bytes`、`non_torch_bytes`。
- predict2 704p 不手调 sbs 直接跑，planner 自算 chunk 尺寸，零或个位数 OOM split。
- 固定一个 shape 做局部最大性验证：系统选择的 `n` 能跑通，`n+1` 要么被 estimator 判
  超预算，要么在 calibration/probe 中触发 OOM split；不能长期保守停在明显可放大的小 chunk。

风险纪律：
- **标定债是 L2 的头号风险**。低估→仍 OOM→落回保留的 split（没消除问题）；
  高估→chunk 过小→吞吐损失、GPU 欠载，且**无 OOM、无报警、纯静默浪费**——
  这是 L2 新引入的失败模式。系数表会像手维护的 ALL_CAPS 常量一样腐烂，每加一个
  新 family/分辨率就静默误估，直到有人重跑 smoke 回填。
- 先**标定后准入**：上线前让 estimate_chunk_bytes 与现有 OOM-split 并行跑，
  对比"预测字节 vs 实测 peak"，误差可控了再让它当准入闸。
- 成功指标不是"显存越低越好"，而是 split 率低、peak/cap 有合理余量、selected chunk
  接近局部最大安全值；高估导致的 underfill 必须被当成回归。
- "split 出现率=模型质量指标"只有**有人盯这条 telemetry** 才成立，否则标定漂移会把
  "吵闹的人肉旋钮"换成"安静的误估"。
- `peak_memory_mb` 是 per-chunk 执行后采集；拿它设 per-**role** 相位预算会把"chunk 峰值"
  和"角色峰值"混为一谈（多并发 chunk + driver 常驻 + reward 同租）。单卡 colocated
  "相位共享预算"恰恰最难标定，又正是 cosmos 在跑的场景。
- 配置可读性：`tiling: auto` 成默认后，读配置不再能告诉运营 runtime 会做什么。
  auto 决策**必须把"选了什么 tiling/microbatch + 驱动它的字节算式"打日志**，否则
  运营无法复现一次 sizing。

---

## Phase 3（L3）：标签式相位交接（替换 kill/relaunch）

```text
现状   RayGenerationRuntime（with_release_after_collect 租约，vrl/generation/ray/runtime.py）
       整 actor 拆毁重建（release-after-collect：release()→runtime.shutdown()，
       下一轮 _ensure_runtime() 用 RayGenerationLauncher 重建）
       NFT 周期重载耗时【尚未实测】——"~5min" 是文档间互引的估值、非测量值；
       L3 排期前需一个 reload-cost probe 先建立 baseline
目标   slime 形状：权重 tag 常驻、激活/缓冲 tag 释放；reward 相位只让出激活
       显存，回 rollout 相位免重载
依赖   torch_memory_saver 或等价 allocator pause/resume；风险在与 CUDA Graph/
       compile 的交互，需单独 spike
验收   先用 probe 测出 worker 重建在 NFT 单步 wall time 的实际占比（baseline），
       L3 落地后该占比归零
```

---

## 框架借鉴（证据在 docs/sprints/reading/）

| 学谁 | 拿什么 | 证据 | 对应 Phase / 我们的痛点 |
| --- | --- | --- | --- |
| vLLM | profiling-admission：dummy/profile run 拆出 weight、peak activation、non-torch、CUDA Graph 等不可用内存，再把剩余预算给 KV cache/请求池；调度器把"分配失败"当背压（停 admit / 抢占） | `vllm.md:72,183,229`；外部 API 文档 `https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/` | L2；我们也要先 profile 不可动基线，再用剩余预算装最大安全 rollout chunk |
| SGLang | 准入前查 allocator：`available_size() >= num_tokens` 不满足先 evict 再查，仍不满足 retract 请求稍后重跑 | `sglang.md:363-374` | L2；我们 OOM 后才反应（split），无事前准入 |
| SGLang-Omni | 字节计价批量收集：encoder 批按 `request_cost_fn` 字节成本 + `max_batch_cost`（10GiB × activation 倍率）；每 stage 显式显存契约（fraction 总和校验） | `sglang-omni.md:304-316,348` | L2（最对症）；VAE decode 微批 / denoise chunk 都该按字节切 |
| slime | 带标签的暂停/恢复：`torch_memory_saver.pause()/resume()` 按 tag（WEIGHTS vs KV_CACHE）分级释放，权重留显存、激活让位 | `slime.md:75` | L3；我们的 release lifecycle 是 kill/relaunch 整个 actor |
| cosmos-rl | 有界暂存队列 + 事件驱动释放：recv 临时张量入队、超界即 sync+free；buffer 内存被训练进度反向约束（`samples_on_the_fly`） | `cosmos-rl.md:281-283` | L1/L3；continuous queue 的 `max_bytes` **已接上**（schedule.py:161 由 `max_ready_bytes_mb` 注入，queue._enforce_caps 按字节丢弃），但只是**事后丢弃 cap**、非**事前准入**（producer._admit 仍按条数） |

不学：paged KV / radix cache 本体——那是"跨请求共享前缀 + 逐 token 增长"的 LLM
serving 形状；diffusion rollout 的 latents 按 chunk 整存整取、无前缀共享，分页买不到
东西。`NEURAL_ECS_ENGINE_DESIGN.md` 的 PagedLatentPool 等跨请求 continuous batching
真出现时再议（同 physical stage scheduler 的 gate）。

---

## 顺序与边界

```text
L2 profiling shadow → L2 标定 → L2 准入 → （独立）L3 spike
（Phase 0 已撤销并落地，见上——trainer 用 ReplayModel，无需 frozen offload；
 Phase 1 warn-only 已撤销，不再作为前置。）
L2 单卡可验。L3 独立 spike，不与多卡耦合（单卡 NFT 周期就有收益），
  但其优先级在 reload-cost probe 出实测前不要排定。
每个 Phase 一个独立 sprint。
```

## 非目标

```text
不做 PagedLatentPool/radix/KV 分页（无跨请求 continuous batching）
不做 OOM auto-tuner 闭环（L2 成本模型 + split 安全网已覆盖，标定靠 smoke）
不动 model.memory 的 target/policy 契约本身——它是 L0-L3 的地基
不把 batch_size/storage/release flags 搬进 model.memory（各有其位）
不拿 L1/L2 当 continuous queue 字节准入的修复方案——queue 的 `max_bytes` 已接上但只做
  事后丢弃（queue._enforce_caps）；把它升级为事前准入是 continuous scheduler redesign
  的活（producer._admit 仍只按条数准入，
  vrl/rollouts/orchestration/continuous/producer.py:155-161），需单独排期
```
