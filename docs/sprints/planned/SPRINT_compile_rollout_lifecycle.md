# SPRINT: torch.compile × rollout worker 生命周期（Cosmos Predict2.5 RL）（planned）

状态：proposed / planned（2026-06-17 按当前代码全量核实重写）。本 sprint 是**测量 +
生命周期正确性**,不动调度架构。聚焦 **Cosmos Predict2.5 + Kling RL**(不是旧的 Predict2)。

> 方法:逐跳核实了 `vrl/ray/resources.py`、`vrl/generation/ray/{config,launcher,runtime}.py`、
> `vrl/generation/execution/worker.py`、`vrl/models/diffusion/cosmos/predict2_5/`、以及
> `configs/`。**纠正了上一版文档的多处陈旧假设**(见 §0)。`TORCHINDUCTOR_CACHE_DIR`
> 的收益已用 `vrl/scripts/perf/inductor_cache_recompile_probe.py` 在 RTX 5090 实测(见 §4)。

---

## 0. 纠正上一版文档的陈旧处(已对照当前代码)

| 上一版说法 | 现状(已核实) |
|---|---|
| 开关 = `distributed.rollout.release_after_collect` / `persistent_colocated_workers`(public YAML) | **这两个 public key 已删除,config load 直接 hard-fail**(`resources.py` 的 removed-key 检查)。它们现在只是 `RayGenerationConfig` 的**内部字段**(`generation/ray/config.py:40-42`),由 topology 派生填入(`:115-121`)。 |
| 公有开关在 `config.py:35,37` | 公有面**只剩** `distributed.rollout.colocate_with_trainer: {memory_fraction: <0..1>}`(`resources.py:1101-1138`)。 |
| 基线是 Predict2(`predict2_2b.yaml` enable:true) | 本 sprint 聚焦 **Predict2.5**;`predict2_5_2b.yaml:30` 的 `torch_compile.enable: **false**`(默认不编译)。 |
| 互斥约束在 `config.py:60-66` | 现在在 `generation/ray/config.py:65-72`(`__post_init__`);`allow_driver_gpu_overlap` 也是内部字段,由 `resources.colocated` 填(`launcher.py`)。 |

---

## 1. 核心结论 (TL;DR)

**(A) Predict2.5 rollout compile 当前是关的。** `model.torch_compile.enable=false`
(`predict2_5_2b.yaml:30`),且 `rollout.denoise_compile.enable` 默认也 false
(`configs/base/rollout/diffusion.yaml:19`)。接线本身是通的(§2),只是没开。

**(B) 真正的变量是 worker 生命周期。** 编译产物**常驻**则 warmup 摊销到 ~0;每周期
**重建**则首次前向重新 codegen。三态由 **GPU 拓扑派生**(不再是 public 开关):

| 生命周期 | 当前 public 触发 | compile 行为 | 收益 |
|---|---|---|---|
| **split-GPU / resident** | trainer/rollout 各占卡(`distributed.resources` 分设备),无 `colocate_with_trainer` | 编一次、常驻 | 全额 ✅ |
| **colocated release-after-collect** | 单卡 overlap、**无** `colocate_with_trainer` 块 → `colocated=true & persistent=false` | worker 每 collect 周期重建 → **首次前向重 codegen** | 每周期扣一次 stall ⚠️ |
| **colocated persistent** | `distributed.rollout.colocate_with_trainer: {memory_fraction: 0.N}` | 编一次、常驻 | 全额 ✅ |

派生逻辑(`resources.py:303-312, 324-326`):
```text
rollout_release_before_train      = colocated AND NOT persistent_colocated_workers
rollout_release_before_reward_model = reward_shared_with_rollout
rollout_release_after_collect     = before_train OR before_reward_model
lifecycle.rollout.mode            = "on_demand" if rollout_release_after_collect else "resident"
```
launcher 据 `resources.lifecycle.rollout.mode == "on_demand"` 决定走 `with_release_after_collect`
(`launcher.py:185-201`)。

**(C) 本 sprint 交付:** (1) 实测稳态 compile 加速(§4.3:rollout 1.37× / train 1.26×);
(2) 查清 inductor 缓存——torch **默认**就跨进程持久(§4.2:暖编译 0.63s vs 冷 1.88s),所以
重建 worker 的重编译默认就便宜,**不需要任何 cache 配置/代码**(原计划的 `inductor_cache_dir`
旋钮经核实是过度设计,已撤销);(3) 给 Predict2.5 RL 一个明确的 compile 决策(§5:直接开)。

---

## 2. 接线路径(Predict2.5,已核实)

```text
predict2_5_2b.yaml  model.torch_compile.enable  (当前 false)
  └─ extract_runtime_spec: model_config = plain_mapping(cfg.model)        runtime_config.py:43
  └─ RuntimeBuildSpec.torch_compile：enable 为假返回 None                  interfaces/runtime.py:112-117
  └─ build_cosmos_predict25_runtime_bundle：enable 真才调               cosmos/predict2_5/runtime.py:77-78
  └─ CosmosPredict25Model.torch_compile_transformer(mode)              cosmos/predict2_5/model.py:245-246
  └─ torch.compile(self.pipeline.transformer, mode=mode, fullgraph=False)  （注:Predict2.5 编的是 pipeline.transformer）
  └─ denoise 前向调用这张编译后的 transformer（rollout+train 共用）
```

**两个 compile knob（集成面）——这是测 ON vs OFF 的开关:**
- `model.torch_compile.enable`(全局):rollout + train replay 都吃(rollout 继承整个 model block)。
- `rollout.denoise_compile.enable`(**rollout 单独、单向覆盖**):`_apply_rollout_compile_override`
  (`launcher.py:384-415`)——`enable:true` 时**覆盖**成开(即使 model.torch_compile 是 false);
  **`enable:false` 是 no-op(第 396-397 行提前 return),关不掉继承来的 model.torch_compile**。
  经 `capability.supports_torch_compile` 门控(Predict2.5 = True,`capabilities.py:61`)。

> **测 Predict2.5 Kling RL 的 compile ON vs OFF,正确实验面:**
> - **ON(rollout 隔离)**:在 `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
>   覆盖 `rollout.denoise_compile: {enable: true, mode: default}`(只动 rollout,train 不变)。
> - **ON(全局,rollout+train 都编)**:`model.torch_compile.enable: true`。
> - **OFF(基线)**:两者都 false(= 当前默认)。

---

## 3. 问题:lazy compile 的 stall 落在"首次前向",不在 build

`torch.compile()` 只是包一层——真正的 Dynamo trace + guard + inductor codegen 发生在
**第一次前向**(首个 denoise step / 首个 rollout chunk)。所以:

- **worker build**(load checkpoint)和 **compile stall** 是两段:build 在 `load_policy`
  (`worker.py:51-65`),compile stall 在该 worker 的**第一个 rollout chunk**。
- release-after-collect 模式每周期重建 worker → 每周期的**第一个 chunk** 都吃一次重 codegen。
- 所以**"compile 成本"在生产里 ≈ 首个 chunk 的 `execution_s` − 后续 chunk 的中位 `execution_s`**
  (同一 worker)。这正好能用现有的 per-chunk 计时测(§4 的 hooks)。
- 净收益 = (周期内所有前向 × 稳态加速省下的) − (一次重 codegen stall)。周期越长越摊得动。

> 稳态加速倍数 Predict2.5 **已实测**(§4.3:rollout **1.37×**、train **1.26×**,RTX 5090)。
> 所以 resident 下开 compile 的稳态收益是确定的正;待测的只剩 release-after-collect 的
> "每周期重 codegen stall vs 周期内省下的"净账(用 §4.1 的 first-vs-later hook 现场量)。

---

## 4. 测量钩子 + inductor cache(本 sprint 的可执行部分)

### 4.1 五个指标的钩子点(最小改动;能复用就不新增)

| 指标 | 钩子点 | 现状 | 最小做法 |
|---|---|---|---|
| worker build time | `worker.py:51-65 load_policy`(wrap `_build_executor()` :63) | 只有 `log_host_memory` bookend(内存,非计时) | 在 `_build_executor` 两侧 `time.perf_counter()`,记进 worker metadata |
| compile time | 等价于"首个 chunk − 后续中位"(lazy,见 §3) | — | 由下面 first/later 派生,不单独插桩 |
| first-rollout latency | `actor_pool.py:140-150`(已记 per-chunk `execution_s`) | **已有** `execution_s`/`queue_wait_s` | 给 schedule row 加 `is_first`(per-worker 首个 job)flag |
| later-rollout latency | 同上 | **已有** | 同上,按 `is_first` 分组算中位 |
| epoch wall-clock | `online.py:328-376` epoch 循环 | 有 `phase_times` 累积(`PhaseTimer`,trainer.py:80-112,带 CUDA sync) | 循环外 `epoch_start=perf_counter()`,记 `phase_times["epoch.wall_s"]` |

唯一需要新增的是 `is_first` flag(actor_pool schedule row,1 行)+ epoch wall 记一笔——两者都
是**测量,不动调度**。compile stall、worker build 走现有 `execution_s` / `log_host_memory`。

### 4.2 inductor codegen cache —— **默认就持久,无需任何配置/代码**

`torch.compile` 在首次前向做 inductor codegen(生成 Triton/C++ kernel),并把产物**写到磁盘
缓存**,下个进程读回来跳过 codegen。这个缓存 **torch 默认就开**,默认位置
`/tmp/torchinductor_<user>`——是**稳定路径、跨进程持久**(不是 per-process temp)。

**实测(RTX 5090,`inductor_cache_recompile_probe.py`,不设任何 env = 默认缓存):**

| 场景 | compile wall |
|---|---|
| 默认缓存,首次编译(冷) | 1.881s |
| **默认缓存,另一进程再编(暖)** | **0.625s** |

→ **同一台机器上,重建的 rollout worker 默认就命中暖缓存、省 ~67%**(只剩 trace+guard)。
**不需要 `inductor_cache_dir` 配置、不需要往 worker 注入 env、不需要任何代码**——`compile on`
本身就够,缓存是 torch 自带的副作用。(早先 probe 的"全新 dir vs 复用 dir"= "史上第一次编 vs
之后每个 worker",而"之后每个 worker"正是默认行为;故知:**默认即暖**。)

**唯一需要手设 `TORCHINDUCTOR_CACHE_DIR` 的情形(env var,仍然零代码):** `/tmp` 易失(某些云
实例 stop 即清)、或想放 NVMe / 跨 run 复用。那时启动脚本 `export TORCHINDUCTOR_CACHE_DIR=
/persistent/path` 即可,**不进 config schema、不进 runtime**。

### 4.3 稳态 compile 加速(已实测,RTX 5090,`compile_benchmark.py --family cosmos-predict2.5`)

config-init 合成 DiT(compile 效果是结构性的、与权重值无关,不需 checkpoint),mode=default,
fullgraph=False——与 runtime 的 `torch.compile` 调用一致(`base.py` / `predict2_5/model.py:246`):

| path | compile | 步延迟 | launches/step | peak MB | speedup |
|---|---|---|---|---|---|
| **rollout** | off → on | 60.96ms → **44.45ms** | 2763 → 947 | 3891 → 3925 | **1.37×** |
| **train** | off → on | 226.8ms → **180.2ms** | 10098 → 3838 | 7778 → 7789 | **1.26×** |

→ rollout 前向 **快 27%(1.37×)**、train step **1.26×**;机制是 **kernel launch 锐减**(rollout
2763→947,≈2.9× 更少,inductor 融 elementwise epilogue → 确认 launch-bound),**显存几乎不变**。
与 Predict2 的 1.37×/1.25× 一致(同 DiT 家族)。注:合成网格是"modest video latent",绝对延迟非
512p/93f 真值,但 **speedup 比值是结构性的**(launch 融合),真分辨率下同样成立。

---

## 5. 给 Cosmos Predict2.5 RL 的建议

**结论:Cosmos Predict2.5 RL 直接开 compile(`rollout.denoise_compile.enable: true`)。**
resident 拿全额稳态加速;release-after-collect 的每周期重编译也因 **inductor 默认持久缓存**
而是暖的(~0.6s 量级,§4.2),不再需要任何 cache 配置。

1. **paper-faithful 跑(多卡 / 大卡,split-GPU resident,或 `colocate_with_trainer` 持久)**:
   **开 compile**。worker 常驻 → 只编一次 → 拿全额稳态加速(rollout 1.37× / train 1.26×,§4.3)。
   这是推荐路径——256-step 论文 batch(见 `SPRINT_cosmos_predict25_rl_paper_parity.md`)本来就要
   resident 多卡。
2. **release-after-collect(单卡 colocated、无 colocate 块)**:**也建议开**。worker 每周期重建,
   但首次前向命中 inductor 默认缓存(`/tmp/torchinductor_<user>`)→ 重编译是 **暖的(~0.6s)而非
   冷的**,P1 要量的只是"这个暖 stall vs 周期内 1.37× 省下的"净账(大概率正)。
3. **净建议**:Cosmos Predict2.5 RL → **compile ON**(翻 `rollout.denoise_compile.enable: true`),
   两种生命周期都开;**不加任何 cache 旋钮**(torch 默认缓存已覆盖重编译场景)。仅当 `/tmp` 易失
   (云实例)时,启动脚本 `export TORCHINDUCTOR_CACHE_DIR=<持久路径>`(零代码)。

> 稳态加速已实测为正(§4.3:rollout 1.37×、train 1.26×),所以 resident 下开 compile 是稳赚。
> 真机跑时仍用 §4.1 的 first-vs-later hook 复核绝对值,并保持 `trainer.py` logprob drift guard 绿。

---

## 6. 执行顺序

- [x] **P0 — 接线 + 生命周期落实(本 sprint §1/§2 即证据)。** Predict2.5 compile 接线已通但
  默认关;三态生命周期由拓扑派生(非 public 开关);实验面 = `rollout.denoise_compile.enable`。
- [x] **稳态 compile 加速已实测(§4.3)** — Predict2.5 rollout 1.37× / train 1.26×(RTX 5090,
  `compile_benchmark.py --family cosmos-predict2.5`)。resident 下开 compile 稳赚。
- [x] **rollout compile 已在 Kling RL 实验里翻开** — `online_nft_kling_video_reward.yaml` 加
  `rollout.denoise_compile: {enable: true, mode: default}`(rollout-only,经 loader 实测解析为 true)。
  train replay 仍 eager(`model.torch_compile.enable=false`),其 +1.26× 留作单独一步(需 LoRA+grad-ckpt parity 核)。
- [x] **inductor cache 已查清(§4.2)** — torch **默认**缓存(`/tmp/torchinductor_<user>`)就跨进程
  持久,重建 worker 默认命中暖缓存(实测 1.88s→0.63s)。**结论:不需要 cache 旋钮/注入/代码**;
  原计划的"加 `inductor_cache_dir` 配置 + 注入 worker env"已**作废、并撤销**(过度设计)。
- [ ] **P1 — 现场量 release-after-collect 的"每周期重建 + 首次前向(暖)重编译"净账。** 用 §4.1 的
  first-vs-later hook + worker build 计时,在单卡 colocated 跑 Predict2.5 Kling,确认净收益为正。
- [ ] **P3 — parity 红线现场确认。** resident / release-after-collect 各跑一个真实 Predict2.5 run,
  确认 logprob drift guard 均差 ≤ 0.01(compile 同时作用 rollout+train 同一张 transformer,
  按构造等价,但要现场验)。

---

## 7. 约束 / 红线 / 非目标

- **不动调度架构**:resident / release-after-collect / persistent 三态及 colocate 调度由
  `vrl/ray/` + `vrl/generation/ray/` 现有代码负责(由拓扑派生)。本 sprint 只在三态**之上**测
  compile 净收益 + 重编译成本,**不改派生逻辑,不加 public 开关**。
- **实验面是 config,不是改代码**:测 ON/OFF 翻 `rollout.denoise_compile.enable` 即可。
- **parity 红线**:rollout 与 train 走等价数值路径(`SPRINT_gemm_utilization.md` §5)。
- **mode=default,不碰 CUDA-graph**:reduce-overhead/CUDA-graph 与 PEFT LoRA + grad-ckpt 冲突
  (项目记忆 `project_torch_compile_wan`),全程 default。
- **不把 perf 脚本接进生产**:`inductor_cache_recompile_probe.py` / `compile_benchmark.py` 是测量
  脚手架(`vrl/scripts/perf/`),pipeline 只需 config flag + 既有 `torch.compile` 调用。
- **不扩到其它家族**:聚焦 Cosmos Predict2.5;sd3.5/wan 的 compile 各自在 config 里。

---

## 关键文件引用(已核实行号)

**生命周期(主轴)**
- `vrl/ray/resources.py:303-312, 324-336` — resident/on-demand 拓扑派生 + RayLifecyclePlan
- `vrl/ray/resources.py:1101-1138` — 新公有面 `colocate_with_trainer`;`:1141-1179` 旧 key hard-fail
- `vrl/generation/ray/launcher.py:185-201` — `rollout.mode=="on_demand"` → `with_release_after_collect`
- `vrl/generation/ray/runtime.py:63-89` — `with_release_after_collect`;`:168-187` — `_ensure_runtime` 重建
- `vrl/generation/ray/config.py:40-42`(内部字段)、`:115-121`(从 resources 填)、`:65-72`(互斥校验)
- `vrl/generation/execution/worker.py:51-65` `load_policy`;`:253-278` `_build_executor`(重建+重编触发)

**Predict2.5 compile 接线**
- `configs/model/diffusion/cosmos/predict2_5_2b.yaml:29-31` — `torch_compile.enable: false`
- `vrl/models/runtime_config.py:43` — model block 发给 worker
- `vrl/models/interfaces/runtime.py:112-117` — `RuntimeBuildSpec.torch_compile`（enable 门控）
- `vrl/models/diffusion/cosmos/predict2_5/runtime.py:77-78` — 调 `torch_compile_transformer`
- `vrl/models/diffusion/cosmos/predict2_5/model.py:245-246` — `torch.compile(pipeline.transformer, mode=mode, fullgraph=False)`
- `vrl/generation/ray/launcher.py:384-415` — `_apply_rollout_compile_override`（单向覆盖）
- `configs/base/rollout/diffusion.yaml:19` — `rollout.denoise_compile.enable: false`（默认）
- `vrl/models/diffusion/capabilities.py:61` — Predict2.5 `supports_torch_compile=True`

**测量**
- `vrl/ray/actor_pool.py:140-150` — per-chunk `execution_s` / `queue_wait_s`（first/later 来源）
- `vrl/scripts/common/online.py:328-376` — epoch 循环 + `phase_times`
- `vrl/trainers/online/trainer.py:80-112` — `PhaseTimer`（CUDA-sync wall-clock）
- `vrl/scripts/perf/inductor_cache_recompile_probe.py` — codegen-cache 跨进程冷/暖 A/B（本 sprint 新增）
- `vrl/scripts/perf/compile_benchmark.py --family cosmos-predict2.5` — 稳态 compile A/B（§4.3 实测来源）
- torch 默认 inductor 缓存:`/tmp/torchinductor_<user>`（跨进程持久,无需配置）

**收益基线**
- §4.3 实测（Predict2.5 RTX 5090）：rollout **1.37×** / train **1.26×**，launch 2763→947 / 10098→3838
- `docs/sprints/planned/SPRINT_gemm_utilization.md` P2（Predict2 1.37×/1.25×，与 Predict2.5 一致）
- `docs/sprints/planned/SPRINT_cosmos_predict25_rl_paper_parity.md`（paper-faithful = resident 多卡）
