# SPRINT: torch.compile × rollout worker 生命周期（planned）

状态：proposed / planned。结论先行：**compile 已经接进 rollout pipeline 了**（翻 `model.torch_compile.enable:true` 即生效，见 `SPRINT_gemm_utilization.md` 的 P2），不需要从 perf 脚本"搬"任何东西——perf 脚本只测量，真正的 `torch.compile` 调用一直在生产 runtime 里。这份 sprint 回答**剩下唯一没结的集成问题**：稳态 1.37×/1.25× 的收益，在不同 **rollout worker 生命周期**下能不能真拿到——因为有一种模式会**每个 collect 周期重建 worker → 重新编译**，那次 stall 可能吃掉本周期的收益。

> 方法：回到 `vrl/generation/ray/` 与 `vrl/models/` 逐跳核实接线与生命周期分支；compile 收益数字来自 `SPRINT_gemm_utilization.md` P2（已实测，RTX 5090）。本 sprint 不重测稳态收益，只补"生命周期 × 每周期重编译"这一维。

---

## 1. 核心结论 (TL;DR)

**(A) rollout 接线已通，无需移植。** `model.torch_compile.enable:true` 经 `extract_runtime_spec`（把整个 `cfg.model` block 带给 rollout worker）→ family builder 读 `spec.torch_compile` → `torch.compile(self.transformer, mode, fullgraph=False)`。rollout 的 denoise 前向调用的就是这张编译后的 transformer。**perf 脚本（`compile_benchmark.py`）只是测量脚手架，不进 pipeline。**

**(B) 真正的变量是 worker 生命周期，不是接线。** 编译产物**常驻**则 warmup 摊销到 ~0、收满收益；每周期**重建**则每个 collect 阶段重新编译。三种生命周期由两个 config 开关决定：

| 生命周期 | 触发条件 | compile 行为 | 收益 |
|---|---|---|---|
| **split-GPU / resident** | rollout 与 trainer 各占 GPU（非 colocated） | 编一次、常驻 | 全额 ✅ |
| **colocated + release_after_collect** | 单卡共享，每个 collect 阶段把 GPU 还给 trainer | worker 每周期重建 → **重编译** | 每周期扣一次 stall ⚠️ |
| **colocated + persistent_colocated_workers** | 单卡，但保活（新增模式 061cfb2，需 GPU overlap） | 编一次、常驻 | 全额 ✅ |

开关位置：`distributed.rollout.release_after_collect`、`distributed.rollout.persistent_colocated_workers`（→ `RayGenerationConfig`，`config.py:35,37`）。

**(C) 这份 sprint 要交付的：** (1) 量清楚 release_after_collect 模式下"每周期重建+重编译"的 wall time，对比一个 collect 周期内 rollout 总时间，算净收益；(2) 用持久 inductor 缓存（`TORCHINDUCTOR_CACHE_DIR`）把每周期重编译摁便宜；(3) 三种模式各跑一个真实 cosmos run，确认净收益为正且 logprob drift guard 绿。

---

## 2. 背景：接线路径（证据）

```
predict2_2b.yaml  model.torch_compile.enable: true
  └─ extract_runtime_spec: model_config = plain_mapping(cfg.model)        runtime_config.py:43
       （docstring 明示带 path/lora/memory/torch_compile 整块 → 一并发给 rollout worker）
  └─ rollout worker 建模: build_cosmos_predict2_runtime_bundle(RuntimeBuildSpec(model_config=…))  worker.py:267
  └─ spec.torch_compile 读 model_config.torch_compile → {enable, mode}     interfaces/runtime.py:111
  └─ model.torch_compile_transformer(mode)                                 cosmos/predict2/runtime.py:91
  └─ torch.compile(self.transformer, mode, fullgraph=False)                base.py:200
  └─ denoise 前向调用的就是这张 transformer（rollout+train 共用调用点）     common/backbone.py:152
```

**两个 compile knob**（集成面）：
- `model.torch_compile`（全局）：rollout + train replay 都吃（rollout 继承整个 model block）。**最省事，已翻 true。**
- `rollout.denoise_compile`（仅 rollout 覆盖，`launcher.py:374`）：想 rollout 单独开 / 用不同 mode 时用；经 `capability.supports_torch_compile` 门控（cosmos = True，`capabilities.py:61`）。注意 `rollout.denoise_compile.enable:false` 是 no-op（`launcher.py:386` 提前 return），**关不掉**继承来的 `model.torch_compile`。

---

## 3. 问题：P2 测的是稳态，没测每周期重编译

P2 的 1.37×/1.25× 是 **warmup 之后的 per-step 中位数**——它假设编译产物常驻。这对 split-GPU / persistent_colocated 成立，但对 **release_after_collect 不成立**：

- `RayGenerationRuntime.with_release_after_collect`（`runtime.py:61-88`）"recreates Ray workers between collect phases"——每个 collect 阶段为了把 GPU 还给 trainer，**销毁 rollout worker（释放显存）→ 下个周期重建**。
- 重建 = 重新 load checkpoint **+ 重新 `torch.compile`**。model load 本来就是这个模式的固有代价（换 GPU 的代价），compile 是叠在上面的额外 Dynamo trace + guard + 首次 codegen。
- 所以净收益 = （本周期内所有 denoise 前向 × 1.37× 省下的）−（一次重编译 stall）。collect 周期越长（35 步 × 多样本），越摊得动；周期越短，越可能被 stall 吃掉。**这是要测的数,不是要猜的。**

---

## 4. 执行顺序

- [ ] **P0 — 接线 + 生命周期落实（基本已做，本 sprint §2 即证据）。** 确认 cosmos rollout worker 经 `model.torch_compile` 生效（grep 接线已闭环）；列清三种生命周期的 config 触发条件。

- [ ] **P1 — 量每周期"重建 + 重编译"成本（release_after_collect 模式）。** 在真实单卡 colocated + `release_after_collect=true` 配置下，测一次 worker 重建的 wall time（拆成 model-load vs compile-warmup 两段），并测一个 collect 周期的 rollout 总 wall time。算净收益 = 周期内 1.37× 省下的 − 重编译 stall。判据：净收益 > 0 才在该模式默认开 compile。

- [ ] **P2 — 持久 inductor 缓存把重编译摁便宜。** 在 colocated launch 路径设 `TORCHINDUCTOR_CACHE_DIR`（跨 worker 进程持久），让重建时的 codegen 走 cache-hit（只剩 Dynamo trace + guard + 首次 launch）。A/B：设 vs 不设 cache dir 时的重编译 wall time。若显著变便宜，决定是否在 colocated 模式默认导出该环境变量（写进 launcher 的 worker env，而非要用户手设）。

- [ ] **P3 — parity 红线现场确认（沿用 P2 那条）。** 三种生命周期各跑一个真实 cosmos run，确认 `trainer.py:603` logprob drift guard 均差 ≤ 0.01 保持绿（compile 同时作用 rollout+train 同一张 transformer，按构造等价，但要现场验）。

---

## 5. 约束 / 红线

- **生命周期开关互斥**：`persistent_colocated_workers=true` 要求 `allow_driver_gpu_overlap=true` 且 `release_after_collect=false`（`config.py:60-66`），不能同时开。
- **parity 红线**：同 `SPRINT_gemm_utilization.md` §5（rollout 与 train 走等价数值路径）。
- **不重写 worker 生命周期**：resident / release_after_collect / persistent_colocated 三态及其 colocate 调度由 `vrl/generation/ray/` 现有代码负责（另有其 owner，正在演进，见 061cfb2 持久 colocated 模式）。本 sprint 只在这三态**之上**测 compile 的净收益与重编译成本，不动调度本身。
- **reduce-overhead / CUDA-graph 已划掉**：P2 实测 rollout 1.36× ≈ default 1.37×（compute-bound，消 launch 零收益），train 有 grad-ckpt 正确性风险——本 sprint 全程 `mode=default`，不碰 CUDA-graph。

---

## 6. 非目标 (KEEP / 不做)

- 不把 `compile_benchmark.py` / `gemm_projection_breakdown.py` 接进生产路径——它们是测量脚手架（synthetic config-init 模型、计时、数 launch），pipeline 只需要 config flag + 既有 `torch.compile` 调用，二者都已就位。
- 不为 release_after_collect 模式去缓存编译产物到磁盘以外的花活（torch 没有可移植的"编译产物序列化"；`TORCHINDUCTOR_CACHE_DIR` 的 codegen 缓存是现成且足够的杠杆）。
- 不扩到其它家族——sd3.5/wan 的 compile 已各自在 config 里（部分默认开），本 sprint 聚焦 cosmos predict2 的生命周期净收益。

---

## 关键文件引用

**接线（证据）**
- `vrl/models/runtime_config.py:43`（+ docstring 34-35）— `model_config = plain_mapping(cfg.model)`，整块 model（含 torch_compile）发给 rollout worker
- `vrl/generation/execution/worker.py:267` — rollout worker 经 `build_runtime_bundle(RuntimeBuildSpec(...))` 建模
- `vrl/models/interfaces/runtime.py:111-117` — `spec.torch_compile` 读 `model_config.torch_compile`
- `vrl/models/diffusion/cosmos/predict2/runtime.py:91-94` — `model.torch_compile_transformer(mode)`
- `vrl/models/diffusion/base.py:200` — `torch.compile(self.transformer, mode, fullgraph=False)`
- `vrl/models/diffusion/common/backbone.py:152` — `_call_transformer`（rollout+train 共用）
- `vrl/generation/ray/launcher.py:371-405` — `rollout.denoise_compile` 覆盖（`_apply_rollout_compile_override`）

**生命周期（本 sprint 的主轴）**
- `vrl/generation/ray/runtime.py:32-39` — resident 与 release-after-collect 双生命周期 docstring
- `vrl/generation/ray/runtime.py:61-88` — `with_release_after_collect`：collect 阶段之间重建 worker
- `vrl/generation/ray/config.py:35,37` — `release_after_collect` / `persistent_colocated_workers` 开关
- `vrl/generation/ray/config.py:60-66` — 互斥约束
- `vrl/models/diffusion/capabilities.py:61` — diffusion 家族 `supports_torch_compile=True`

**收益基线**
- `docs/sprints/planned/SPRINT_gemm_utilization.md` 的 P2 实测结果（稳态 1.37× rollout / 1.25× train；CUDA-graph 划掉；跨家族对比）
