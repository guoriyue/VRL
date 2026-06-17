# SPRINT: Whole memory plan (current repo)

状态：plan。本篇是面向当前仓库实际状态的**完整内存计划**——从已落地的机制
统一，到尚未实现的内存系统。每个 Phase 独立成 sprint、单卡可验。

## 0. 诊断：有机制，无系统；而且机制只统一了一半

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
核算   peak_memory_mb 遥测只记录，从不裁决
准入   batch/chunk 是数量旋钮非字节旋钮——sbs=8 在 512p=23GB、704p=OOM，
       旋钮语义随分辨率漂移；dispatch 前无人回答"这个 chunk 装得下吗"
```

---

## Phase 0：trainer 侧 frozen offload —— ❌ reverted 2026-06-13（伪需求）

2026-06-12 曾把 frozen offload 收进与 VAE 对称的"声明 target → 单一 policy"框架
（`trainer_frozen_targets()` + `apply_trainer_memory_policy` + `frozen_module.py`）。
2026-06-13 整套删除：见上节诊断——trainer 用 ReplayModel，这些模块从未加载，policy
全程空转，不是有效省显存方案。

删除清单：

```text
vrl/trainers/frozen_module.py                              整文件删
DiffusionModelBase.trainer_frozen_targets()                删
sd3_5/wan/cosmos train.py 的 apply_trainer_memory_policy   删（_after_bundle_built 只剩 grad-ckpt）
configs/model/diffusion/sd3_5/medium.yaml 的 frozen_offload 删
MODEL_MEMORY_SECTIONS                                       收回为 ("vae_decode",)
```

保留：所有 ReplayModel / `build_replay_bundle` / `minimal_replay_bundle_metadata`
（`generation_only_modules` 是"不加载"的长期证据）、`model.memory.vae_decode`
（generation 侧真实消费者）。测试改为断言 replay bundle 声明 generation-only 模块
absent（`test_minimal_replay_runtime_wiring.py`）。

教训：判断"机制未统一"前，先确认两侧是否真有同一个被管理对象。generation 侧
VAE 在 trainer 与 rollout 都存在；trainer 侧 text encoder/VAE 只在 rollout 存在，
强行"统一"会把一个根本不存在的对象的管理逻辑也铺过去。

---

## Phase 1（L1）：预算 + 核算（最便宜，先做）

```text
预算   ResolvedDistributedResources 派生 role_memory_budget_mb：
       单卡 colocated = 卡显存 × 安全系数（相位串行共享）；多卡按角色卡数
核算   chunk peak_memory_mb 遥测对照预算 → 写入 ray_chunk_schedule，
       超预算 warning（先观测一版不 fail，收集标定误差分布）
落点   vrl/ray/resources.py（预算）+ vrl/generation/ray/executor.py（核算）
```

---

## Phase 2（L2）：字节准入 + 执行器阶梯（核心增量）

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
地基   estimate_chunk_cost（为 LPT 建）已在 chunk_placement.py
增量   长出 estimate_chunk_bytes(request, chunk, capability)
       —— latent_h×w×frames × 步数缓冲 × CFG 倍率 + VAE decode 峰值
标定   用 cross-model smoke 的 peak_memory_mb 实测回填系数（现以静态 yaml
       数字死在 6 个配置里——把它从"人肉抄配置"变成"成本模型系数"）
       注意：denoise 激活=持续高原、VAE decode=短时尖峰，可能需两个系数；
       成本模型须读 policy metadata 知道 tiling 是否已应用（闭环已埋）
准入   planner 按预算切：chunk_samples = min(sbs, budget ÷ per_sample_bytes)
       sbs 降级为上限；tiling 配置三态化 auto|true|false（auto=系统决定）
```

验收 gate：predict2 704p 不手调 sbs 直接跑，planner 自算 chunk 尺寸，零或个位
数 OOM split。

---

## Phase 3（L3）：标签式相位交接（替换 kill/relaunch）

```text
现状   ReleasableRayGenerationRuntime 整 actor 拆毁重建（NFT 周期 ~5min 重载）
目标   slime 形状：权重 tag 常驻、激活/缓冲 tag 释放；reward 相位只让出激活
       显存，回 rollout 相位免重载
依赖   torch_memory_saver 或等价 allocator pause/resume；风险在与 CUDA Graph/
       compile 的交互，需单独 spike
验收   cosmos2.5 NFT 单步 wall time 中 worker 重建占比归零
```

---

## 框架借鉴（证据在 docs/sprints/reading/）

| 学谁 | 拿什么 | 对应 Phase |
| --- | --- | --- |
| vLLM | profile 定预算；分配失败=背压 | L1/L2 |
| SGLang | 准入前查 allocator，先 evict 再 retract | L2 |
| SGLang-Omni | 字节计价批量 + 每 stage 显存契约 | L2（最对症） |
| slime | tag 暂停/恢复（权重常驻、激活让位） | L3 |
| cosmos-rl | 有界暂存队列 + 事件驱动释放 | L1/L3 |

不学：paged KV / radix——diffusion latents 按 chunk 整存整取、无跨请求前缀共享，
分页买不到东西（等真有 continuous batching 再议，同 physical stage scheduler gate）。

---

## 顺序与边界

```text
L1（预算核算）→ L2 标定 → L2 准入 → （独立）L3 spike
（Phase 0 已撤销，见上——trainer 用 ReplayModel，无需 frozen offload。）
L1/L2 单卡可验。L3 独立 spike，不与多卡耦合（单卡 NFT 周期就有收益）。
每个 Phase 一个独立 sprint。
```

## 非目标

```text
不做 PagedLatentPool/radix/KV 分页（无跨请求 continuous batching）
不做 OOM auto-tuner 闭环（L2 成本模型 + split 安全网已覆盖，标定靠 smoke）
不动 model.memory 的 target/policy 契约本身——它是 L0-L3 的地基
不把 batch_size/storage/release flags 搬进 model.memory（各有其位）
```
