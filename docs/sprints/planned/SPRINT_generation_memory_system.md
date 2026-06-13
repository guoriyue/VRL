# SPRINT: Generation memory system（设计，未实施）

状态：proposed。前置 `SPRINT_generation_memory_policy.md`（planned，其核心已由
`SPRINT_single_gpu_generation_memory_hardening.md` Part 2 落地）。本篇回答的问题：
静态 policy 之上，"统一内存系统"应该长什么样，向 vLLM / SGLang / SGLang-Omni /
slime / cosmos-rl 各学哪一块。

## 0. 诚实的现状：我们有"机制"，没有"系统"

已有机制（各自正确，互不知晓）：

```text
静态策略   generation memory policy（targets + apply，2026-06-12 统一）
          trajectory storage policy / uint8 wire / reward artifact release
相位交接   release lifecycle（拓扑派生）、driver model offload、frozen offload、
          empty_cuda_cache 在相位边缘
反应降级   chunk OOM split（裂半重试）
观测       per-chunk peak_memory_mb telemetry、host memory snapshot
```

缺的是系统三件套：**预算**（没人声明每相位/角色可用多少字节）、**核算**（peak 遥测
只记录不裁决）、**成本准入**（batch/chunk 尺寸全是数量旋钮，不是字节旋钮——
sbs=8 在 512p 是 23GB、在 704p 是 OOM，旋钮语义随分辨率漂移）。

## 1. 五个框架各拿一块（证据在 docs/sprints/reading/）

| 框架 | 拿什么 | 证据 | 对应我们的痛点 |
| --- | --- | --- | --- |
| vLLM | **先 profile 再定预算**：init 时 `determine_available_memory()` → 据此 size KV pool；调度器把"分配失败"当背压信号（停止准入/抢占） | vllm.md:72,183,229 | 我们的容量边界靠人肉二分（b16 OOM 实验史） |
| SGLang | **准入前查 allocator**：`available_size() >= num_tokens` 不满足先 evict 再查，仍不满足 retract 请求稍后重跑 | sglang.md:363-374 | 我们 OOM 后才反应（split），没有事前准入 |
| SGLang-Omni | **字节计价的批量收集**：encoder 批不按条数、按 `request_cost_fn` 字节成本 + `max_batch_cost`（10GiB×activation 倍率）；**每 stage 显式显存契约**（fraction 总和校验） | sglang-omni.md:304-316,348 | 最直接适用——VAE decode 微批 / denoise chunk 都该按字节预算切 |
| slime | **带标签的暂停/恢复**：`torch_memory_saver.pause()/resume()` 按 tag（WEIGHTS vs KV_CACHE）分级释放，权重留显存、激活让位 | slime.md:75 | 我们的 release lifecycle 是 kill/relaunch 整个 actor（cosmos2.5 NFT 每周期 ~5min 重载） |
| cosmos-rl | **有界暂存队列 + 事件驱动释放**：recv 临时张量入队、超界即 sync+free；buffer 内存被训练进度反向约束（samples_on_the_fly） | cosmos-rl.md:281-283 | 权重同步暂存与 continuous queue 的字节上限（已有 max_bytes，但孤立） |

**不学的**：paged KV / radix cache 本体——那是"跨请求共享前缀 + 逐 token 增长"的
LLM serving 形状；diffusion rollout 的 latents 是按 chunk 整存整取、无前缀共享，
分页买不到东西。`NEURAL_ECS_ENGINE_DESIGN.md` 的 PagedLatentPool 等跨请求
continuous batching 真出现时再议（同 physical stage scheduler 的 gate）。

## 2. 目标形态：三层渐进，每层独立有收益

### L1 预算 + 核算（最便宜，先做）

```text
MemoryContract：从 resolve_distributed_resources 的拓扑结果派生每角色字节预算
  （单卡 colocated：rollout/reward/trainer 分相位共享同一预算；多卡：按卡）
核算：chunk peak_memory_mb 遥测已在 → 对照预算写入 schedule telemetry，
  超预算 warning（先观测一个版本，不 fail）
落点：vrl/ray/resources.py（预算派生）+ executor telemetry（核算）
```

### L2 字节计价的 chunk 准入 + 执行器阶梯（核心增量）

机制归管理：tiling/slicing/decode 微批今天是人肉 bool 旋钮（配置的输入），
目标形态里它们是决策的输出——延迟↔峰值的交换值不值得做只有持有预算的一方
知道。决策按执行器阶梯逐级升级（vLLM 驱逐→停admit→抢占 的同构）：

```text
预算装得下整图 decode → 不 tiling（不白付延迟）
装不下               → 开 tiling / slicing（最便宜的执行器）
仍装不下             → decode 微批、缩 chunk（字节准入）
成本模型估错         → OOM split（安全网，出现率=模型质量指标）
```

配置语义随之三态化：`tiling: auto | true | false`（auto=系统决定，默认；
true/false 保留为复现/硬约束）。人声明约束，系统在约束内选做法。
现有 targets/policy 契约不变——它本来就是"系统对机制的唯一施策点"，
只是施策的依据从"读配置 bool"升级为"读预算+成本模型，受配置约束"。



```text
现有地基：chunk_placement.estimate_chunk_cost 已为 LPT 调度估算相对成本
增量：同一估算函数长出第二个维度 estimate_chunk_bytes(request, chunk)
  （latent 尺寸 × 步数缓冲 × CFG 倍率 + VAE decode 峰值，按家族 capability 标定）
planner 按 L1 预算切 chunk：sample_batch_size 从"人肉数量旋钮"降级为上限，
  实际 chunk 尺寸 = 预算 ÷ 单样本字节成本（SGLang-Omni 的 max_batch_cost 形状）
OOM split 降级为安全网（成本模型标定误差的兜底），不再是首道防线
标定：用 cross-model smoke 的 peak_memory_mb 实测回填系数，不拍脑袋
```

### L3 带标签的相位交接（替换 kill/relaunch）

```text
现状：ReleasableRayGenerationRuntime 整 actor 拆毁重建（NFT 周期 ~5min 重载）
目标：slime 形状——权重 tag 常驻、激活/缓冲 tag 释放；reward 相位只让出激活
  显存，回到 rollout 相位免重载
依赖：torch_memory_saver 或等价的 allocator pause/resume；风险在与
  CUDA Graph / compile 的交互，需单独 spike
收益验收：cosmos2.5 NFT 单步 wall time 中 worker 重建占比归零
```

## 3. 非目标

```text
不做 PagedLatentPool / radix / KV 分页（无跨请求 continuous batching 前提）
不做 OOM auto-tuner 闭环（L2 的成本模型 + split 安全网已覆盖；标定靠 smoke 实测）
不动 model.memory 的 targets/policy 契约（刚落地，是 L1-L3 的地基不是改造对象）
L3 不与多卡 sprint 耦合——单卡 colocated 就有收益（NFT 周期）
```

## 4. 实施顺序

```text
L1 → L2 标定 → L2 准入 → （独立）L3 spike
L1/L2 纯单卡可做可验；每层一个独立 sprint，本篇只是蓝图。
```
