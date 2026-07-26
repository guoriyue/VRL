# SPRINT: Single-GPU generation memory hardening

状态：Part 2 / Part 3 implemented（2026-06-12，本仓工作区）；Part 1 owned by the
wm-infra working-tree process（见 §1 分工边界）。

> **当前状态修正（2026-07-25，`003ad92e`）。** 本文 §2–§3 的
> `memory_config`、`vae_decode_memory_from_config`、`_VAE_DECODE_KEYS` 是当时实现记录，
> 不是当前 API。现在 YAML typo 由 closed Pydantic schema 拒绝，registry 唯一解析出
> `GenerationMemoryPolicy`，Ray worker 重建 typed policy，runtime 只调用
> `configure_vae_decode_memory` 施加行为。family target contract 和 fail-loud 语义保持不变。

## 0. Core Decision

没有多卡的前提下，把单卡 generation/training 的 memory 行为收束成**可控、统一、
fail-fast**。三块范围，按所有权切开执行，与 wire diet / OOM split 的并行纪律一致：
谁的工作树里有在飞改动，那块就归谁。

## 1. 分工边界（避免两进程撞车）

```text
Part 1（config/scripts 层）—— wm-infra 工作树进程（文件已在其工作区改动中）:
  vrl/config/loading.py            ??? mandatory marker fail-fast
  vrl/scripts/common/online.py     可选 key（model.use_lora 等）统一 OmegaConf.select(default=...)
  vrl/scripts/diffusion/cosmos/train.py  trainer 侧 frozen offload 完成测试与文档

Part 2 + 3（models 层）—— 本仓（vrl2），本 sprint 落地:
  vrl/models/diffusion/common/vae_decode_memory.py + base.py
  5 个 family model.py / 5 个 runtime.py / 测试
```

## 2. Part 2 — generation memory policy 统一（已落地）

### 改动前的问题

`apply_vae_decode_memory` 在 5 个 family `model.py` 的 `from_build` 里各调一次
（wan 两个类共 6 处），loader 持有 `memory_metadata` 状态，runtime builder 再用
`getattr(model, "memory_metadata", None) or {}` 拼回 bundle——施策点、状态、上报
散在三层，每个新家族都要重抄一遍，漏抄即静默无 tiling（predict2 704p OOM 的根因
正是这个模式漏了一家）。

### 改动后的契约

```text
family model   只声明 WHAT：generation_memory_targets() -> {"vae_decode": <vae>}
               （base.py 单一实现：pipeline.vae 优先、anima 的 self.vae 兜底、
               replay 模型 pipeline 抛错→无 targets；家族可 override 扩展）
policy         只负责 HOW/WHEN：apply_generation_memory_policy(model, memory_config, owner)
               —— 解析 model.memory.vae_decode、应用到 target、返回 bundle metadata；
               配置了 vae_decode 但模型无 target ⇒ ValueError（不再静默）
runtime builder 唯一施策点：构造 model 后调一次 policy，metadata 直接进 bundle；
               loader 不再 import policy、不再持有 memory_metadata
```

设计取舍：
- targets 发现放 base 单一实现而非 5 份同体 override（本周 dedup 纪律）；
  replay 模型的 `pipeline` property 抛 RuntimeError 被显式接住——这是
  "replay 不拥有 VAE" 的契约表达，docstring 写明。
- `apply_vae_decode_memory` 旧入口删除（零调用方残留）；
  `configure_vae_decode`/`vae_decode_memory_from_config` 保持为 policy 内部件。
- metadata key（`vae_tiling`/`vae_slicing`）是下游契约，逐位不变。

## 3. Part 3 — 架构测试（已落地，tests/models/diffusion/common/test_vae_decode_memory.py）

```text
unknown memory key fail loud        （已有：_VAE_DECODE_KEYS 派生校验）
配置 vae_decode 但无 target fail loud（新增：missing-target ValueError pin）
family model.py 禁止 import policy / 持有 memory_metadata（新增：源码结构 pin）
所有 runtime builder 必须走共享 policy（新增：源码结构 pin）
existing configs 全部 load         （已有 tests/config/test_load_all_experiments.py，
                                    该文件在 wm-infra 工作区改动中，归 Part 1）
```

## 4. 验证

```text
tests/models + tests/config: 205 passed
全量: 745 passed / 8 skipped（含 4 个新 pin）
```

## 5. Non-Goals

```text
不做 physical stage scheduler（无多卡时只能验证 queue/contract，证不了 overlap 收益）
不做 slime overlap T3（排本 sprint 之后：per-collect phase times 从
  collector.last_collect_phases 挪到 item/batch 上）
不动 frozen offload 的 models 层（trainer 侧归 Part 1）
不为 text encoder / transformer 增加新的 memory target 种类——等真实需求
  （704p 之后的下一个 OOM 点）出现再扩 targets dict，接口已就位
```
