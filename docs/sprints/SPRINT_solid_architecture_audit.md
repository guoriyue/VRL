# SPRINT: 全仓 SOLID / 架构审计（主索引）

状态：A/C/D implemented（2026-06-09，commit ed2bba0，子 sprint doc 已随实现删除）；B 暂跳过（用户决定，doc 保留）。审计完成于 2026-06-08。

本文件是**索引 + 决策记录**，不直接写实施步骤。具体改动在 4 个子 sprint 里，每个独立可执行。

## 0. Core Decision

对全仓 `vrl/`（264 文件 / 4.5 万行 / 1728 函数 / 663 个 `_` 前缀函数）做一次跨切面 SOLID 审计，结论是：

> **骨架健康，不推倒重来。只在正确骨架上收口几处*系统性*裂缝。**

12 个模块簇里 5 个（`diffusion-wan-sd3` 的状态层、`generation-diffusion-exec`、`trainers`、`rollouts` 的核心、`algorithms`）被判**无严重违例**——protocol 边界（`vrl/engine/interfaces.py`、`nn/layers/attention/paged.py`）、跨家族一致性（三套 sampling-state dataclass、并行 GRPO 变体）都是**有意设计**而非腐化，按 AGENTS.md「一致性优于清理」原则**严禁动**（见 §6）。

## 1. 审计方法（可复现）

- **手段**：12-cluster fan-out workflow，25 个子代理，870 次工具调用。每簇先「清点 + 批判」，再用独立的对抗式 agent 逐条**尝试否决**，过滤误报（把「协议边界薄函数 / 跨家族有意一致 / schema 常量」全部判 reject）。
- **守则**：每个 agent 都被注入 AGENTS.md 哲学约束——`_` 私有函数只在单文件用是**正常的**，只有跨文件泄漏或一个文件堆太多才算味道；薄函数在协议/适配/一致性边界要**保留**；常量只有「复制了类型结构」才该派生。
- **产出**：36 条 confirmed 发现（4 high / 19 medium / 13 low）+ 10 个 god-file 候选。

分类统计：

| 类别 | 数量 |
|---|---|
| duplication（跨家族真重复） | 10 |
| single_use_indirection | 7 |
| god_module | 5 |
| srp_violation | 5 |
| ocp_violation | 5 |
| dip_violation | 3 |
| constant_should_derive | 1 |

## 2. 唯一的系统性模式

**「家族入口类 / 模块持续吸附本应*下沉到共享层*或*上移到执行层*的职责」**，在三个方向反复出现：

- **(a) god-class / god-module**：契约方法和基础设施细节塞进同一对象。
  `JanusProModel`(1300/31)、`ray/resources.py`(932/26)、`scripts/data/danbooru.py`(1798/65)、`kling_video_reward.py`(770)、`VllmDecoderPagedAttentionBackend`(612/20，实现协议只占 2 方法)。→ **子 sprint B**
- **(b) 跨家族真重复未下沉**：`apply_lora`（实测 6 处）、`_require_tensor`（4 处）、`_to_builtin`、`_embed`/`_ar_runner` 逐字节复制，因为缺 `diffusion/common/` · `ar/common/` 薄共享层。→ **子 sprint C**
- **(c) 字符串/字段名分支替代多态**：`generate_with_refine`(308行)、`_extract_logprobs`、`_apply_rollout_compile_override`、`RootConfig._cross_field_validate`、`JanusProR1PipelineExecutor`。→ **子 sprint D**

外加一条独立的、**最该先做**的 DIP 收口（rollouts 编排层 `getattr` 走对象内部嵌套结构探测能力，而不是问 protocol）→ **子 sprint A**。

## 3. 已亲自核实的关键证据（写入子 sprint 前抽查）

| 论断 | 证据 | 核实 |
|---|---|---|
| rollouts 三级 getattr 走链 | `collector/core.py:160-166`、`orchestration/lifecycle.py:121-135` | ✅ 一字不差 |
| `apply_lora` 多家族复制 | `wan_2_1/model.py:140`、`sd3_5/model.py:144`、`cosmos/{predict2:141,predict2_5:185+584,anima:158}`、`base.py:193`(no-op) | ✅ 实测 6 处（比报告还多） |
| `_require_tensor` 复制 | `cosmos/predict2_5/runner.py:91`、`cosmos/predict2/runner.py:115`、`sd3_5/runner.py:93`(带 `name`)、`wan_2_1/runner.py:148` | ✅ 4 处 |
| `PIXEL_SIZE` 可派生 | `janus_pro/model.py:54-59`：`TOKEN_NUM=576`(24×24)、`PATCH_SIZE=16`，注释自写「→ 384 px」 | ✅ `int(576**0.5)*16 = 384` |

## 4. 优先级（影响 × 确定性）

1. **A** — rollouts DIP 收口（最低 LOC、最高确定性、防静默失配）
2. **B1** — `danbooru.py` 拆分（重复+god 最严重）
3. **D1** — `_extract_logprobs` 改多态
4. **D2** — `JanusProR1PipelineExecutor` 改 Strategy
5. **C1** — `apply_lora` 下沉
6. **B2** — `ray/resources.py` 拆 5 类
7. **B3/B4** — `VllmDecoderPagedAttentionBackend` / `JanusProModel` 拆分
8. **D3** — `generate_with_refine` 提取 `RefinementPolicy`（与 B4 合做）
9. **C2–C5** — 其余跨家族 helper 下沉
10. **D4/D5** — `cross_field_validate` 注册表化、compile-override 注册表化

## 5. 子 Sprint 索引

| 子 sprint | 主题 | 状态 |
|---|---|---|
| **A** | rollouts 编排层 DIP 收口（getattr → protocol） | ✅ done（含审计漏掉的第三处 `runtime_is_colocated`；契约测试 `tests/rollouts/test_runtime_protocol_contract.py`） |
| **B** `SPRINT_solid_god_file_splits.md` | 5 个 god-file 拆分 | ⏭️ 暂跳过（用户决定） |
| **C** | 跨家族真重复下沉到共享层 | ✅ done（T6 与 B 绑定一并跳过；predict2_5 的 apply_lora 有意偏离保留） |
| **D** | 字符串分支 → 多态 / 策略 | ✅ D1 done；D2-D5 评估后不实施（D2 热路径需 golden、D3 绑 B、D4/D5 YAGNI） |

A/C/D 的子 sprint doc 已随实现删除（详细落地记录见 commit ed2bba0 及删除前的 doc 历史）。

low 级 single-use 内联（`_max_batch_size`/`_require_hooks`/`_build_sequences`/`_dtype_label`/`select_tensor_tree` 等）**不开专项 PR**，路过对应文件时顺手处理。

## 6. Non-Goals（看似味道、实为有意设计，严禁清理）

- **三套 sampling-state dataclass / 并行 GRPO 变体**：有意的跨家族并行 shape，提升 grepability/调试，**保留**。
- **`anima/model.py`(664行) 单模块 bundling**：反映 Anima 单文件 checkpoint 设计（docstring line 3），架构性而非偶然；仅当该模式扩散到其他家族才拆。
- **`trajectory/validation.py`(418行)**：当前内聚合理（单入口 `validate_batch()` + 共享 `tensor_refs` cache）。
- **`diffusion` executor 薄 hook、`_import_from_path` 双份、`_copy_adapter_weights` 文件内复用、`MISSING` sentinel 本地重建**：审计明确判为「per philosophy 可接受」。
- **`JANUS_IMAGE_PATCH_SIZE=16` / `TOKEN_NUM=576` / `VOCAB_SIZE`**：真实模型架构维度常量，**保留**（只 `PIXEL_SIZE` 改派生，见子 sprint C）。

## 7. 关键参考

- 审计原始报告（完整 §4 重复对照表 + 每个 god-file 拆分方案）：workflow `wf_aea1665b-211` 输出。
- 仓库哲学源：`AGENTS.md`（Architecture Hygiene / Long-term Assets vs One-shot Validation）。
