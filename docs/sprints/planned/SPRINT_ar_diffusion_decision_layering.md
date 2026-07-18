# SPRINT: AR vs Diffusion 决策记录 + 分层整理 — 方向裁决与层图固化

**日期**: 2026-07-13  **状态**: PLANNED（§4 为决策记录即刻生效；§5 变更清单为轻量文档/分类动作）
**触发**: "AR 是未来、diffusion 画质不行，是否砍掉 diffusion 支持 / 是否把 ar 放到
diffusion 之上重新分层"。本文把方向裁决和层图一起钉死，防止反复重开。

---

## 1. 问题分解：三个被捆在一起的命题

| 命题 | 裁决 | 依据 |
|---|---|---|
| **C1** 因果/序列组织是世界模型的未来 | **接受（~95%）**，且已落地 | 交互性数学上要求因果；provider + self_forcing 两个 sprint 就是 C1 的兑现 |
| **C2** 离散 token 化取代连续 diffusion 作为每步生成数学 | **暂不接受（~30%），设 tripwire** | VQ 瓶颈未破；世界模型阵营（Cosmos3 = AR reasoner + diffusion generator、FlashDreams 7 因果模型全是 AR-chunk × diffusion-step）在因果组织下仍选 diffusion 出每步 |
| **C3** 视觉生成并入统一多模态 LLM 栈 | **观望（~50%）** | 图像侧信号强（4o 类），视频侧 token 量级问题未解 |

**核心认知**：行业收敛形态是 **AR(时间/因果) × diffusion(空间/连续值)**——AR 赢
"怎么组织序列"，diffusion 赢"每一步怎么生成"。"拥抱 AR"应拥抱其**因果性**，
不是其**离散性**；离散化是当前的画质税，因果性才是未来。

## 2. 仓位决定（决策记录，即刻生效）

- **主仓位 ~80%**：因果 × diffusion（self_forcing 家族 + provider 路线），
  兼顾双向 diffusion 试验台（sd3_5/flux/sana 是 RL 正确性的 CI，39/12/5 个
  真实 run）。
- **对冲 ~20%**：离散 AR 保温不冷冻——token-GRPO 五家族保留；nextstep_1
  （AR 骨干 + flow 头，恰是 C2 翻转时最可能的中间形态）优先级略升。
- **不做**：禁用 diffusion 支持（会拆掉主仓位的地基）；全面转向离散 AR
  （进最拥挤赛道、弃已领先的稀缺生态位、且 32GB 上无可执行实验）。

期权结构：现路线在 C2 翻转/不翻转两个未来里资产都存活（AR chunk 调度、
KV cache、trajectory 轴、reward 栈、weight sync 全可迁移，且离散似然模块已建）；
全面 AR 只在翻转的未来里存活。

## 3. Tripwires（触发任一 → 正式重开本决定；未触发不重想）

1. 出现开源、≤7B、质量对标 diffusion 的**离散 token 视频**模型（tokenizer 突破实锤）。
2. Cosmos/Wan 级世界模型团队下一代**生成器**改用 token-AR。
3. 自家 self_forcing RL 撞上离散模型没有的结构性墙（连续 latent reward hacking、
   credit 无法分配）。
4. 4o 级统一模型出现开源等价物且对视觉 RLHF 响应良好（token-GRPO 立即升为
   最高杠杆目标）。

## 4. 分层问题："AR 是 diffusion 之上的技术，不该同级"——轴分离，层图如下

**正确的核心洞察（采纳）**：现在的 "ar" 与 "diffusion" 兄弟结构各自融合了
两根正交的轴——**每步似然**（gaussian denoise | categorical token）×
**时间组织**（bidirectional 单发 | causal-chunked AR+cache）：

```text
                     gaussian step          categorical step
bidirectional   │ generation/diffusion/ │      （不存在）
causal-chunked  │  self_forcing 目标格   │  generation/ar/
```

现在的"ar" = causal×categorical 融合体，"diffusion" = bidirectional×gaussian
融合体。self_forcing 要填左下空格（causal×gaussian），兄弟分类法在那一刻
开裂：chunk 推进、KV cache 生命周期、ar_index 调度这些 **causal 组织机器**
目前长在 token 侧，diffusion 侧即将复用。"AR 是 diffusion 之上的技术"的准确
表述：**causal 组织是独立的一根轴，不与每步似然同层**。

**唯一的边界修正**："lower level 全是 diffusion"对世界模型类成立，对离散
token 家族不成立——janus/emu3/llamagen 的底层是 categorical softmax
（`models/ar/janus_pro/model.py:47` 只 import `vrl.math.ar.logprob`），
下面没有 diffusion。右下格真实存在；它的存续与家族 Tier 决定耦合
（若离散家族降为对冲/冻结，则"主线模型的底层皆 diffusion"成立）。

**数学层已经体现该分层（不需要动）**：`vrl/math/` 同时服务两个世界——
`math/diffusion/`（SDE/flow logprob、ddim）与 `math/ar/`（categorical
logprob + flow_matching）；实证：nextstep_1 直接消费
`vrl.math.ar.flow_matching.flow_sample_with_logprob`
（`models/ar/nextstep_1/runner.py:15`）。

**目标结构（终态蓝图）**：

```text
vrl/generation/
  step/            ← 每步似然轴
    denoise.py     ← 现 generation/diffusion/executor 的 denoise 循环核心
    token.py       ← 现 generation/ar/decode_loop 的 token 步进核心
  composition/     ← 时间组织轴
    bidirectional.py  ← 单发全序列（现 diffusion 侧隐含形态）
    causal.py         ← chunk 推进 + 携带状态(cache)生命周期
  execution/ ray/ pipeline/  ← 不动（轴无关的请求/chunk/引擎契约层）
```

家族 executor = composition × step 的薄绑定；registry 的一维
`CollectorKind` 拆为两字段：`step_kind: denoise|token` +
`composition: bidirectional|causal|multiseg_r1`（旧值映射：
diffusion=(denoise,bidirectional)、ar_discrete=(token,causal)、
ar_continuous=(token+flow头,causal)、ar_r1=(token,multiseg_r1)、
self_forcing=(denoise,causal) 新格）。"ar" 一词从分类学退休——它同时指
两件事（token 步/因果组织），拆开后不再需要；`models/{diffusion,ar}/`
目录物理不动，含义重定义为 step 轴（"ar"=历史名，语义=token）。

**拆轴的节奏（evidence-gated，不预拆）**：
① 本 sprint 只把四格表 + 上述终态蓝图钉进文档与 registry 注释；
② self_forcing 落地时先在 generation/diffusion 内实现 causal 组织，
   **故意允许与 ar/decode_loop 的调度机器重复**——借此测量真共性
   （chunk 序列推进、携带状态生命周期、双 finish 防护）与假共性
   （token 调度器的同 position 分组对 diffusion 无意义）；
③ 拿到 diff 后只上提测量出的共性到 composition/causal.py，step 核心
   各自下沉，两个旧 executor 变薄绑定，enum 同步拆两字段。
   **省钱变体**（与 Tier 决定耦合）：若届时离散家族已冻结，只提取
   self_forcing 所需，token 侧保持融合——冻结代码不为对称性重构。

**固化的层图**（唯一权威版本，写入 CONTRACT/架构文档）：

```text
L0  vrl/math/            似然与采样数学（diffusion 高斯步 | AR categorical | AR flow 头）
                         ── 依赖：无。这是"core 是 diffusion(数学)"成立的层。
L1  vrl/generation/      执行器与引擎契约（diffusion denoise 循环 | AR decode 循环 |
                         chunk/请求/launch contract/Ray）
                         ── 依赖 L0；禁止依赖 L2/L3（架构边界测试已强制）。
L2  vrl/models/<likelihood>/<family>/   家族 = 骨干前向 + 采样态 + replay 投影
                         ── 目录按**每步似然族**分（diffusion/ = 高斯步；ar/ = token 序列），
                            不按时间组织分。backbone（DiT/UNet/…）是 L2 内部细节，
                            契约只有"在 (x_t, t, cond) 上出 flow/logits"，不设"unet 层"。
L3  vrl/families/registry + trainer/rollout   分发与 RL 生命周期
```

**放置规则（回答未来所有"放哪"问题）**：目录归属看**每步似然**——
self_forcing 每步是高斯 renoise ⇒ 落 `models/diffusion/self_forcing/`
（其 sprint 已如此写）；它的"AR 性"（chunk 组织、KV cache）表达在 L1 执行器
与 registry 的 kind 上，不表达在目录树上。

## 5. 变更清单（全部轻量，无包移动）

1. **层图入文档**：§4 的 L0–L3 图 + 放置规则 + 依赖方向写入 CONTRACT.md
   （或 docs/ 架构页），并注明由
   `tests/architecture/test_generation_rollout_boundaries.py` 强制 L1↛L3。
2. **分类轴显式化**：`vrl/families/registry.py:17` 的
   `CollectorKind = Literal["diffusion","ar_discrete","ar_continuous","ar_r1"]`
   是"时间组织 × 每步似然"两根轴的扁平投影。本 sprint 只在其定义处加注释表
   （bidirectional×gaussian=diffusion；token×categorical=ar_discrete；
   token×flow-head=ar_continuous；multiseg=ar_r1；**causal-chunked×gaussian=
   自 self_forcing 起的新 kind**）。是否把 enum 拆成两个字段，推迟到
   self_forcing 落地时按实际分发需求裁——不做投机重构。
3. **决策记录归档**：§1–§3 即为记录；tripwire 清单在季度 review 或触发事件时
   核对一次。
4. **composition 层上提（显式延迟项）**：按 §4 的节奏③，self_forcing 落地后
   diff diffusion 侧新写的 causal 组织与 `generation/ar/` 既有 chunk 机器，
   把测量出的共性上提为共享 composition 层；届时另立执行 sprint，本文 §4
   四格表即其蓝图。
5. **相关但未决**（不在本 sprint 内定）：家族支持分层（Tier1 世界模型 /
   Tier2 算法试验台 / Tier3 零使用冻结）——注意其与 §4"右下格存续"耦合，
   建议一并裁决。

## 6. 非目标

- 本 sprint 内不移动 `vrl/models/` 与 `vrl/generation/` 任何包——拆轴重排
  只在 self_forcing 提供第二个真实消费者之后按 §5.4 执行（从重复中提取，
  不从猜测中提取）。
- 不引入 "unet 层" 或 backbone 抽象层——backbone 是 L2 家族内部细节。
- 不禁用任何模态支持。
- 不预先把 CollectorKind 拆成二维结构（同上，落地时按分发需求裁）。

## 7. 验证

- 文档/注释变更为主：架构边界测试保持通过（注意：该测试当前在 HEAD 上因
  `f37e4e93` 的 launcher 越界 import 已红——属既有失败，本 sprint 不背锅，
  但落层图文档时应一并修复或豁免，使"层图 + 强制手段"同时为真）。
- registry 注释表与现有四 kind 一一对应，新 kind 名在 self_forcing sprint
  落地时启用。
