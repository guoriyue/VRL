# SPRINT: AR vs Diffusion 决策记录 + 分层整理 — 方向裁决与层图固化

**日期**: 2026-07-13  **状态**: LAYOUT LANDED（2026-07-18）；causal-chunked policy 仍待真实实现
**触发**: "AR 是未来、diffusion 画质不行，是否砍掉 diffusion 支持 / 是否把 ar 放到
diffusion 之上重新分层"。本文把方向裁决和层图一起钉死，防止反复重开。

---

## 1. 问题分解：三个被捆在一起的命题

| 命题 | 裁决 | 依据 |
|---|---|---|
| **C1** 因果/序列组织是世界模型的未来 | **接受（~95%）；方向与 taxonomy 已落地，执行尚未落地** | 交互性数学上要求因果；provider + self_forcing 两个 sprint 仍是待完成的执行兑现 |
| **C2** 离散 token 化取代连续 diffusion 作为每步生成数学 | **暂不接受（~30%），设 tripwire** | VQ 瓶颈未破；世界模型阵营（Cosmos3 = AR reasoner + diffusion generator、FlashDreams 7 因果模型全是 AR-chunk × diffusion-step）在因果组织下仍选 diffusion 出每步 |
| **C3** 视觉生成并入统一多模态 LLM 栈 | **观望（~50%）** | 图像侧信号强（4o 类），视频侧 token 量级问题未解 |

**核心认知**：行业收敛形态是 **causal 时间组织 × continuous denoise step**——
因果组织回答"怎么组织序列"，continuous denoise 回答"每一步怎么生成"。应拥抱
AR 传统里的**因果性**，而不是把它与**离散性**绑定；离散化是当前的画质税，
因果性才是未来。

## 2. 仓位决定（决策记录，即刻生效）

- **主仓位 ~80%**：causal_chunked × denoise（CausVid 首接、MAGI-1 contract reference、
  exact chunk-wise Self-Forcing variants + provider 路线），
  兼顾 joint-denoise 试验台（sd3_5/flux/sana 是 RL 正确性的 CI，39/12/5 个
  真实 run）。
- **对冲 ~20%**：causal categorical-token 策略保温不冷冻——token-GRPO 五家族保留；nextstep_1
  （AR 骨干 + flow 头，恰是 C2 翻转时最可能的中间形态）优先级略升。
- **不做**：禁用 denoise 策略支持（会拆掉主仓位的地基）；全面转向 categorical token
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

## 4. 分层问题："AR 是 diffusion 之上的技术，不该同级"——轴分离已落地

"ar" 与 "diffusion" 不再是模型 taxonomy。registry 分类的是一个 entry 暴露的
**可训练、产生 RL action 的策略变体**，并显式记录：

```text
                       denoise step          token step
joint             │ current policies      │ （当前无）
causal            │ （当前无）             │ current policies
causal_chunked    │ CausVid/MAGI-1 等目标格 │ （未来可存在）
```

时间组织、policy step、action distribution 与 trajectory layout 分开记录。
multisegment 是 trajectory layout，不是假装成第四种时间组织。分类范围是 trainable
policy，而不是整个 checkpoint：GLM-Image 是 causal categorical-token policy，后接
frozen joint-denoise renderer；Cosmos3 当前训练的是 joint-denoise vision stream。
权威定义与完整映射见 [`MODEL_TAXONOMY.md`](../../MODEL_TAXONOMY.md)。

物理布局也已从历史两桶迁到 family-first + 正交执行层：

```text
vrl/math/{denoise,token}/

vrl/models/
  families/<family>/          family-owned model/replay/runtime
  steps/{denoise,token}/      shared model contracts and builders

vrl/generation/
  steps/denoise/              config + denoise loop + TeaCache
  steps/token/                token-step protocol
  composition/causal/         current ordered-prefix token state machine
  bindings/joint_denoise/     joint × denoise request/executor/gatherer
  bindings/causal_token/      causal × token request/executor/gatherer
  execution/ ray/             axis-neutral execution and lifecycle

vrl/config/presets/{model,experiment}/<family>/
```

家族 executor 是 composition × step 的具体 binding；registry 通过显式 import path
选择 binding，不从目录名猜。扁平 `collector_kind` 已删除。旧
`Diffusion*`/`AR*` 类名可作为实现 API 保留，但新 import 不再使用
`models/{diffusion,ar}` 或 `generation/{diffusion,ar}` 路径。

当前只抽取了已经有多个真实 family 消费的 `composition/causal/token_loop.py`。
joint 的 stage orchestration 仍只有 `joint_denoise` 一个真实 owner，因此没有为对称性
制造空 `composition/joint.py`；`causal_chunked` 也等首个可执行 policy 再落。token
scheduler 的同-position 分组不是天然可复用于 denoise chunk 的通用因果抽象。

**固化的层图**：

```text
L0  vrl/math/{denoise,token}/          action likelihood and sampling math
L1  vrl/generation/{steps,composition,bindings,execution,ray}/
                                       policy execution and engine contracts
                                       ── 禁止反向依赖 rollout/trainer
L2  vrl/models/families/<family>/      backbone, private state, replay projection
    vrl/models/steps/{denoise,token}/  shared model-side contracts/builders
L3  vrl/families/registry + rollouts/trainers
                                       dispatch and RL lifecycle
```

backbone（DiT/UNet/Transformer）仍是 family 内部实现细节，不增加一根 taxonomy 轴。
新 causal-chunked family 应放 `vrl/models/families/<family>/`，并注册为
`(causal_chunked, denoise, continuous, denoise)`；不要新建
`models/diffusion/<family>/`。候选的证据、顺序与 release gates 见
[`MODEL_TAXONOMY.md`](../../MODEL_TAXONOMY.md#future-causal-chunked-support)。

## 5. 已完成与剩余项

1. **已完成：taxonomy。** `PolicySemantics` 是 source of truth；生产分发读取
   semantics 或显式 binding/capability，旧 `collector_kind` 已删除。
2. **已完成：family-first 物理重排。** model、test、preset 路径按 family 组织；共享
   math/model/generation 代码按 step、composition、binding ownership 组织。
3. **已完成：现有 composition 抽取。** causal-token state machine 已进入
   `generation/composition/causal/token_loop.py`；denoise hot loop 已进入
   `generation/steps/denoise/loop.py`。
4. **待真实实现：causal-chunked denoise。** 首个技术候选是 Wan2.1-1.3B-based
   CausVid，因为它最贴近现有 Wan seam；但 upstream WIP 状态和 non-commercial
   checkpoint license 是 promotion gate。MAGI-1 先作为 24-frame causal-chunk contract
   reference，再评估其 custom/larger runtime。exact chunk-wise Self-Forcing variant 仍是
   候选，但不能用 family 总称替代 variant 分类。首个 family 在自己的 binding 内拥有
   专用 chunk lifecycle；只有第二个实现的 diff 证明了共性，才上提共享 composition，
   不从四格表猜接口。
5. **决策记录维护。** §1–§3 的 tripwire 只在季度 review 或触发事件时重开。

## 6. 架构卫生与非目标

- **应改变**：新 family/import/config/test 一律使用 family-first 与轴化路径；遗留活跃
  文档不再把 AR/diffusion 写成互斥模型种类。
- **保持不变**：薄 family runtime/runner、binding facade、gatherer 保留，因为它们是
  lazy-import、协议、tensor adapter 或 driver/worker 边界；跨 family 一致性比省 LOC
  更重要。
- **ALL_CAPS**：`FAMILY_REGISTRY` 是刻意隔离的 taxonomy/config table；
  `GENERIC_DENOISE_EXECUTOR` 是跨 neutral registry/worker 的 import-path protocol，均
  保留。不要再维护平行 `SUPPORTED_MODELS`/`SUPPORTED_KINDS` 大表。
- **非目标**：不禁用任何模态，不把 UNet/DiT 变成分类层，不批量重命名仍有消费者的
  `Diffusion*`/`AR*` 符号，不创建无真实 owner 的对称空模块。

## 7. 验证

- registry mapping tests 覆盖现有兼容投影及 synthetic causal-denoise，防止 taxonomy
  重新退化成 AR/diffusion 两桶。
- architecture tests 强制 generation 不反向 import rollout/trainer，并检查旧 package
  前缀不回流。
- config lint 与 family registry/interface tests 从 family-first preset/import path 解析。
