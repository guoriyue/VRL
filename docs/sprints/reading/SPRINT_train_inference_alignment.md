# SPRINT: 训推差异是 RL 的基石问题 —— 对标我们扩散/视频 RL 的现状与缺口

状态：reading + 决策（2026-06-25）。性质：**研读外部文章（Ling Team 萧轩《RL 老训崩？训推差异是基石》）+ 逐项对标我们 infra 的训推对齐现状**，产出一个判断和一份未来门控清单，不是功能移植。

> 来源文章：萧轩《RL 老训崩？训推差异是基石》知乎，2025-10-30（Ling Team / inclusionAI）。
> 核实路径：本仓库 `vrl/algorithms/logprob_mismatch.py`、`vrl/trainers/online/precision_guard.py`、
> `vrl/math/diffusion/flow_matching.py`、`vrl/generation/diffusion/executor.py`、
> `vrl/rollouts/evaluators/diffusion/sde_logprob.py`、`vrl/scripts/common/factory.py`。
> 相关：[[SPRINT_fullparam_and_fp8_precision]]（精度轴 + TIS/RS 地基，已落地）、
> [[SPRINT_moe_support_decision]]、[[SPRINT_physical_ai_model_support]]、
> [[SPRINT_speculative_diffusion_rollout]]（whole-latent exact-coupling 已测得负结果）、
> [[SPRINT_framework_lessons_vrl]]、
> 记忆 `project_first_trustworthy_curve`（Cosmos+Kling GRPO 实跑 NO learning）。

## 0. 一句话

**文章的论点对 LLM-AR + 双引擎（Megatron vs vLLM）成立，但对我们当前的扩散/flow 家族基本不适用——因为我们采样和训练走的是同一套 DiT forward + 同一个 scheduler 对象，根本没有第二套推理引擎去产生算子级分歧。** 我们的训推差异已经坍缩成**唯一一根轴：精度**（bf16/fp8 rollout vs fp32 logprob 数学），而这一根轴已经被 `logprob_mismatch.py`（TIS/RS 修正）+ `precision_guard.py`（漂移哨兵）完整覆盖。文章的全套"逐模块算子对齐"工作量，要等我们引入 **AR/MoE 推理引擎**或**独立的投机扩散引擎**时才会从 0 变成 live risk——这份 sprint 就是把那个门提前标出来。

## 1. 文章的核心论点（提炼）

- **训崩 = reward 曲线骤降**，常被归因复杂、无从下手；文章把它统一归到**训推差异**这一基石问题。
- 即便 RMSNorm / RoPE / Attention / KVCache / LM_Head / MoE router 这类标准组件，训练框架（Megatron/FSDP）和推理框架（vLLM/SGLang）的**实现也多多少少不同**，误差逐层累积放大，极端时同一 token 训推概率分别为 0 和 1，理论上的 on-policy 不成立。
- **两个放大器**：① MoE —— 训推两边专家选择不同导致误差阶跃；② 长推理 —— 输出越长误差累积越大。故 `MoE + 长输出` 与 `Dense + 短输出` 是两类难度。
- **解法重心在框架层而非算法层**：相同逻辑实现 + 合适精度 + 消除不确定性，逐模块逐激活值对账（prefill↔prefill、prefill↔decode、不同并行下对齐三阶段）。
- **对齐到位后算法无需改**：可以**直接用 rollout probs 当 PPO 分母**，省掉用训练引擎重算 training probs 的那次前向，且训练后期 reward 更高、训推差异更稳（文章图 3）。verl / OpenRLHF 的"重算 training probs"是退而求其次、实质有偏。
- 训推对齐与所有算法（GRPO / GSPO / TIS）**正交**，是未来 RL 框架的前提；off-policy / 异步场景下对齐后框架只需处理"参数更新带来的差异"。

## 2. 逐模块对标：文章的对齐点 vs 我们的现状（已用代码核实）

| 文章的对齐点（LLM-AR） | 我们扩散/flow 的对应 | 现状（证据） |
|---|---|---|
| 训练框架 vs 推理框架是两套实现 | **不成立**：采样和训练都走同一个 DiT `forward_step` / `replay_forward`；scheduler 是**同一个对象** | `factory.py:202` 把 generation 用的 `scheduler` 直接传给 `DiffusionSDELogProbEvaluator`；离散化逐项一致 |
| KVCache 精度（FP32 累加 vs BF16 初始化） | 扩散无 KV-cache 自回归累加；但有 logprob 的高斯密度数学 | `flow_matching.py:58` `model_output.to(md)`，`math_dtype` 默认 fp32，bf16/fp8 强制 upcast 后再算 log-density |
| LM_Head 需 FP32 softmax | 无 softmax 分类头；对应的是 SDE step 的高斯 log-prob | 同上，math 轴恒 fp32 |
| RMSNorm / RoPE / Attention 训推实现差异 | **当前无分歧**：同一份 module，同一次前向定义 | 采样 `executor.py:720 model.forward_step` 与训练 `sde_logprob.py:78 model.replay_forward` 是同一模型的两次调用 |
| MoE router 不稳定排序 / 专家选择分歧 | **当前家族无 MoE**（sd3_5/wan/cosmos 都是 dense DiT） | —— 见 §4 门控 |
| 长输出误差累积 | 去噪步 T（如 35 步）的误差累积 | `fp8_rollout_drift_probe.py` 实测：单步 ratio_dev ~0.14，沿 T=35 累积后 max ~0.87 |
| 直接用 rollout probs 当分母 | **我们已经是这样**：PPO 分母用采样时存的 logprob | `grpo/continuous.py:105` `raw_ratio = exp(signals.log_prob - old_log_probs)`，`old_log_probs` 来自采样 `executor.py:770` 存入、`gather.py:102 old_log_prob=log_probs` |

**关键结论**：我们走的不是文章批评的 verl/OpenRLHF "纯重算" 范式，而是**已经把 rollout probs 当分母**（分子是带梯度的 replay(θ)，分母是采样时存下的 behavior logprob），再用 **TIS 截断 + RS 拒绝**把 rollout→replay 的精度 gap 显式 bound 住。这正是文章承认的修正路线，且比它批评的 verl 范式更进一步。

## 2.5 "算子 set 相同"怎么保证（评论区"本质上还是算子 set 不一致"的正面回答）

文章评论第一条点破：训推差异**本质是算子 set 不一致**。我们保证算子 set 相同的方式不是"逐模块对齐两套实现"，而是**只留一条前向定义 + 用逐位数值测试做证伪兜底**。

### 结构性保证：采样和 replay 汇聚到同一个 `transformer.__call__`
文章那些团队要逐模块对（RMSNorm/RoPE/Attention），是因为他们**真有两套 kernel**（Megatron vs vLLM）。我们没有第二套：
- 采样 `sd3_5/model.py:281 forward_step` → `DiffusionBackboneCaller(transformer, runner)`
- 训练 `base.py:117 replay_forward` → `self.forward(state, 0)` → **同一个** `forward_step`

CFG 也同源：两边都 `batched_cfg`，都 `torch.cat([uncond, cond])` 拼 2x batch（`common/cfg.py:67`），shape 一致 → kernel 触发点一致。LoRA 两边都不 merge、adapter 都挂着 → 同一条 low-rank GEMM。**所以算子 set 相同对我们是 by-construction，不是对齐努力的产物。**

### 但"同一个 module"≠"算子 set 一定相同"——4 个自动选择的轴
同一个 module，采样和训练照样可能踩到不同 kernel：

| 轴 | 现状 | 是否已治 |
|---|---|---|
| **SDPA 注意力后端** | **没 pin**（`executor.py` 用 diffusers SDPA，无 `sdpa_kernel(...)`）。PyTorch 按 grad-mode/shape 自动在 flash/mem-efficient/math 间挑；采样 `no_grad`（`executor.py:706`）vs 训练 grad-enabled 可能选到不同后端 → 数值微差 | ❌ **真口子**，见 §4 行动 |
| **autocast dtype 来源** | 采样从 `prompt_embeds/latents` dtype 推断（`executor.py:692`），训练从 `trainer.train_precision` config（`trainer.py:195`）。默认相等就一致；故意拉开就是**精度轴** | ✅ 就是 §3 的精度轴，TIS/RS/guard 已覆盖，非新问题 |
| **cosmos 双 model 实例** | cosmos predict2 有 `build_..._runtime_bundle`（生成）+ `build_..._replay_runtime_bundle`（训练）两个独立实例，各自 `compile_transformer(model, mode)`。compile 配置同源但**两次独立编译**，无断言保证融出同图。sd3_5/wan 是 colocated 单 module 无此问题 | ⚠️ 同 mode + ratio 测试兜底，低风险 |
| **train()/eval() 模式** | 训练显式 `model.train()`（`trainer.py:745`），采样不设留默认。扩散 transformer 无 dropout/BN 故当前**无行为差异**——但是"碰巧安全"不是"强制安全" | ❌ 见 §4 行动 |

### 兜底：用 ratio==1 逐位测试证伪，而非静态证明
不去静态证明算子 set 相等，而是直接量端到端 logprob 是否真相等，绕过"到底哪个 kernel 跑了"：
- `test_diffusion_flow_matching.py:85`：精度一致时断言 replay vs 采样 logprob **fp32 逐位相等**（ratio==1）。
- `precision_guard.py:run_precision_drift_guard`：精度一致时 mode=`off` 期望逐位齐，不齐 fail-fast。

**这条测试跑过 = 那条前向路径上所有算子的合成结果一致**，不管中间踩了 flash 还是 math。它是算子 set 对齐的最终证伪器。

## 3. 我们已落地的训推对齐机器（无需重造）

- **测量**：`vrl/algorithms/logprob_mismatch.py:compute_logprob_mismatch_stats` —— `fresh(replay) - old(rollout)` 的 logprob 漂移、ratio 偏离、mismatch_kl。
- **修正**：同文件 `apply_truncated_importance_weight`（TIS：off/truncate/clip/mask）+ `apply_rejection_sample_mask`（RS：seq_mean_k1 / seq_max_k1），两个 GRPO 家族共用。
- **哨兵**：`vrl/trainers/online/precision_guard.py:run_precision_drift_guard` —— 首步逐 timestep 校验；`resolve_guard_mode` 在 `rollout_precision != train_precision` 时自动转 `fail`，精度一致时 `off`（此时 `test_diffusion_flow_matching.py:85` 已断言 replay vs rollout logprob 在 fp32 下逐位相等）。
- **自动派生**：`build_trainer_config` 在 BF16 training 与 BF16+FP8-quantized rollout 的 stage policy 不同且无显式 expert block 时，自动注入 `PrecisionCorrectionConfig(tis_mode="truncate", rs_mode="seq_mean_k1")`（见 [[SPRINT_fullparam_and_fp8_precision]] §2）。
- **诊断探针**：`vrl/scripts/perf/fp8_rollout_drift_probe.py` —— 真 `_scaled_mm` fp8 GEMM 测漂移 → 喂 stats → guard 触发 → TIS 在轨迹尾部 engage。

> 一句话：**精度这一根轴上，文章要的"测量+修正+不确定性门控"我们都有了。**

## 4. 真正的缺口 = 三个未来门（这是本 sprint 的价值）

文章的"逐模块算子对齐方法论"现在对我们是**休眠资产**，下列**门 1/2** 任一落地会把它整套激活，必须在引擎接入**之前**把 parity harness 备好，否则就是文章描述的"训崩、无从改起"。**门 0** 是当前就敞着、可现在收口的算子 set 小口子（详见 §2.5）。

### 门 0：两个没 pin 的算子 set 轴（现在就能收口）
- **SDPA 后端没 pin**：采样（`no_grad`）和训练 replay（grad-enabled）依赖 PyTorch 自动选 flash/mem-efficient/math，可能选到不同后端。行动：在 `vrl/generation/diffusion/executor.py` 的去噪段和 `vrl/rollouts/evaluators/diffusion/sde_logprob.py` 的 replay 段外面都套**同一个** `torch.nn.attention.sdpa_kernel([...])` context，把后端锁成一致。当前靠 `test_diffusion_flow_matching.py:85` 的 ratio==1 兜底，但那是事后证伪，不是事前锁定。
- **train/eval 模式没强制**：replay 前显式 `model.eval()`（或断言模型无 train-only 模块）。当前扩散 transformer 无 dropout/BN 故无害，但属"碰巧安全"——加一个带 dropout 的家族就静默分叉。
- 非目标：不动 cosmos 双实例 compile（同 mode + ratio 测试已兜底）和 autocast 精度轴（就是 §3 精度轴，TIS/RS 已覆盖）。

### 门 1：引入 AR / MoE 家族 + 真正的推理引擎
- 触发：[[SPRINT_physical_ai_model_support]]、[[SPRINT_moe_support_decision]]、NextStep AR（记忆 `project_cross_model_smoke`）。
- 后果：一旦 rollout 走 vLLM/SGLang 式 KV-cache decode、训练走 FSDP/Megatron prefill，文章的**全部**根因回归——算子集分歧、prefill↔decode 差异、MoE router 不稳定排序、长序列累积。我们现在"scheduler 同对象 → 离散化一致"的保证**消失**。
- 备料（前置工作，不是现在做）：把 `precision_guard` 从"只比 logprob 标量"扩成**逐模块激活值对账**的 parity harness，按文章三阶段（prefill↔prefill → prefill↔decode → 不同并行下）跑。MoE 要额外做：router 高精度 + 稳定排序（替换 `torch.topk`）+ 固定 token permutation 加和顺序。

### 门 2：未来新的独立投机扩散 / paged 扩散引擎

- 当前状态：[[SPRINT_speculative_diffusion_rollout]] 的 whole-latent exact-coupling 已在真实
  SD3.5 上测得 acceptance≈0、省算 0%，原型已删除；它是关闭的历史案例，不是 live trigger。
- 重新触发：未来出现采用不同算法、拥有独立 execution path 的 speculative/paged diffusion
  engine 时，才重新打开此门。
- 后果：若新引擎使用 draft 模型 + 验证步，rollout 轨迹的 logprob 可能不再等于训练 replay
  的 logprob——引入 prefill/decode 式分歧，打破“采样和 replay 用同一个 scheduler 严格重算”的前提。
- 备料：新引擎产出的 logprob 必须能被训练 replay **精确重放**，或显式纳入 TIS 分母；
  integration 前先跑 `fp8_rollout_drift_probe` 同款 parity 测试。

### 门 3：批不变性（batch-invariance）—— 已 live 的隐性漂移
- 来源：Thinking Machines Lab《Defeating Nondeterminism in LLM Inference》（文章评论引用）—— 归约顺序 / batch size 不同会引入非确定性，即使同一套 forward。
- 我们这里 live：采样的 `chunk_batch` 与训练的 `microbatch_size` 不同 → 同一 DiT 前向在不同 batch 形状下归约顺序可能不同，产生**精度一致时仍存在的**微小 logprob 漂移。当前被 drift guard 的 fp32 逐位断言掩盖（因为测试用同 batch）。
- 行动（小，可现在做）：在 `fp8_rollout_drift_probe` 加一个**变 batch 形状**的 parity case，量"同精度、不同 batch"下的 ratio_dev，确认是否在 TIS cap 之下；若显著则记录为已知项。

## 5. 一个可直接验证的结论（省 recompute）

文章主张"对齐后直接用 rollout probs，省掉重算 training probs"。**对我们扩散家族，这个省法的一半已经达成**：PPO 分母本就是采样存下的 rollout logprob（§2）。但分子 `replay(θ)` 带梯度、必须重算——这次前向省不掉（梯度从这里来）。所以我们能省的那部分已经省了，无需再动。`test_diffusion_flow_matching.py:85` 的"精度一致时 ratio==1 逐位相等"即此结论的回归保证。

> 这条要写进结论而不是 TODO：**扩散家族的训推对齐在精度一致时是 by-construction 成立的，不存在文章那种"对齐红利尚未兑现"的空间可挖**——红利只在精度故意拉开（fp8 rollout）时才需要 TIS/RS，且已自动派生。

## 6. 非目标

- 不为当前 dense 扩散家族新建任何算子对账 harness——它现在是休眠资产，门 1/2 触发时才建。
- 不改 `logprob_mismatch` / `precision_guard` / TIS/RS 现有语义（[[SPRINT_fullparam_and_fp8_precision]] 已锁定）。
- 不把"直接用 rollout probs"包装成新功能——分母早已是 rollout probs。
- 不引入 LLM-PPO 风格的 token 级长序列累积假设到扩散 loss。

## 7. 文献与模型引用

### 主文章
- 萧轩，《RL 老训崩？训推差异是基石》，知乎，2025-10-30（Ling Team / inclusionAI）。

### 论文 / 技术报告
- Ling Team, *Ring linear models technical report*, arXiv:2510.19338 —— https://arxiv.org/abs/2510.19338
- Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*（batch-invariant kernels / 累加顺序一致 → 消除推理非确定性；文章评论引用）。
- Yingru Li, *When …*（训推差异分析，§2）—— https://yingru.notion.site/When

### 模型（inclusionAI 开源混合线性）
- Ring-mini-linear-2.0 —— https://huggingface.co/inclusionAI/Ring-mini-linear-2.0
- Ring-flash-linear-2.0 —— https://huggingface.co/inclusionAI/Ring-flash-linear-2.0

### 配套工程博文（Ling Team）
- 《FP8 混合线性训练，MFU 起飞金手指》—— 训推两侧套同一份高效 FP8 融合算子，从根上对齐（与本仓库 [[SPRINT_fullparam_and_fp8_precision]] 的 fp8 rollout kernel 思路同源）。
- 《FLOOD：探索大模型离线推理吞吐极限》—— 离线推理吞吐。

### 相关算法
- GRPO、GSPO、TIS（Truncated Importance Sampling）—— 文章强调三者与训推对齐正交；TIS/RS 已在本仓库 `vrl/algorithms/logprob_mismatch.py` 落地。

## 相关
- [[SPRINT_fullparam_and_fp8_precision]] —— 精度轴 + TIS/RS + drift guard + fp8 rollout kernel（本 sprint 引用的对齐机器都在那落地）
- [[SPRINT_moe_support_decision]]、[[SPRINT_physical_ai_model_support]] —— 门 1 触发源
- [[SPRINT_speculative_diffusion_rollout]] —— 门 2 的历史关闭案例；未来新独立 engine 才触发
- [[SPRINT_framework_lessons_vrl]] —— 同类外部框架研读
- 代码：`vrl/algorithms/logprob_mismatch.py`、`vrl/trainers/online/precision_guard.py`、`vrl/math/diffusion/flow_matching.py`、`vrl/generation/diffusion/executor.py`、`vrl/rollouts/evaluators/diffusion/sde_logprob.py`、`vrl/scripts/perf/fp8_rollout_drift_probe.py`
