# SPRINT：对照大厂后训练报告的实施清单（planned）

状态：**planned / 未开始（2026-09-16）**。来源与证据在三份 reading：
[[SPRINT_image_post_training_reports]]、[[SPRINT_video_post_training_reports]]、
[[SPRINT_diffusion_rl_method_notes]]。每项写清：出处、VRL 现状（已核对 HEAD）、
要做什么、验收。按"被多家独立采用"和"VRL 缺口大小"排序；单家的、或纯 RM 训练侧
的不列。

规则：每项落地前重读定义、生产者、消费者、测试与 dotted-string 引用（README 约束）。
新增的 knob 走一处 seam 解析默认值，不加 `x/default_x` 对，不加 helper 模块。

---

## A. 多 reward 的 advantage 级合成 ★★★

- **出处**：DanceGRPO `A_i = Σ_k (r_i^k − μ^k)/σ^k`；Flow-Factory 的 GDPO 式逐 reward
  归一；MixGRPO 等权多 reward "更稳"；DanceGRPO 教训：HPS 单独用出油腻伪影，要 CLIP
  一起。Seedance 三 RM 也是 composite。
- **VRL 现状**：`MultiReward` 出加权总分，分量原始分数已在 `RewardOutput.components`
  里；`GroupAdvantageEstimator` 只见总分。
- **做什么**：
  1. 把 `components` 从 collector 一路带到 `RolloutBatch`（核对是否已带）。
  2. `GroupAdvantageConfig` 加 `advantage_aggregation: Literal["weighted_sum",
     "per_reward_normalized"]`，后者对每个分量各自组归一后按权重求和。
  3. 指标里分别记每个分量的组均值/方差，方便看 reward hacking 是哪一路。
- **验收**：两路合成在单 reward 时数值相同（回归测试）；双 reward 合成的 advantage
  等于各自归一之和；`tests/algorithms` 通过。

## B. 组内共享初始噪声 ★★★

- **出处**：DanceGRPO——视频上不共享会发散并加剧 hacking；Flash-GRPO 的"同组同
  timestep"思路同源。
- **VRL 现状**：生成侧无开关。`v_grpo.py` 的 `_group_shared_noise` 只在 loss 侧。
- **做什么**：`rollout.group_shared_noise: bool`（默认 False 保持现状）；在
  full_sequence_denoise 的 executor 按 prompt 组派生同一初始 latent（组内样本索引
  不进种子）。SDE 的每步噪声仍各自独立。
- **验收**：开启时同组样本第 0 步 latent 逐位相等、不同组不等；关闭时行为不变
  （`tests/generation` 有现成 layout 测试可扩）。

## C. 直接 reward 反传目标（REFL / Direct-Align） ★★★

- **出处**：Seedream 2.0/3.0/4.0、Seedance 1.0 都以"直接最大化多 RM 输出"为主力，且
  明确说比 DPO/PPO/GRPO 有效；HunyuanImage 3.0 的 SRPO 是其单步版本；SRPO 报告 32 卡
  10 分钟收敛。
- **VRL 现状**：没有任何可微 reward 反传路径（`algorithms/` 只有 GRPO 族、NFT、DPO、
  V-GRPO）。reward 走 reward service 时梯度不可达。
- **做什么**（分两步，先图像）：
  1. `algorithm.kind: reward_backprop`：rollout 到随机 t（`timestep_selection` 复用），
     用 Direct-Align 闭式 `x₀ = (x_t − σ_t ε)/α_t` 单步恢复，VAE 解码，进**同进程**
     可微 RM（HPSv3 / PickScore / aesthetic 已在 `rewards/models/`），loss = −reward。
     需要 VAE 解码开梯度（`VaeDecodeMemoryPass` 的 tiling 要允许 autograd）。
  2. **语义相对偏好**选项：CLIP 系 RM 上 `r = f_img·(C_pos − C_neg)`，
     `reward.semantic_relative: {positive: str, negative: str}`。
  3. 稳定性：EMA（已有）、timestep 选早期区间、reward 归一。
- **不做**：视频版（reward 经视频 VAE 反传显存不可控），先在 SD3.5 / FLUX 上证明。
- **验收**：SD3.5 + HPSv3 上 200 步内 held-out reward 上升 >2σ，且 PickScore 不降
  （防单 RM hacking，配合 A）。

## D. Best-of-N 训练样本筛选 ★★

- **出处**：DanceGRPO 从 16/64/256 池只训 top-k + bottom-k，加速收敛。
- **VRL 现状**：只有 `_nonzero_advantage_mask`。
- **做什么**：`actor.group_select: {keep_top: int, keep_bottom: int}`，在 collector
  → trainer 之间用 `RolloutBatch.select` 按组内 reward 排序保留；被丢的样本仍计入
  组统计（μ/σ 用整池算）。
- **验收**：keep_top=keep_bottom=G/2 时等价于不筛；单测覆盖组统计不受筛选影响。

## E. 训练步数少于推理步数（denoising reduction） ★★

- **出处**：Flow-GRPO 训练 T=10、推理 T=40，4× 无损；SRPO 训练 25 / 推理 50。
- **VRL 现状**：`sampling.num_steps` 只有一份；`config/schema.py` 里没有 eval 级
  sampling 覆盖。
- **做什么**：加 `eval.sampling.num_steps` 覆盖，其余继承训练 sampling。
- **验收**：训练 10 步、eval 40 步的配置能过 `tests/config` 的 schema 校验并实际生效。

## F. 通用 VLM "Yes 概率" 图像 RM ★★

- **出处**：Seedream 3.0（RM 从 CLIP 换成 VLM，reward = 归一化 Yes token 概率，1B→20B
  scaling）。
- **VRL 现状**：`videocon_physics.py` 有 P(Yes)/(P(Yes)+P(No)) 模式但绑定视频物理
  rubric；`qwen_vl_judge.py` 是生成式解析。
- **做什么**：`rewards/models/vlm_yes_prob.py`：复用 `qwen_vl_judge` 的加载与 chat
  构造，rubric 问题可配置，输出 P(Yes)；支持 `device_map` 分卡。
- **验收**：tiny Qwen2-VL 上 P(Yes)+P(No)=1 的单测；与 `videocon_physics` 共用
  Yes/No token 解析（把它的 token 预解析上提到 judge 前置）。

## G. 训练中的 bad-case 导出（RM 迭代闭环的 policy 侧） ★★

- **出处**：Seedream 2.0、Seedance 1.0 的多轮 policy↔RM 迭代："在优化后的模型上标注
  bad case → 训 bad-case-aware RM → 再优化"。
- **VRL 现状**：`rewards/artifacts.py` 能落盘样本；没有"按 reward 分位 / 按分量分歧
  定期导出带 prompt 与各分量分数的样本包"。
- **做什么**：`artifacts.export_bad_cases: {every_n_steps, per_step, rule:
  lowest_total | max_component_disagreement}`，导出图/视频 + prompt + 分量分数 JSON。
  备注项：collector 前的可替换 prompt rewriter（Seedream PE、Grok Imagine 访谈都指出
  线上效果一大块来自改写），离线预处理实现即可。
- **验收**：导出目录结构固定、可被人工标注工具直接读；不影响训练吞吐（异步落盘）。

## H. Recipe 级串联：offline DPO → online GRPO；SFT + 模型汤 ★

- **出处**：Qwen-Image、HunyuanImage 3.0（DPO 修结构 → GRPO 提美学）；Seedance、
  Kandinsky、Movie Gen（分桶 SFT 后权重平均）。
- **VRL 现状**：offline DPO 和 online GRPO 都有，但没有把前者 checkpoint 作为后者
  起点的 preset；无 SFT trainer、无权重平均脚本。
- **做什么**：
  1. preset：`dpo_then_grpo` 两段配置，第二段 `model.init_from` 指向第一段产物。
  2. `vrl/scripts/soup.py`：对同架构 checkpoint 做等权 / 根号权平均（Kandinsky 结论）。
  3. SFT trainer 暂不做（VRL 是 RL 库；若要做，是 `trainers/offline/sft.py` 的
     flow-matching loss，与 DPO trainer 同形）。
- **验收**：两段 preset 跑通 smoke；soup 脚本对两份 LoRA 权重平均后可加载。

## I. CFG 下训练分支审计 ★

- **出处**：DanceGRPO——CFG 让训练不稳；FLUX/HunyuanVideo 关 CFG；CFG 依赖模型要同时
  优化 cond+uncond。
- **VRL 现状**：`DenoiseCFGMode = batched_cfg | separate_cfg | single_branch`；
  未审计 replay 是否对 uncond 分支也计算策略比值。
- **做什么**：读 `models/steps/denoise/common/backbone.py` 与 replay 路径，写清三种
  模式下 log-prob 用的是哪个分支；若 replay 只走 cond 而 rollout 用了 CFG，记为
  已知偏差并在 recipe 默认 `guidance_scale=1`。
- **验收**：一页 done 文档 + 若有偏差则加 config 校验警告。

## J. MixGRPO 的滑动窗口调度 + 窗后高阶 ODE 求解器 ★

- **出处**：MixGRPO——窗口从低 SNR 向高 SNR 每 τ=25 步移 s=1（progressive-constant
  最好）；窗口之后的 ODE 段用 DPM-Solver++ 再省 ~40% 时间，窗口之前不能。
- **VRL 现状**：`sde_window` 是固定区间（`loop.py:250`），无滑动；窗外 Euler。
- **做什么**：
  1. `rollout.sde.window_schedule: {shift_every: int, stride: int}`，由 trainer 步数
     驱动窗口位置，collector 每次请求带当前窗口。
  2. `rollout.sde.post_window_solver: euler | dpm_solver_pp`，仅作用于窗口后的步；
     replay 不涉及（这些步不进策略比值）。
- **验收**：窗口位置随步数单调移动的单测；开启高阶求解器后 held-out reward 与
  Euler 在噪声内，rollout 时间下降。

## K. 评测口径：best-of-k、方差、按属性分层 ★

- **出处**：Seedream 4.0（单次均分低但 best-of-4 最强，"greater variability"）；Movie
  Gen bench 按运动量分层；HunyuanImage SSAE 按 12 字段报准确率。
- **VRL 现状**：held-out eval 报均值（`SPRINT_sana_aesthetic_trustworthy_curve` 一线）。
- **做什么**：eval 报告加 best-of-k（k=4）、组内方差、以及按 prompt 标签（若数据集
  有）分层均值。
- **验收**：同一 eval 产物能回算出三种口径；不改训练。

## L. previous policy 脱离 LoRA（NFT / V-GRPO 全参） ★★ — 代码已落地 2026-09-19，待 GPU 验证

- **出处**：A–K 之外的内部缺口。工业界（Seedream、Seedance、HunyuanImage、
  Qwen-Image）全部全参后训练；VRL 里 NFT 与 V-GRPO 是仅有的两个不依赖
  log-prob 比值的目标，却被 `runtime.py:373 require_lora_for_previous_policy_adapter`
  锁死在 LoRA 上。
- **VRL 现状**：算法需要的只是"一份冻结的上一步策略的 forward + 每步以 decay
  刷新"。实现绑在 PEFT 第二 adapter 上：`attach_previous_policy_adapter` /
  `sync_previous_policy_adapter` / `activate_adapter("previous")`（`base.py:490–524`），
  算法基类也叫 `PreviousAdapterObjective`。`trainers/online/ema.py` 已经在维护
  "可训练参数的 decay 副本"，只是没有换入 forward 的接口。
- **做什么**：
  1. 模型侧契约改为 `sync_previous_policy(decay)` + `previous_policy()` 上下文，
     不出现 adapter 字样；实现是"可训练参数的 shadow 副本 + forward 期间换入"，
     LoRA 时 shadow 就是 adapter 参数（等价现状，不再需要第二个 PEFT adapter），
     全参时 shadow 是全部可训练参数（显存 +1× 可训参数；FSDP2 下按 shard 保存）。
  2. 去掉 `require_lora_for_previous_policy_adapter`；`PreviousAdapterObjective`
     改名 `PreviousPolicyObjective`。
- **验收**：SD3.5 LoRA 上 NFT 的 first-step invariant 与现状数值一致；全参 NFT
  在 tiny 模型上 lr=0 invariant 通过；FSDP2 两卡 smoke。
- **状态**：`vrl/models/policy_snapshot.py` + `DenoiseModelBase.previous_policy /
  reference_policy / sync_previous_policy`；PEFT 第二 adapter、`previous_policy_adapter`
  build 字段、LoRA 门全部删除；tiny Wan 上 LoRA 与全参的 lr=0 invariant 都过
  （`tests/algorithms`）。未做：FSDP2 两卡 smoke（DTensor 换入换出走的是 EMA 已验证的
  `copy_` 路径，但没有在卡上跑过）；全参 rollout worker 的 `cache_ref_noise_pred`
  现在会报"no reference policy"，需要 worker 侧在 build 时拍参考快照。

---

## 已有、不需要做的（防止重复立项）

- GRPO / Flow-GRPO SDE + 闭式 KL、clip、组归一 advantage：`GRPO`。
- 随机 timestep 子集（DanceGRPO τ）：`timestep_selection=random`，done sprint 已验证机制。
- MixGRPO 窗口：`rollout.sde.window_*`、`timestep_selection=sde_window`。
- GRPO-Guard、Flash-GRPO、FlowDPPO、DiffusionNFT、V-GRPO、offline DPO。
- EMA：`trainers/online/ema.py`。
- 20B 级 VLM judge 分卡：reward service 远程路径 + `device_map`。
- 多维视频 RM：`kling_video_reward`、`videoscore2`、`unified_reward_video`、
  `motion_dynamics`。

## 建议顺序

A → B → E（都是小改、且是 C 的前提：C 的验收要靠 A 防单 RM hacking）→ C（最大
缺口，图像先）→ L（与 C 同为全参路线的前提）→ D → F → G → K → I → J → H。
