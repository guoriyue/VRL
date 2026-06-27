# Reading: Speculative Sampling for Diffusion Models (De Bortoli et al. 2025)

状态：**P0 done（2026-06-24）**。这份 reading note 是
`docs/sprints/planned/SPRINT_speculative_diffusion_rollout.md` 的 P0 产出：把
*Accelerated Diffusion Models via Speculative Sampling*
(https://arxiv.org/abs/2501.05370, PMLR v267) 的算法读到“能实现 toy version”，
并明确它和 VRL `flow_matching.sde_step_with_logprob` 的对接关系、以及 RL
`old_log_prob` contract 是否能保留。

P1/P2 已经按这份 note 实现并验证：
- 代码：`vrl/math/diffusion/speculative.py`
- 测试：`tests/math/test_speculative_diffusion.py`（CPU，10 passed）

---

## 0. 一句话结论

对 flow-matching 这种 **高斯转移、协方差与模型输出无关** 的 diffusion，
speculative sampling 退化成 **reflection maximal coupling**：draft 先提一个候选，
target 要么接受、要么把它“镜像反射”一下，输出 **精确等于 target 转移分布**，
不需要 residual rejection sampling，也不会吐出任何未校正的 draft 样本。

更关键的（论文没回答、但 RL 需要的）：因为每个吐出的样本都是 target 转移的精确
抽样，它的 behavior log-prob 就是 **target 转移的高斯对数密度**，可以用
`sde_step_with_logprob(target_output, prev_sample=Y)` 原样复算。所以
`old_log_prob` contract 是干净的（见 §6）。

实测（toy Gaussian Markov chain，window=4，draft 贴近 target）：在分布统计检验
通不过拒绝的前提下，**省掉 ~72% 的串行 target 评估**。

---

## 1. 先回顾 LLM speculative decoding（离散）

target 自回归分布 `p`，cheap draft `q`：

```text
1. draft q 连续生成 K 个 candidate token
2. target p 一次并行 forward，给出这 K 个位置的分布
3. 逐个 token：以 min(1, p/q) 接受
4. 第一次 reject：从 residual 分布 ∝ max(0, p-q) 重采一个 token，停止
最终 token 分布 == target p
```

省的是什么：target 的 **并行一次 forward** 覆盖了 K 个候选位置，期望一轮能确定
多个 token。FLOP 没省（draft+target 都跑），省的是 **target 的串行步数 / 墙钟延迟**。

residual `∝ max(0, p-q)` 在离散词表上可以直接归一化采样。**连续空间**没法这么做——
这正是 De Bortoli 这篇要解决的事。

---

## 2. 连续版本：reflection maximal coupling

把 diffusion / flow 看成连续 Markov chain `x_T → ... → x_0`，每步 target 转移
`p_t(x_{t-1}|x_t)`、draft 转移 `q_t`。论文的核心 observation：**对很多 sampler，
单步转移是高斯，且两个策略的高斯只差均值、协方差相同**（协方差只由噪声 schedule
决定）。在这个“同协方差高斯对”的设定下，最优耦合不是 residual rejection，而是
**reflection maximal coupling**（论文 Algorithm 4 / Proposition 3.1）。

记 draft 高斯 `p = N(m_p, σ²I)`、target 高斯 `q = N(m_q, σ²I)`，draft 抽样
`Ỹ ~ p`。令：

```text
Z = (Ỹ - m_p) / σ           # Ỹ 在 draft 下的标准化噪声 ~ N(0, I)
Δ = (m_p - m_q) / σ          # 两均值差（标准化）
```

**accept / reflect 规则：**

```text
accept_prob = min(1, N(Z+Δ; 0,I) / N(Z; 0,I))
            = min(1, exp(-(Z·Δ) - ½‖Δ‖²))        # 全维内积/范数，真·联合密度
U ~ Uniform(0,1)
若 U ≤ accept_prob:   Y = Ỹ                         # 接受，原样保留 draft 候选
否则:                e = Δ/‖Δ‖
                     Y = m_q + σ (I - 2 e eᵀ) Z     # 沿 e 反射，得到精确 target 抽样
```

**Proposition 3.1**：上式输出 `Y ~ q` 精确成立，并且最大化 `P(Y = Ỹ)`，
拒绝概率正好是总变差距离

```text
P(Y ≠ Ỹ) = ‖p - q‖_TV = 2Φ(‖m_p - m_q‖ / (2σ)) - 1
```

直觉：接受区对应两个高斯密度的重叠部分（maximal coupling 的“公共质量”）；
拒绝时不是丢弃重采，而是用一次 **确定性反射** 把 draft 噪声映射到 target 高斯上
缺的那部分质量。反射是 measure-preserving 的，所以拼起来恰好是 `q`。这就是为什么
连续空间不需要“从 residual 分布采样”——反射本身就给出 residual 那块质量的精确样本。

> 注意两种“范数”不要混：接受判据里的 `‖·‖`、`Z·Δ` 是 **对所有 latent 维求和** 的
> 真·联合高斯密度；而 repo 里 `sde_step_with_logprob` 返回的 `log_prob` 是 **逐元素
> 均值**（只为 ratio 记账方便）。耦合的正确性必须用 sum 版（见 `reflection_maximal_coupling`
> 内部），trainer 消费的 `old_log_prob` 用 mean 版（见 §6）。

---

## 3. 为什么 target NFE 会减少（论文 Algorithm 3）

单步耦合本身不省算（draft、target 都要算均值）。省算来自 **block 化 + 并行验证**：

```text
1. draft 从当前 accepted 状态 x 出发，串行往前提 L 个候选 ỹ_1..ỹ_L
   （draft 便宜：见 §4，frozen-target draft 连 target forward 都不用重算）
2. target 对这 L 个输入状态 (x, ỹ_1, ..., ỹ_{L-1}) **一次 batched forward** 算出 L 个 m_q
   —— 这是 1 个 target round（而不是 L 次串行 forward）
3. 逐 k 做 reflection coupling：只要 accept，accepted 前缀 == draft 前缀，
   下一步的 target 条件状态就是 ỹ_k，自洽；
   第一次 reject：吐出反射后的 Y_k（精确 target 抽样），停止本 block，从 Y_k 重启
```

每个 block 花 **1 次 batched target round**，期望推进 `> 1` 个 step。所以

```text
target_rounds = total_steps / E[每 block 推进步数]
NFE_saved   = 1 - target_rounds / total_steps
```

这是 **串行 target 评估次数 / 墙钟** 的节省，不是 FLOP 节省（和 LLM spec decoding
同构）。F2 的答案：target NFE 能减少，**前提是** L 个候选状态能 batch 成一次 target
forward——这正是 VRL diffusion executor 已有的能力（一次 forward 吃一个 batch 的 latent）。

每 block 推进步数随逐步接受率 `a` 增长很快（`a` 越接近 1，越接近满 L）。toy 实测：
`a ≈ 0.94`（draft 贴近 target）时每 block 推进 ≈ 3.7，省 ~72%。

---

## 4. Draft 选择（哪些不用训练）

论文给了 **out-of-the-box、不训练 draft model** 的策略，最适合先做 proof：

- **Frozen target draft**（论文 Eq. 7）：在 window 起点 `y_n` 算一次 target 的
  drift/score `b^q_{t_n}(y_n)`，window 内后续 draft step 复用这个冻结的 drift：
  `m^p_k = y_k + γ · b^q_{t_n}(y_n)`。也就是 **window 内只算一次 target forward**，
  靠它廉价地往前滚 L 步当 draft，再批量验证。零额外模型、零训练。

VRL 自己的额外 draft（sprint §6，论文之外的 RL 系统面）：

- **previous-policy draft**（§6.1）：draft = 上一个 rollout policy 权重，target =
  当前 policy。RL 天然有“上一步策略”，policy update 小 ⇒ `‖m_p-m_q‖/σ` 小 ⇒ 接受率高。
  toy 里 `b_draft = b_target + 小量` 就是这个 regime 的代理，实测省 ~72%。
- cheap/low-order solver draft、low-precision draft（§6.2/6.3）：同样靠 target 校正
  保证 exact，先不做 distilled 小模型（§6.4，成本最高）。

---

## 5. 和 VRL `sde_step_with_logprob` 的契合度

**结论：天然契合，因为协方差与模型输出无关。** 看
`vrl/math/diffusion/flow_matching.py` 的 flow_grpo 分支：

```python
std_dev_t        = sigma_min + (sigma_max - sigma_min) * sigma      # 只看 schedule
prev_sample_mean = sample*(1 + std²/(2σ)*dt) + model_output*(...)*dt # 唯一含 model_output 的量
prev_sample      = prev_sample_mean + std_dev_t*sqrt(-dt) * noise    # 噪声尺度 = std_dev_t*sqrt(-dt)
```

- 单步转移就是 `N(prev_sample_mean, noise_scale²I)`，`noise_scale = std_dev_t·sqrt(-dt)`。
- `noise_scale` **只依赖 sigma/schedule/step**，完全不含 `model_output`。
- 所以 draft（draft 的 `model_output`）和 target（target 的 `model_output`）产生的
  两个高斯 **协方差严格相同、只差均值** —— 正好是 §2 reflection coupling 的设定，
  无需任何近似。

适配的 sampler/scheduler 条件：

- ✅ flow-matching SDE（`sde_type="flow_grpo"`、`"cps"`）：高斯、同协方差，直接适用。
  EDM-domain scheduler（Cosmos Predict2，sigma_max>1）已被 `sde_step_with_logprob`
  内部转成 flow [0,1] 域，耦合在 flow 域做即可。
- ✅ 任意把单步写成“均值 + schedule 决定的高斯噪声”的 ancestral/SDE sampler。
- ⚠️ `deterministic=True` / SDE window 外的 ODE 步：噪声为 0，没有耦合空间（也不需要，
  本来就是确定性映射，draft==target 才有意义）。
- ⚠️ 协方差依赖 `model_output` 的 sampler（本 repo 目前没有）：要回到更一般的耦合，
  不在本 sprint 范围。

实现见 `speculative_sde_step()`：分别用 draft/target 的 `model_output` 调
`sde_step_with_logprob` 拿两套均值 + 共享 `noise_scale`，抽 draft 候选，做反射耦合。
当 `draft_output is target_output` 时 `Δ=0` ⇒ 永远接受 ⇒ 退化成普通 target 采样
（P2 的 “speculative disabled == baseline” 已用测试钉住）。

---

## 6. RL `old_log_prob` contract（P0 的核心问题，F3）

sprint 把这条标成最危险：**“final sample exact” 不自动等于 “每个 trainable
transition 的 behavior logprob 已经清楚”**。这份 note 的结论是 **对高斯/flow 情形，
contract 干净且可保留**：

1. 每个吐出的 `Y`——无论 accept 还是 reflect——都是 **target 转移 `q_t` 的精确抽样**
   （Proposition 3.1）。
2. 因此它的 behavior log-prob 就是 `q_t` 在 `Y` 处的高斯对数密度：
   `log N(Y; m_q, noise_scale²I)`。
3. 这个量可以用 **target 的 `model_output` 原样复算**：
   `sde_step_with_logprob(scheduler, target_output, t, x_t, prev_sample=Y).log_prob`。
   ——和 replay evaluator (`vrl/rollouts/evaluators/diffusion/sde_logprob.py`) 用的
   **完全同一个函数、同一套 mean 约定**，所以 rollout 时的 `old_log_prob` 与训练时
   replay 的 logprob 在数值上一致，replay ratio≈1 不变。

关键区别（和 LLM 的 residual 不同）：连续反射耦合 **没有“来自 residual 分布”的样本**。
LLM 里 reject 后从 `∝max(0,p-q)` 采的 token，其 behavior prob 不是单纯的 `p`；而这里
reject 后的反射样本仍然是 `q` 的精确抽样，密度就是 `q(Y)` 闭式可算。所以 sprint §7
担心的“residual sample 进不了 policy-gradient”在高斯情形 **不会发生**——每个 step 都能
给 trainer 一个明确的 target `old_log_prob`。

落到 trajectory contract：

```text
old_log_prob          = speculative step 返回的 target 转移 logprob（= 上面第 3 条）
behavior_policy_version= target rollout policy version（不是 draft）
accepted_mask / accept_prob / draft_log_prob / target_nfe / draft_nfe
                       = 只进 context / metrics，给 acceptance 调度和 debug，
                         绝不进 policy-gradient 数学
```

`SpeculativeStepResult`（`vrl/math/diffusion/speculative.py`）已经按这个分工返回字段。

---

## 7. 我用自己的话写一遍算法（P0 验收）

```text
输入: 当前状态 x, 窗口 L, draft 均值函数 m_p(·), target 均值函数 m_q(·),
      噪声尺度 σ（只看 schedule）
重复直到走完所有 denoise step:
  # draft 阶段（便宜）
  states = [x]; props = []
  cur = x
  for k in 1..L:
      ỹ = m_p(cur) + σ·ξ_k          # ξ_k ~ N(0,I)
      props.append(ỹ); states.append(ỹ); cur = ỹ
  # verify 阶段：一次 batched target forward 算出所有 m_q(states[0..L-1])  ← 1 个 NFE round
  for k in 0..L-1:
      Z = (props[k] - m_p(states[k]))/σ
      Δ = (m_p(states[k]) - m_q(states[k]))/σ
      若 U ≤ min(1, exp(-(Z·Δ) - ½‖Δ‖²)):
          x = props[k]               # accept
      否则:
          e = Δ/‖Δ‖
          x = m_q(states[k]) + σ(Z - 2(e·Z)e)   # reflect → 精确 target 抽样
          break                       # 停止本 block，从 x 重启
  记录每个吐出的 x 及其 target logprob log N(x; m_q, σ²I)
```

- transition density 需要的量：每步只要 **两个均值 `m_p, m_q` 和共享 `σ`**（高斯）。
- target NFE 为什么少：L 个候选状态 batch 成一次 target forward，期望一轮推进 >1 步。
- accept/reject 后如何记 target logprob：吐出的 `x` 恒为 target 抽样，logprob =
  `log N(x; m_q, σ²I)`，用 target output 复算。
- continuous residual 怎么处理：**不需要 residual 采样**，反射代替。
- 哪些 sampler 满足：单步高斯且协方差不依赖 model_output 的 SDE/ancestral sampler
  （本 repo flow_matching 全系满足）。

---

## 8. Failure modes（已用测试佐证）

`tests/math/test_speculative_diffusion.py::test_speculative_savings_track_draft_quality`
扫了 draft 质量 vs 省算：

```text
|Δ|/σ ≈ 0.057  → 73.6% saved   # previous-policy / 小 policy update
|Δ|/σ ≈ 0.141  → 72.0% saved
|Δ|/σ ≈ 0.849  → 57.8% saved   # 反射耦合相当鲁棒，偏一点也很能省
|Δ|/σ ≈ 4.24   →  3.3% saved   # F1：draft 太差 → 几乎全 reject → 退化成 baseline
```

- **F1 接受率太低**：`|Δ|/σ` 大时省算崩塌，必须 `min_acceptance_rate` 自动 fallback
  到普通 target rollout（否则净增 overhead）。
- **F2 target NFE 没省**：只有当 L 候选能 batch 成一次 target forward 才省（§3）；
  VRL executor 满足。
- **F3 logprob 不清楚**：高斯/flow 情形不发生（§6），每步都有干净 target old_log_prob。
- **F4 early training draft 偏**：previous-policy draft 在训练早期可能 `|Δ|/σ` 大 →
  需要 acceptance-aware 调度（接受率低就关）。
- **F5 短 schedule 收益不足**：本 repo 很多 config 才 10–35 步，绝对收益要实测（P3）。

---

## 9. 对后续 phase 的指向

- **P2（已做）**：`speculative_sde_step` 单步原型，disabled==baseline 已钉住。
- **P3（real model micro-probe）❌ 实测 NEGATIVE（2026-06-24，SD3.5-medium，28 步/1024²）**：
  在真实 latent（D=262144，sqrt(D)=512）上实测（探针已删除，数字保留于此）：
  frozen-target draft 的 `‖Δ‖/σ ≈ 18`（offset 1）→ acceptance≈0、省算 0%。previous-policy
  draft 的接受率悬崖在相对权重更新 1e-4~1e-3 之间——只有 policy 几乎没动才接受。
  **toy（D=8）严重高估了。** 根因：接受率 `a=2Φ(-‖Δ‖/2)` 用全维范数，逐元素 drift × sqrt(D)
  把 ‖Δ‖ 撑到 ≫1。这是 F2 的实证答案：whole-latent 整步接受这个省算机制被高维击穿；
  子向量耦合能保 exactness 但不省 NFE。P4/P5 因此 gate 掉。详见 sprint §9 P3。
- **P4（RL gate）**：把 §6 的 `old_log_prob = target logprob` 接进 trajectory builder
  的 `context`/replay；small GRPO run 对比 baseline reward 曲线 + rollout mismatch guard。
- **P5（video）**：target verifier batching / latent memory / Ray payload，系统 moat。

---

## 10. 参考

- De Bortoli, Galashov, et al. *Accelerated Diffusion Models via Speculative Sampling*,
  ICML/PMLR v267, 2025 — https://arxiv.org/abs/2501.05370 /
  https://proceedings.mlr.press/v267/de-bortoli25a.html
- Leviathan et al. *Fast Inference from Transformers via Speculative Decoding* —
  https://arxiv.org/abs/2211.17192
- SpecInfer（tree-based verification）— https://arxiv.org/abs/2305.09781
- 代码对接点：`vrl/math/diffusion/flow_matching.py`（`sde_step_with_logprob`）、
  `vrl/math/diffusion/speculative.py`（本 sprint 新增）、
  `vrl/generation/diffusion/executor.py`（denoise loop，P3 接入点）、
  `vrl/rollouts/evaluators/diffusion/sde_logprob.py`（replay logprob，contract 对齐）。
```
