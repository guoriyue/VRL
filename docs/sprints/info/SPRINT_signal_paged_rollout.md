# SPRINT: Signal-Paged Rollout / diffusion shared-prefix tree rollout

状态：**negative measurement archive（2026-06-26）**。KIND：**info**；P0 只共享前 4 步时，
latent 多样性已降到 44% retention，在约 1.5× forward 减少点仅剩 31%–36%，明确低于
≥70% 的验收门。本文自己的关闭条件已经触发，P1–P3 全部取消；这里保存测量、推导和历史
提案，不是等待事件后重启的 parked 工作。

> 原状态为 planned / proof-gated（2026-06-23）。实验不是把 vLLM/SGLang 的 AR paged KV
> 搬进 diffusion，而是让同 prompt group 共享 denoise 前缀。实测已经回答“是否值得推进”：
> 扩散早期步锁住全局结构，增加 SDE 噪声仍无法保住足够多样性，因此不进入 executor 或
> block-native 实现。

## 0. 一句话

PagedAttention 的杀伤力来自一个资源抽象：把 KV cache 变成可分页、可共享、可调度的 block。diffusion/video RL 里对应的候选能力不是 KV page，而是：

```text
denoise trajectory -> trajectory blocks -> per-sample block table
```

也就是 **Signal-Paged Rollout**：

```text
同一个 prompt group
  shared prefix block: denoise step 0..k，只算一次
    branch block sample 0: denoise step k..T
    branch block sample 1: denoise step k..T
    branch block sample 2: denoise step k..T
```

这能把一个 group 的 forward 数从：

```text
G * T
```

降成：

```text
k + G * (T - k)
```

但它不是免费午餐：共享越多，sample 越相关，group 内 reward variance 可能越低；GRPO 的信号正来自这个 variance。

## 0.5 外部研究背书（2026-06-26，deep-research 已验证）

跑了一轮无损加速 diffusion RL 的 deep-research（23 源 → 22 确认 claim，全文 [[SPRINT_lossless_diffusion_rl_research]]）。对本 sprint 两条直接结论：

1. **shared-prefix 是报告认定的"唯一真正待证的无损 group 级杠杆"。** 报告综合性结论：没有任何无损的"diffusion 连续批处理/分页"能省 per-step 计算（无 KV cache）；而同 prompt 的 GRPO group 共享早期去噪前缀是少数**可能既省 forward 又不改输出分布**的角度——前提是保住 per-sample `old_log_prob`。这正是本 sprint 的命题，被列为报告开放问题 #1。
2. **正确性铁律有了外部权威背书**：verl 逐字"old_log_prob 必须用 rollout 参数和 tokens 算，不能用 trainer 算"（`bypass_mode` 默认 True）。对本 sprint = prefix step 的 `old_log_prob` 要么按 `trainable_mask=0` 排除、要么必须能被训练 replay 精确重放；**且 off-policy 复用里只有 version-rejection 是 exactly lossless，IS/TIS 是"近似但有修正"**——所以 shared-prefix 必须走"前缀对 centered gradient 零贡献 → mask=0"这条 exact 路，不能依赖 IS 兜。

> 这把本 sprint 从"throughput 实验"升级成"报告点名的无损候选"：proof gate 不变（reward variance 不塌），但它现在是三份性能 sprint 里**唯一同时满足 compute-bound 后仍可能省计算 + 无损**的那个。其余（paged store / stepwise batching）本轮已被实测证伪，feature cache 是近似（拿正确性换）。

## 0.6 P0 实测：latent 多样性悬崖 —— 负结果，关闭（2026-06-26）

P0 曾用现已退役的 `shared_prefix_divergence_probe.py` 跑 latent-diversity
实验（G=6 T=28 768²，repo 真实 `sde_step_with_logprob`）。结论和完整数据保留如下；
该方向已关闭，因此不再保留可执行命令：

```
            noise_level=1.0       noise_level=1.4
k    fwd_saved  retention         retention
0    0%         89%               87%
4    11.9%      44%               61%
14   41.7%      31%               36%   ← 1.5x forward 减少点
24   71.4%      13%               23%
```

**P0 验收门是"≥1.5x forward 减少(≈k14)同时 reward variance retention ≥70%"。实测该点只有 31%(高噪声 36%),门没过;只共享前 4 步(省 12%)多样性就塌到 44%。** 扩散早期步锁全局结构,提高 SDE 噪声有缓解但救不回来。这正是 §7 R1 的风险,实测坐实。

**按本 sprint §9 自己写的关闭条件（“如果 P0/P2 显示 reward variance 被压平就关闭”），
本方向已经关闭。** reward-model 专用版本若将来出现新的正面证据，应另立假设与 proof gate，
不能把本负结果重新解释成未完成事项。完整数据与对标见
[[SPRINT_lossless_diffusion_rl_research]] §0.6 验证 2。

> caveat：latent 多样性是 reward variance 的代理而非本体，且只测了单 prompt / SD3.5 /
> 768²。这个限制保留在测量档案里，但不把已经失败的验收门改写成“待完成”；若新的 reward-model
> 假设值得验证，应另立 Sprint。

## 1. 先修正旧类比

已有文档里拒绝过“把 AR paged KV / prefix cache 引入 diffusion”。这个判断仍然成立，但范围要说准：

- **仍然不做**：AR-style paged KV、automatic prefix caching、chunked prefill、continuous batching。diffusion denoise 没有 KV decode 序列，也没有跨 step 可直接复用的 attention KV。
- **本 sprint 新增讨论**：同 prompt、同 policy、同 conditioning 下，**人为让 group 内样本共享一段 latent denoise 前缀**，然后 copy-on-write 分叉。这是 tree rollout / trajectory block 设计，不是 serving engine 的 KV cache。

因此本 sprint 不推翻 `SPRINT_diffusion_rollout_system` 的结论；它把“AR cache 不适用”细化成“KV cache 不适用，但 RL group 内的 trajectory-prefix sharing 值得 proof”。

## 2. 为什么可以共享前缀

这部分不是凭空发明的。**TreeGRPO** 已经把 diffusion / flow generator 的 denoise process 组织成 search tree，从共享 initial noise 开始，在中间 step 分叉，并复用 common prefixes。本 sprint 的 shared-prefix proof 直接受 TreeGRPO 启发；我们额外强调的是如何落到 VRL 的 `TrajectoryBatch` / dense fallback / trainable mask contract。

### 2.1 必须满足的 exact reuse 条件

diffusion 的 transformer forward 只有在这些输入相同的时候才能无损复用：

```text
policy weights / policy_version
prompt_embeds and negative_prompt_embeds
conditioning（image/video/control/ref frame）
latent state x_t
timestep / scheduler state
noise/RNG decision（如果当前 step 是 stochastic）
sampling config（CFG、SDE type、noise_level、window）
```

所以“同一个 prompt”本身不够。如果每个 sample 从不同 initial noise 开始，第一步 latent 就不同，后面所有 forward 都不能复用。要复用，必须主动制造相同状态：

```text
step 0..k 用同一个 latent/noise path
step k 之后再给每个 branch 独立噪声 / perturbation / stochastic path
```

### 2.2 对 GRPO 为什么不是明显 biased

GRPO / group-relative 方法会把同一 prompt 的 group reward 做中心化。设 group 内 advantage 是 `A_i`，则：

```text
sum_i A_i = 0
```

如果前缀 step `t < k` 被所有 sample 完全共享，那么这些 step 的 state/action/logprob gradient 对每个 sample 都一样：

```text
g_prefix = grad log pi(a_t | s_t)
```

前缀对 policy gradient 的贡献变成：

```text
sum_i A_i * g_prefix = g_prefix * sum_i A_i = 0
```

结论：

- 共享前缀 step 对 centered GRPO gradient 本来就抵消。
- 所以第一版可以把 prefix 标成 `trainable_mask=0`，不在这些 step 上训练。
- 真正产生学习信号的是分叉后的 suffix，因为那里 sample 开始不同，reward 才有 variance。

这不是说共享前缀永远安全。它只说明 **共享前缀不会直接丢掉一段有效的 centered-policy-gradient**；真正的风险是探索变少，导致最终 reward variance 下降。

### 2.3 对非 GRPO 算法不自动成立

这个零贡献推导依赖 `sum_i A_i = 0`。如果算法不是 group-centered advantage，或者使用非中心化 reward、pairwise preference、DiffusionNFT/DPO 等目标，prefix 是否能 mask 要重新定义。第一版只对 Flow-GRPO / GRPO-like objective 开门。

## 3. 这是不是只对 RL 有用

不是只对 RL 有用，但 **RL 是最自然、最有价值、也最容易定义正确性的场景**。

### 3.1 RL rollout：强适用

RL 里同一个 prompt 本来就要采 `G` 个 samples 来估计 group-relative signal。shared prefix 可以直接吃这个结构：

```text
same prompt + same policy + fixed group -> tree rollout
```

收益目标也明确：

```text
少算 forward，但保留足够 group reward variance
```

正确性边界也明确：

```text
prefix steps: trainable_mask=0
suffix steps: trainable_mask=1
behavior_policy_version / old_log_prob: 只记录 suffix 或按 mask 消费
```

### 3.2 普通 inference serving：有限适用，不能透明替换

对普通图片/视频 serving，用户通常以为 `n` 个 samples 是独立采样。shared prefix 会让多个输出共享早期构图/运动主干，结果更相关。这可能是 feature，也可能是 bug。

适用场景：

- “给我同一构图/同一运动主干的多个细节变化”
- interactive editing / variation generation
- 需要稳定布局、只探索局部细节的 batch generation
- tree sampling UI：用户先看粗结构，再从某个分叉继续生成多种细节

不适用场景：

- 用户期望完全独立、多样性最大的 `n` 个 samples
- benchmark 要求与独立 full rollout 分布一致
- API 没有暴露 correlation / branch semantics，却内部偷偷共享前缀

所以 serving 可以用，但必须作为一个显式 sampling mode，例如：

```yaml
sampling:
  branch_mode: shared_prefix
  shared_prefix_steps: 12
  diversity: layout_locked
```

不能把它作为所有 inference 的透明优化。

### 3.3 vLLM-Omni / SGLang backend 的位置

外部 engine 可以成为 block executor，但不是这个能力的 source of truth：

```text
execute_block(prompt_embeds, latent_start, t_start, t_end, rng)
  -> latent_end, log_probs, timesteps, optional media
```

Signal-Paged Rollout 的核心资产是 block table / trainable mask / materialization contract；backend 可以是当前 diffusers executor，也可以是 vLLM-Omni custom pipeline。

## 4. 当前系统落点

现状是 dense trajectory：

```text
vrl/generation/diffusion/executor.py
  run_denoise_steps()
    preallocate observations/actions/log_probs/timesteps/kl
    every step writes dense [B, T, ...]

vrl/generation/diffusion/gather.py
  cat chunks -> build_diffusion_trajectory(...)

vrl/trajectory/builders.py
  build_diffusion_trajectory(...)
    observations: role=observation
    actions: role=action
    old_log_prob: role=old_log_prob
    mask: role=mask
```

MVP 不要先改成全 block-native。先做 **execution tree + dense fallback**：

```text
run prefix once with batch=1
repeat/copy latent_end to branch batch G
run suffix with batch=G
materialize dense observations/actions/log_probs/timesteps
set trainable mask=0 for prefix, 1 for suffix
```

这样 algorithm 和 evaluator 先不大改，只需要能读 mask。

## 5. 设计草案

### 5.1 Rollout plan

新增一个计划对象，不先进 schema 公共面，先作为 generation 内部实验结构：

```text
SharedPrefixRolloutPlan
  prompt_index: int
  group_size: int
  num_steps: int
  prefix_steps: int
  branch_steps: range(prefix_steps, num_steps)
  seed_base: int | None
  policy_version: int | None
  trainable_mask: [num_steps] bool
  step_kind: [num_steps] enum(shared_prefix, stochastic_branch)
```

第一版只支持单分叉点 `prefix_steps=k`。后续再考虑多个 branch point / high-entropy window。

### 5.2 Execution

当前路径：

```text
prepare_denoise_state(batch=G)
run_denoise_steps(batch=G, steps=0..T)
```

实验路径：

```text
prepare_denoise_state(batch=1)
run prefix steps 0..k once
snapshot latent_end
expand latent_end to batch=G
run branch steps k..T with independent branch RNG
```

注意：如果后缀没有 stochastic noise / perturbation，branches 会完全一样。shared-prefix 只在后缀有分叉机制时有意义：

```text
SDE suffix noise
independent branch noise
latent perturbation at branch point
or model-specific stochastic sampler
```

### 5.3 Trajectory materialization

第一版仍然生成 dense tensors：

```text
observations[:, t]
actions[:, t]
old_log_prob[:, t]
timesteps[:, t]
mask[:, t]
```

prefix step 的规则：

```text
observations/actions: 可以展开成 G 行相同值（dense fallback）
old_log_prob: 写 0 或真实 log_prob，但 mask=0，algorithm 不消费
mask/trainable_mask: 0
step_kind: shared_prefix
```

suffix step 的规则：

```text
observations/actions/log_probs/timesteps: 正常写
mask/trainable_mask: 1
step_kind: stochastic_branch
```

后续 block-native 才把 prefix 物理只存一份，训练时按 resolver lazy materialize。

## 6. 历史 phase plan（P0 关闭后 P1–P3 已取消）

以下阶段保留用于解释当时的决策门。P0 已经失败，因此这些内容不再拥有执行状态。

### P0 — one-shot proof：数学和分布诊断

目标：不写 engine，只证明什么 `k` 可能有价值。

- 对一个现有 diffusion recipe，采 full independent baseline。
- 离线模拟 prefix sharing 的多样性上界：比较不同 timestep latent 的 pairwise distance / reward variance。
- 输出 `prefix_steps -> reward variance retention` 曲线。

验收：

```text
找到至少一个 k，使 rollout forward 理论减少 >= 1.5x，
同时 reward variance retention >= 70%。
如果找不到，关闭 sprint。
```

### P1 — executor MVP：shared prefix + dense fallback

目标：在当前 `DiffusionChunkExecutorBase` 上加实验路径，不动 algorithm 主流程。

- 支持 `sampling.shared_prefix_steps`（默认 off）。
- 只在 `sample_count > 1` 且同 prompt chunk 生效。
- prefix batch=1，suffix batch=G。
- dense fallback 输出当前 `DiffusionChunkResult`。
- context/counters 记录：

```text
shared_prefix_steps
shared_prefix_forward_saved
shared_prefix_theoretical_speedup
shared_prefix_reward_variance
```

验收：

```text
shared_prefix_steps=0 与 baseline bit/shape parity
shared_prefix_steps>0 不崩、mask 正确、训练能跑至少一个小 epoch
forward count 按 k + G*(T-k) 下降
```

### P2 — RL gate：learning curve A/B

目标：证明不是只省算，而是同等 budget 下学得不差。

对比：

```text
baseline: independent G samples
shared-prefix: same prompt budget, prefix_steps=k
```

指标：

```text
rollout wall-clock
reward mean / reward std / within-group reward variance
adv_zero_rate
rollout-vs-replay mismatch
final eval reward curve
```

验收：

```text
wall-clock >= 1.3x speedup
eval reward curve 不显著退化
adv_zero_rate 不显著上升
```

### P3 — block-native trajectory（只有 P2 通过才做）

目标：把 dense fallback 升级成真正的 trajectory paging。

- `TrajectoryBlock` / block table。
- prefix block ref_count。
- lazy materialization in `TrajectoryResolver`。
- algorithm-declared required fields。

P3 是 killer capability 的正式化；P1/P2 只是证明它值得做。

## 6.5 论文依据：哪些是 paper，哪些是本 repo 的系统化

### Directly paper-backed

- **TreeGRPO**：直接支持“共享 denoising prefix + 分叉生成多条候选轨迹 + 复用 common prefixes”。这是本 sprint shared-prefix tree rollout 的最直接来源。
- **Flow-GRPO**：直接支持 diffusion / flow matching 上的 online GRPO、ODE-to-SDE 转换、denoising reduction。它说明为什么 rollout 是瓶颈，以及为什么只训练部分 denoise steps 是合理优化方向。
- **TeaCache / DeepCache / ToCa**：支持 diffusion inference 里跨 timestep 存在可利用的冗余，以及 feature/model-output caching 能减少 denoise forward。不过它们是 within-sample approximate feature/output reuse，不是本 sprint 的 within-group exact shared-prefix。
- **PagedAttention / vLLM**：支持 block table、ref-count、copy-on-write、prefix sharing 这种资源抽象。它是 P3 “trajectory block table” 的系统类比来源，不是 diffusion denoise 的直接算法来源。

### Our synthesis / not directly claimed by one paper

- **Signal-Paged Rollout** 这个名字和 `TrajectoryBlock` / `TrajectoryResolver` lazy materialization 是 VRL 系统设计，不是 paper 原名。
- **Dense fallback first, block-native later** 是为了贴合当前 VRL contract 的工程落地路径，不来自 paper。
- **prefix `trainable_mask=0` 的 GRPO 推导** 是从 group-centered advantage `sum_i A_i = 0` 推出来的；TreeGRPO 做了树结构和 credit assignment，但 VRL 第一版可以先用 mask 保守落地。
- **serving 不能透明默认启用** 是系统/API 判断：shared-prefix 会改变 samples 间相关性，paper 的训练效率结论不能自动推广成通用 serving 默认行为。

## 7. 风险

### R1. reward variance 塌掉

共享高噪声前缀会锁住全局构图/运动。如果 reward 关注物体数量、空间关系、动作主干，太晚分叉会直接让 group reward 变平。缓解：

```text
prefix_steps 从小到大 sweep
记录 within-group reward variance
adv_zero_rate 作为 fail-fast 指标
```

### R2. sampler 分布改变

shared-prefix 不等价于独立采样。RL 可以把它当成新的 exploration policy，并用 behavior logprob/mask 记录；普通 serving 不能透明替换。

### R3. replay/logprob contract 混乱

prefix step 不训练，就必须用 mask 明确排除。不要让 algorithm 误以为 prefix 的 old_log_prob 是有效 PG 信号。

### R4. 与 TeaCache 混淆

TeaCache 是 within-sample approximate forward skip，会引入 rollout-vs-replay drift；shared-prefix 是 within-group structural execution plan，第一版不依赖近似 skip。两个能力可以并存，但不要在同一个实验里同时打开。

## 8. 非目标

- 不做跨 iteration 旧 trajectory replay buffer。
- 不做 AR paged KV / vLLM continuous batching 的 diffusion 移植。
- 不把 shared-prefix 作为普通 serving 的透明默认优化。
- 不在 P0/P2 通过前改 `TrajectoryBatch` 成 block-native。
- 不同时叠 TeaCache/fp8/shared-prefix；先隔离变量。

## 9. 成功标准

这个 sprint 原本的成功标准不是“代码跑起来”，而是证明新的资源抽象值得继续：

```text
同 prompt group 的 denoise 前缀可以作为 block 共享，
在不毁掉 reward variance 的前提下降低 forward 数，
并且能通过 trainable mask 保持 RL loss contract 清晰。
```

P0 已经显示 reward variance 被压平，因此结论是 shared-prefix 不是本 repo workload 的
killer capability；该方向已关闭，不继续投入 P1–P3。

## 10. 参考代码

- `vrl/generation/diffusion/executor.py`：当前 denoise loop、TeaCache gate、dense replay buffer 写入。
- `vrl/generation/diffusion/gather.py`：当前 chunk cat + dense `GenerationOutput`。
- `vrl/trajectory/builders.py`：`build_diffusion_trajectory(...)` 的 dense `observations/actions/old_log_prob/mask` contract。
- `vrl/trajectory/types.py`：`TrajectoryBatch` / `TrajectorySegment` / `TrajectoryTensor` contract。
- `vrl/rollouts/collector/batch_builder.py`：trajectory -> `RolloutBatch` materialization。
- `vrl/generation/diffusion/teacache.py`：已有 approximate forward reuse，作为对照而不是本 sprint 主线。
- `docs/sprints/reading/SPRINT_diffusion_rollout_system.md`：旧的 AR-paged-cache 类比边界。
- `docs/sprints/done/SPRINT_rollout_vllm_migration.md`：TeaCache 结论和 vLLM-Omni engine adoption 边界。

## 11. 论文 / 外部参考

- TreeGRPO: Tree-Advantage GRPO for Online RL Post-Training of Diffusion Models — https://arxiv.org/abs/2512.08153
- TreeGRPO project page — https://treegrpo.github.io/
- Flow-GRPO: Training Flow Matching Models via Online RL — https://arxiv.org/abs/2505.05470
- Flow-GRPO code — https://github.com/yifan123/flow_grpo
- DiffusionNFT: Online Diffusion Reinforcement with Forward Process — https://arxiv.org/abs/2509.16117
- DiffusionNFT project page — https://research.nvidia.com/labs/dir/DiffusionNFT
- Efficient Memory Management for Large Language Model Serving with PagedAttention / vLLM — https://arxiv.org/abs/2309.06180
- vLLM PagedAttention blog — https://vllm.ai/blog/2023-06-20-vllm
- TeaCache: Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model — https://arxiv.org/abs/2411.19108
- TeaCache code — https://github.com/ali-vilab/TeaCache
- DeepCache: Accelerating Diffusion Models for Free — https://arxiv.org/abs/2312.00858
- ToCa: Accelerating Diffusion Transformers with Token-wise Feature Caching — https://arxiv.org/abs/2410.05317
