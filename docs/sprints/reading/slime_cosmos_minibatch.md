# Mini-batch + 跨 rollout/train 的 mini-batch — slime vs cosmos-rl（reading）

类型：cross-system 源码对照（2026-06-17）。file:line 相对各 repo root，我们的相对 `vrl/`。
index：[slime_cosmos_study_index.md](slime_cosmos_study_index.md)。

## 先分清两个概念
- **"跨 rollout 和 train"**:一次 rollout 产出的数据,切成**几次优化步**。
- **"train 内部 mini-batch"**:一次优化步内,梯度怎么累积(micro-batch / 时间步 / μ 重放)。

## slime —— 一次 rollout → N 个独立优化步，无内层重放

- 一次 rollout 产 `R = rollout_batch_size × n_samples_per_prompt` 个样本,trim 到 `global_batch_size`
  的整数倍(`ray/rollout.py:598-605`)。
- **跨 rollout/train**:`global_batch_size = R // num_steps_per_rollout`(`utils/arguments.py:1768-1776`,
  默认 `num_steps_per_rollout=1` → 一次 rollout 一个优化步)。DP split(`rollout.py:754-805`,round-robin 或
  `--balance-data` 的 Karmarkar-Karp)后每 rank 拿 `R // dp_size`。
- `get_data_iterator`(`backends/megatron_utils/data.py:321-324`):每 rank `num_local_gbs =
  global_batch_size // dp_size`,把本地样本切成 `num_steps_per_rollout` 个**不重叠窗口**,每窗口=一个优化步。
- **train 内部**:每窗口再切 micro-batch(`num_local_gbs // micro_batch_size`,或 `--use-dynamic-batch-size`
  的 token 打包 + Karmarkar-Karp 平衡 + DP-MAX all-reduce 让各 rank 同 schedule,`data.py:338-385`);
  梯度累积走 Megatron `forward_backward_func`(`no_sync` 到最后一个 micro)。
- **关键:slime 没有 μ-iteration / PPO 内层重放**——一次 rollout 的数据训练**恰好 `num_steps_per_rollout`
  个独立优化步,不重采样、不重放**。

## cosmos-rl —— 三层嵌套 + μ 重放（NFT 还多一层时间步）

- **跨 rollout/train**(controller 层):`rollouts_per_global_batch = train_batch_per_replica × n_replicas`
  (`controller.py:264-266`);每 replica 从全局 FIFO 轮询收 `train_batch_per_replica` 个 rollout
  (`status.py:1435-1439`)。
- **train 内部**(`policy/config/__init__.py:546-549`,`mini_batch` 默认 2,`batch_size_per_optimize` 可选):
  GRPO 三层(`grpo_trainer.py:940+`):
  1. **`mu_iterations`**:对**冻结 old 引用**的整批重放(randperm 打乱)。
  2. **per-optimize chunk**:每 chunk 一次 `optimizer.step`。
  3. **mini-batch**:每个一次 forward+backward,`loss_scaling = cur_mini / optimize_batch`。
- **DiffusionNFT 变体**(`policy/trainer/diffusers_trainer/nft_trainer.py`,**和我们同算法**):
  - `mu_iterations`(`:374`,randperm `:382`)→ optimize chunk(`:401`)→ mini-batch(`:426`)。
  - mini-batch 内**再 loop `num_train_timesteps`**(`:478`),按 `loss_scaling × 1/num_train_timesteps`
    缩放(`:441` / `:608-609`),`backward()`(`:611`),optimizer step 经 `all_reduce_states`(`:666`),
    或每 N 步(`COSMOS_NFT_STEP_INTERVAL`)。
  - 梯度沿 **sample × timestep** 累积,每个 optimize chunk 一步。

## 对照

| | slime | cosmos-rl |
|---|---|---|
| 跨 rollout/train | `GBS = R // num_steps_per_rollout` | `rollouts_per_global_batch = batch/replica × n_replicas` |
| 一次 rollout 产生几步 | `num_steps_per_rollout`(默认 1) | `mu_iterations × optimize_chunks` |
| 内层重放(μ) | **无** | **有**(`mu_iterations`,对冻结 old) |
| train 内累积维度 | micro-batch(token 打包) | mini_batch ×(NFT 还 × timestep) |
| 优化步定义 | 每窗口一步 | 每 optimize chunk 一步 |

## 4. 诚实的反证据：cosmos NFT 有 mu_iterations

我们前面把 `ppo_epochs>1`(内层重放)撤了,理由是"DiffusionNFT likelihood-free、inner-replay 科学正当性不足"。
但 **cosmos 自己的 `nft_trainer.py:374` 就在用 `mu_iterations`**——对**冻结 old 引用**、randperm 打乱的内层重放,
正是我们撤掉的那种(我们的实现也是对冻结 previous-adapter)。

- inner-replay(`mu_iterations`)≠ async-overlap:cosmos 用前者、不用后者(diffusion 仍 serial colocated)。
- 所以:**inner-replay 对 DiffusionNFT 也许没我当时说的那么不可取**。`nft_trainer.py:374-666` 是可对照的真实
  实现(注意它对"冻结 old"重放、randperm、按 `1/num_train_timesteps` 缩放)。若以后再试 inner-replay,照它。

## 对我们
- **我们 streaming 已对上 NFT 范式**:`gradient_accumulation_steps = microbatches × timesteps`
  (`vrl/trainers/online/trainer.py` 流式累积)= cosmos NFT 的 per-optimize ×(mini_batch × timestep)。
- 多卡时再 mirror:cosmos `mu_iterations`(若决定要内层重放)+ slime DP-MAX micro-batch 数对齐
  (`data.py:353`,让各 rank 共享 pipeline schedule)。
