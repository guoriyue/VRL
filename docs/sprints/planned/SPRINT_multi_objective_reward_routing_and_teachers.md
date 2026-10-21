# SPRINT：多目标的两个新实验——按 prompt 路由 reward、多 teacher on-policy 蒸馏

状态：**planned（2026-10-20），实验优先**。两个实验都通过后才考虑合入库结构。

**两个实验的代码都没有入库**：试点都是零结果（§4）。实验 A 的路由（`metadata_filter`、`metadata.source_manifest` 标签、路由 reward preset、`dataset/pickscore_safety_mixture` 和两组实验 preset）与实验 B 的蒸馏脚本按用户决定撤掉。§4 保留为试点记录，其中引用的代码和 preset 已不存在。`--joint-min` 保留。

来源：PyTorch Conference NA 2026 Day 1：043 Nemotron 3 技术报告 §3.2、海报 P83 MOPD、031 联合指标、091/109 gate 语义。

## 0. 背景与结论先行

固定权重的 reward 组合（weighted_sum / normalized_sum）推不开质量和安全之间的前沿。离线跑过的 Lagrangian 分支也没有赢过加权标量化（研究笔记见 commit `869c79ab9`，结论是"do not retry unchanged"）。另外，MiMo 式的 hard floor 已经实现过又撤回了，不再提。

会上有两个此前没试过的方向：

1. **按 prompt 路由 reward（043）**：每个 prompt 只用它所属环境的 reward，同时混合多个环境一起训练。
   - Nemotron 3 的实测：只训一个环境会让其他 benchmark 严重退化；同时在 21 个环境上训，则能稳定提升。
   - 这和 VRL 的做法不同：VRL 对同一个样本加权合成所有 reward。
2. **多 teacher on-policy 蒸馏（P83 MOPD）**：
   - 先为每个目标单独训一个 specialist，作为 teacher。
   - student 在自己的轨迹上，每个去噪步都对 teacher 的预测做 dense 蒸馏。
   - 用蒸馏代替 reward 合成，可能绕开"标量化推不开前沿"的问题。

## 1. 当前代码现实（HEAD a914644c6 核对）

- **combiner**：只有两种，`weighted_sum_raw` 和 `normalized_sum`（`vrl/algorithms/advantages.py:164`），默认 `normalized_sum`；只有 `v_grpo` 覆盖成 `weighted_sum_raw`。`normalized_sum` 下，每个分量先在 prompt 组内标准化，再加权求和（`vrl/algorithms/base.py:125-171`）。
- **推论**：如果一个分量在某个组内是常数，它在这个组里的 advantage 贡献就是 0。所以"在不适用的 prompt 上返回常数"的 reward，本身就等价于按 prompt 路由，**不需要改 combiner**。
- **prompt 元数据已经能传到 reward**：
  - 链路：JSONL 里未知的 key 进入 `PromptExample.metadata`（`vrl/trainers/data/prompts.py:174-209`）→ `RewardSample.metadata`（`vrl/rollouts/collector/batch_builder.py:60-64`）。
  - `wd_tagger` 和 `qwen3vl_preference` 已经在读元数据；`nsfw_safety` 不读。
  - `load_prompt_mixture`（`prompts.py:212-246`）能混合多个数据集，但不会写入来源标签。
  - `manifests/danbooru/safety/*.jsonl` 已经带 `metadata.category`。
- **teacher**：目前没有任何 teacher 或蒸馏代码。可以复用的有三处：
  - `replay_forward_with_latents`（`vrl/models/steps/denoise/base.py:141`）：在 student 的 batch 上做前向，conditioning 已经编码好。
  - NFT 的 KL 项（`vrl/algorithms/diffusion_nft.py:63-75,104`）：对冻结预测做 dense MSE，是最接近的模板。
  - `TrainableWeightsSnapshot`（`vrl/models/weight_snapshot.py:22-60`）：给同架构 teacher 做权重原地交换。
- **anima 已删除**（commit `ac53ecb5a`）。NSFW/质量实验现在没有 preset，只剩 `reward/nsfw_safety.yaml`、`reward/wd_tagger.yaml` 和 `dataset/anime_safety_*.yaml` 这些组件。实验要在现存家族（SANA、SD3.5）上重建。
- **eval**：`vrl/scripts/eval/score_report.py` 只能按单个分量出报告，没有"安全 ∧ 质量 ≥ τ"这类联合指标。这一点和 `SPRINT_industry_post_training_gap_backlog.md` K 项、`SPRINT_production_reward_and_rl_data.md` G2 重叠；**实现时合并成一项，不重复**。

## 2. 阶段

### P0. 联合指标（两个实验共同的判据，做在 K/G2 项里）

- `score_report.py` 加一列：在同一批 eval 样本上，"所有约束分量过阈值，且质量分量 ≥ τ"的样本比例，附 CI。阈值来自 CLI，不进 schema。
- 这个联合比例作为选 checkpoint 的主指标之一（031："useful completion"）。

### P1. 实验 A：按 prompt 路由 reward（不改 combiner）

1. 在数据侧给 prompt 打上环境标签：用 `load_prompt_mixture` 的来源信息，或 manifest 里已有的 `metadata.category`。
2. 在实验 preset 中，把每个 reward 包装成"只在本环境的 prompt 上打分，其余返回常数"。优先做成 reward 的一个 metadata 过滤参数，在 reward 注册处解析，一个参数就够。不新建 combiner。
3. 对照组：在同一个家族（SANA 或 SD3.5）、同一个 prompt 混合、同一个算力预算下，比较"全部 reward 加权"和"按 prompt 路由"。
4. **判据**：在 P0 联合指标上，路由组的前沿是否外推，即同等安全下质量更高，或反之。给出 CI；差异小于 2σ 不算。

### P2. 实验 B：两个 teacher 的 on-policy 蒸馏

1. 用现有 GRPO 分别训两个 LoRA specialist：一个只用质量 reward，一个只用安全或约束 reward。各自取最好的 checkpoint。
2. student 从 base 出发，生成自己的 rollout 轨迹。每个 SDE 步上，对两个 teacher 的 `replay_forward_with_latents` 预测做加权 MSE，形式同 NFT 的 KL 项。
   - teacher 用同架构权重交换实现：LoRA teacher 就是切换 adapter，peft 支持多个 adapter。
   - 这一步写成一次性实验脚本或实验分支，**不进** `vrl/algorithms/`。
3. **判据**：与实验 A、与单 reward GRPO，在同一张联合指标表上比较。只有两个都外推前沿，才立项把蒸馏目标做成正式 algorithm。

### P3. 采纳

- 实验 A 胜出：把 metadata 过滤参数正式化，并补测试。
- 实验 B 胜出：另立 sprint 设计 teacher 的持有方式和显存成本。
- 都没有胜出：把结论写进本文件，然后删除本 sprint。

## 3. 不做

- 不加 Lagrangian、hard floor 或 gating 这类 combiner：都已经试过或撤回。
- 零方差 prompt 过滤和 pass-rate 课程归现有计划：`SPRINT_production_reward_and_rl_data.md` 的 G4/G5、`SPRINT_ngu_adaptive_sampling.md`、`SPRINT_rollout_admission_and_failure_attribution.md` §2。本 sprint 不重复。会上可补充的输入是 065 的 PFSP `p(1−p)` 加权，以及 043 的"丢掉 SFT 总是答对的 prompt"，已写进对应计划的补充段落。

## 4. 实施记录（2026-10-20）

### 代码
- **P0 联合指标**：`feat(score_report): add the joint pass-rate score for constrained objectives`。用法是 `--joint-min COMPONENT=MIN`，每个约束写一次、质量也写一次。报告里会多一个 `joint` 分数：它走现有的按 prompt 聚合的 bootstrap，所以绝对均值就是通过比例，带 CI，同时有相对 base 的配对差。阈值只来自 CLI，不进 schema。

### 实验（2026-10-21，单卡 RTX 5090，试点规模）
设置：
- SD3.5 Medium LoRA GRPO，每组 30 次 update，每次 4 个 prompt × 8 个样本，512px，10 步，CFG 4.5，per-prompt std。
- 质量和安全的 specialist 用同样的设置，只训单个 reward 和单个数据集。
- 蒸馏：student 从 base 出发，两个 teacher 等权，80 轮，每轮 2 个 prompt × 2 个样本，每条轨迹训 3 步。80 轮里，对质量 teacher 的 MSE 从 0.029 降到 0.016，对安全 teacher 的从 0.016 降到 0.013。
- 评估：PickScore test 和 danbooru `eval_baseline` 各取 40 个 prompt，每个 2 个样本；所有组用同一组 prompt 和 seed；只读入各组的 adapter，用 `rescore_media` 打分。
- 联合指标：`nsfw_safety ≥ 0` 且 `pickscore ≥ τ`，τ 取 base 的 PickScore 中位数 0.7919；bootstrap 按 prompt 做。

全部 80 个 prompt 上相对 base 的配对差，括号内是 95% CI，"明确"表示 CI 不含 0：

| 组 | PickScore | nsfw_safety | 联合通过率（base 0.381） |
|---|---|---|---|
| 加权 | +0.0058 [+0.0019, +0.0100] 明确↑ | −0.046 [−0.090, −0.006] 明确↓ | +0.019 [−0.031, +0.069] |
| 路由 | −0.0079 [−0.0119, −0.0043] 明确↓ | −0.007 [−0.038, +0.025] | −0.050 [−0.106, +0.006] |
| 质量 specialist | −0.0028 [−0.0069, +0.0011] | −0.023 [−0.068, +0.015] | −0.019 [−0.075, +0.044] |
| 安全 specialist | −0.0023 [−0.0058, +0.0011] | −0.003 [−0.032, +0.025] | −0.019 [−0.075, +0.038] |
| 两 teacher 蒸馏 | −0.0016 [−0.0050, +0.0016] | −0.024 [−0.067, +0.018] | 0.000 [−0.050, +0.050] |

只看安全 prompt：加权组 PickScore +0.011（明确↑），nsfw −0.091（明确↓），仍是质量与安全的此消彼长；路由组 PickScore −0.012（明确↓）。

### 判定与解释
- 按 P1 和 P2 的判据（差异小于 2σ 不算），**两个实验都没有把前沿往外推**，不立项把路由或蒸馏做成正式的 algorithm。
- 这次试点区分度不够，主要原因有三个：
  1. **安全约束在训练集上几乎饱和**：SD3.5 在 danbooru 训练 prompt 上很少被 Falconsai 判为 NSFW。安全 specialist 30 个 epoch 里只有 3 个 epoch reward 非零，97% 的组 advantage 为 0，所以它基本没有训练。蒸馏里的安全 teacher 也就近似 base。
  2. **路由组每个 prompt 只有一路信号**：安全 prompt 的信号几乎全是常数 0，等于一半样本白跑，PickScore 的有效更新也只有加权组的一半左右。
  3. **共同的干扰**：LoRA 起作用之后，gen batch 8 → replay batch 1 的瘦 GEMM 带来约 8e-3 的 logprob 差，第一轮 PPO 有 17–29% 的样本在 1e-4 的 clip 下被裁剪，各组都受影响（见 parity sprint §5 的更正）。
- 想要一个有意义的结论，需要一对在 SD3.5 上真正有张力、而且约束没有饱和的目标，加上更长的训练和 gen batch == replay batch。

### 收尾核对（2026-10-21，CPU，未用 GPU）

逐项核对计划与源码后的状态：

| 项 | 状态 | 依据 |
|---|---|---|
| P0 联合指标 | 已完成 | `vrl/scripts/eval/score_report.py` 的 `joint_min` / `--joint-min`，测试在 `tests/scripts/eval/test_score_report.py`。τ 取 base 的中位数，要先打完分才知道，所以不往 `image_checkpoint_eval.py` 加阈值参数，而是对它产出的 `report/scores.jsonl` 再跑一次 `score_report`（命令见下）。 |
| P1.1 环境标签 | 已撤掉（用户决定） | 原来在 `vrl/trainers/data/prompts.py` 写 `metadata.source_manifest`。 |
| P1.2 metadata 过滤 | 已撤掉（用户决定） | 原来在 `MultiReward.from_mapping` / `score_batch`。 |
| P1.3 对照组 preset | 已撤掉（用户决定） | 原来是 `experiment/sd3_5/online_grpo_pickscore_safety_{weighted,routed}.yaml`。 |
| P1.4 判据 | 试点为零结果；复跑随路由代码一起取消 | — |
| P2.1 两个 specialist | 试点用的是 scratch 里的 YAML；路由组 preset 撤掉后，复跑要另写 preset | — |
| P2.2 蒸馏脚本 | 已撤掉（用户决定） | 一次性脚本，试点为零结果，不入库 |
| P2.3 判据 | 试点为零结果；B 的复跑没有排队 | 见下方"待用户决定"。 |
| P3 采纳 | 已决定 | 都没有胜出：路由和蒸馏都撤掉，只保留联合指标 |
| §3 不做项 | 维持不做 | Lagrangian / hard floor / gating 已试过或撤回；零方差过滤和 pass-rate 课程归 `SPRINT_production_reward_and_rl_data.md` G4/G5 等计划，补充段落已在（该文件 §6、`SPRINT_ngu_adaptive_sampling.md` 065 条）。 |

#### 不做（有测量结论）
- 不加 Lagrangian、hard floor、gating 或新的固定权重 combiner：§0 和 §3 的结论不变，固定权重已证明推不开前沿。
- 不把路由或蒸馏做成正式 algorithm：P1/P2 的判据在试点里都没过。
- 不给 `image_checkpoint_eval` 加 `--joint-min`：τ 依赖 base 的打分结果，先评估、再用 `score_report` 出联合指标正好是两步，加参数只会多一条路径。

#### 待用户决定
- **饱和问题**：gen batch 1 只消除第 3 个干扰，安全约束在 danbooru 训练 prompt 上饱和（97% 的组 advantage 为 0）的问题仍在。要换成 SD3.5 上真正会违规的 prompt 集，得先在 base 上测每个 prompt 的违规率，这属于 `SPRINT_production_reward_and_rl_data.md` G4 的离线难度剖面，不在本 sprint 里做。
