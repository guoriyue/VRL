# SPRINT: Cosmos + Kling fixed eval and learning-signal calibration（planned）

状态：**done（已提交 main，b00c373；2026-06-17 归档至 done/）**。T1-T5 全部完成：EvalConfig (vrl/trainers/core/types.py) + schema 接线、load_eval_prompt_examples_from_config (vrl/trainers/data/prompts.py)、_run_fixed_eval/_prepare_eval_metrics_csv/_write_eval_metric_row (vrl/scripts/common/online.py) 以及 cosmos kling 实验 eval block 已开启。未做（属未来工作）：T6 signal-calibration sweep (lr + advantage scale) 尚未跑。（注：早先记录的"未提交/单测缺 import pytest"已过时——代码已随 b00c373 落 main。）

> **Superseded（2026-07-13）：**上述训练内 fixed-eval 接线随后由 `6dc7219` 删除；
> `load_eval_prompt_examples_from_config` 和 online fixed-eval helper 当前都不存在。
> 固定 checkpoint 评估现在走独立入口 `vrl/scripts/eval/cosmos_predict25_kling_eval.py`。
> 本文其余内容保留为当时的信号诊断记录，不是当前运行手册。
VideoReward 训练里最危险的误判：`reward_mean` 没涨不一定等于 RL 没学，因为当前在线指标混在
“每个 epoch 换一批 prompt”与“小梯度更新”两件事里。先把学习曲线测准，再调更新强度；不要用更多
epoch 或吞吐优化掩盖问题。

本 sprint 依赖已落地的 paper-shaped batch 内存修复：

- `SPRINT_streaming_rollout_accumulation.md`: 一个 optimizer target batch 分成 microbatch
  `collect -> backward -> release`，避免一次持有 32 个 prompt group。
- `SPRINT_memory_budgeted_microbatch.md`: `rollout.microbatch_size` 是唯一手填切片大小，
  `actor.gradient_accumulation_steps` 派生。

不依赖 async overlap。`SPRINT_microbatch_pipeline_overlap.md` 已退役为 scope guard；microbatch 只保留
同步内存切片语义，不能证明 reward 真的变好。

---

## 0. 核心判断

**先加 fixed eval，再做信号校准。**

当前在线训练 loop 每个 epoch 都重新抽 prompt：

```python
idx = sample_prompt_indices(
    rng,
    num_examples=len(examples),
    rollout_batch_size=trainer_config.rollout_batch_size,
    ...
    epoch=epoch,
)
example_batch = [examples[i] for i in idx]
```

随后 metrics CSV 只写这批训练 prompt 的 reward：

```python
"epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,..."
```

所以这条曲线同时包含两种变化：

- 模型参数有没有变好；
- 这一轮抽到的 32 个 prompt 本身难不难。

只看这个 `reward_mean` 不能判断学习。正确的学习曲线必须来自同一组 eval prompt、同一套 seed grid、
同一套采样配置，每个 eval 周期重复生成并打 Kling reward。

第二个问题是更新强度。Cosmos Kling 当前配置是：

```yaml
actor:
  optim:
    lr: 3.0e-5
  ppo_epochs: 1
  timestep_fraction: 0.5

rollout:
  n_samples_per_prompt: 8
  rollout_batch_size: 32
  microbatch_size: 1
```

DiffusionNFT loss 里 reward 信号被 `advantage_high` 缩放：

```python
adv = torch.clamp(advantages, -cfg.advantage_high, cfg.advantage_high)
reward_mix = ((adv / cfg.advantage_high) / 2.0 + 0.5).clamp(0.0, 1.0)
policy_loss = original_policy_loss.mean() * float(cfg.advantage_high)
```

当同一 prompt 的 8 个样本 reward 很接近时，advantage 本来就小；再用 `advantage_high=5.0` 缩放，
`reward_mix` 会长期贴近 0.5，梯度自然很弱。这个要用 fixed eval 证明后再调，不能靠 rotating
train reward 猜。

---

## 1. 目标

### 1.1 可观测性目标

新增在线 fixed eval：

```text
before training:
  evaluate fixed eval prompt set once -> baseline row

after every eval_freq epoch:
  evaluate the same prompt set with the same seed grid
  write eval_metrics.csv
```

输出至少包含：

```text
epoch,global_step,prompt_count,samples_per_prompt,reward_mean,reward_std,reward_stderr,seed
```

有 reward components 时增加：

```text
r_<component_name>
```

训练 metrics 和 eval metrics 分文件：

- `metrics.csv`: 训练 batch 指标，继续保持原 schema。
- `eval_metrics.csv`: fixed eval 指标，用来判断学习。

这样不会让旧分析脚本因为 `metrics.csv` schema 变动而坏掉。

### 1.2 学习信号目标

在 fixed eval 可用后，做小范围 signal sweep：

```yaml
actor:
  optim:
    lr: 1.0e-4

algorithm:
  advantage_high: 1.0
```

第一批 sweep 只动：

- LoRA learning rate；
- DiffusionNFT advantage scale。

暂时不动：

- `sampling.num_steps=20`;
- `actor.timestep_fraction=0.5`;
- `rollout.n_samples_per_prompt=8`;
- `rollout.rollout_batch_size=32`;
- `rollout.microbatch_size=1`。

原因：这些是 paper-shaped batch 与当前 OOM 修复的边界。先证明 RL 信号方向，再考虑吞吐或算法大改。

---

## 2. 现状证据

### 2.1 数据集已经有 eval manifest

`configs/dataset/videophy.yaml` 已经声明固定 eval 集：

```yaml
data:
  loader: prompt_manifest
  manifest: datasets/videophy/train.txt
  eval_manifest: datasets/videophy/eval.txt
```

所以不要新增第二份 prompt-list 配置。`data.eval_manifest` 是 fixed eval 的 prompt source of truth。

### 2.2 当前 loader 只读 train manifest

`load_prompt_examples_from_config()` 当前只读 `data.manifest`：

```python
manifest = cfg_get(data_cfg, "manifest", None)
if not manifest:
    raise ValueError("config missing required field: data.manifest")
```

fixed eval 需要新增 `load_eval_prompt_examples_from_config(data_cfg)`，读取 `data.eval_manifest`。
当 `trainer.eval.enabled=true` 但没有 `data.eval_manifest` 时应 fail fast，不要 fallback 到 train
manifest。否则 eval 可能和训练 prompt 混在一起，曲线又会失真。

### 2.3 不能直接用 `collect_training_batch()` 做 eval

`OnlineTrainer.collect_training_batch(prompts)` 会写 trainer 状态并计算 advantage：

```python
if prompts is not None:
    self.prompts = prompts
...
advantages_all = self.algorithm.compute_advantages_from_tensors(
    all_rewards,
    all_group_ids,
)
```

fixed eval 不应该改 `self.prompts`，也不需要 advantage/backward。正确切点是 collector 层：

```python
unscored = await collector.collect_unscored(...)
batches = await collector.score_rollouts([...])
```

eval helper 应该只做 `generate -> reward -> summarize -> release`。

### 2.4 seed 可以走现有 request builder

`GenerationRequestBuilder` 已经支持从 kwargs 注入 seed：

```python
seed = kwargs.get("seed")
if seed is not None:
    sampling["seed"] = seed
```

fixed eval 应该用固定 seed grid。collector 每个 request 只能接一个 base seed，样本级 offset
由 generation execution 层处理，所以 eval helper 应传 prompt-group base seed：

```text
group_seed(prompt_index) = base_seed + prompt_index * samples_per_prompt
sample_seed(prompt_index, sample_index) = group_seed(prompt_index) + sample_index
```

如果一个 request 内生成多个 samples，生成执行层已有 sample offset 逻辑；eval helper 只要保证每个
prompt group 的 base seed 固定即可。

---

## 3. 目标设计

### 3.1 配置

新增 trainer 级 eval config：

```yaml
trainer:
  eval:
    enabled: true
    freq: 1
    samples_per_prompt: 2
    max_prompts: 35
    seed: 20260614
```

字段语义：

| 字段 | 语义 |
|---|---|
| `enabled` | 是否在 online loop 里跑 fixed eval。默认 `false`。 |
| `freq` | 每多少个 epoch eval 一次。`1` = 每个 epoch 结束后 eval。 |
| `samples_per_prompt` | eval 每个 prompt 生成几个视频。默认不要复用训练的 8，避免 eval 成本过高；需要统计更稳时再加。 |
| `max_prompts` | eval prompt 上限。`0` = 全部 `data.eval_manifest`。VideoPhy eval 当前 35 条，`35` 可完整覆盖。 |
| `seed` | fixed eval base seed。resume 后必须保持不变。 |

不要新增 `trainer.eval.manifest`。eval prompt 来自 `data.eval_manifest`，避免两个 source of truth。

### 3.2 输出

新增：

```text
<output_dir>/eval_metrics.csv
```

baseline 行建议用：

```text
epoch=-1
```

之后每个 eval 周期写当前 epoch：

```text
epoch=0,1,2,...
```

如果 resume：

- 不覆盖已有 eval rows；
- 继续 append；
- seed/grid 不变；
- checkpoint metadata 不需要写进 CSV，`global_step` 足够定位。

### 3.3 eval 时机

推荐顺序：

```text
build runtime / restore checkpoint
prepare metrics files
run fixed eval baseline if enabled

for epoch:
  train one optimizer update
  write training metrics.csv row
  run fixed eval if due
  save checkpoint if due
```

先写 training row，再跑 eval。原因是 `reward_fn.last_components` 会被 eval 覆盖；训练 row 必须先从
training batch 的 reward components 取值。下一轮训练开始时 streaming path 已经会
`reward_fn.reset_components()`，所以 eval 后不需要恢复旧 components，只需要在 eval helper 内写自己的
component means。

### 3.4 helper 放置

优先在 `vrl/scripts/common/online.py` 内实现小 helper，直到出现第二个 online recipe 复用需求。

只有当 helper 同时被 `vrl/scripts/eval/cosmos_predict25_kling_eval.py` 或多个 recipe 复用时，再抽到：

```text
vrl/scripts/common/fixed_eval.py
```

这个薄文件只有在它成为共享 online-eval 协议边界时才值得存在。不要为了省几行把它提前拆出去。

---

## 4. 分阶段计划

### T1 — 配置边界

改动：

- `vrl/trainers/core/types.py`
  - 新增 `EvalConfig` dataclass。
  - `TrainerConfig.eval: EvalConfig = field(default_factory=EvalConfig, metadata={"yaml": "trainer.eval"})`。
  - 校验：
    - `freq >= 1`;
    - `samples_per_prompt >= 1`;
    - `max_prompts >= 0`;
    - `seed >= 0`。
- `vrl/config/schema.py`
  - `TrainerSection.eval = Annotated[Any, ConfigBlock(EvalConfig)]`。

测试：

- `tests/config/test_schema.py`: `trainer.eval` 合法字段被接受，未知字段仍报警/拒绝。
- `tests/config/test_load_all_experiments.py`: 所有实验继续 load+validate。
- 新增直接 `TrainerConfig` 测试，覆盖非法 `freq/samples_per_prompt/max_prompts`。

### T2 — eval prompt loader

改动：

- `vrl/trainers/data/prompts.py`
  - 新增 `load_eval_prompt_examples_from_config(data_cfg)`。
  - 根据 loader 类型复用现有 manifest reader：
    - `prompt_manifest`: `load_prompt_manifest(data.eval_manifest)`；
    - `prompt_image_manifest`: `load_prompt_image_manifest(data.eval_manifest, ...)`。
  - `trainer.eval.enabled=true` 但缺 `data.eval_manifest` 时 fail fast。

测试：

- eval loader 对 `prompt_manifest` 正确读取 `.txt`。
- eval loader 对 `prompt_image_manifest` 保留 image/caption/task metadata。
- 缺 `data.eval_manifest` 时错误信息点名 `data.eval_manifest`。

### T3 — online fixed eval helper

改动：

- 在 online recipe runtime 建好之后加载 eval prompts。
- 新增 helper，语义是：

```text
run_fixed_eval(
  collector,
  reward_fn,
  eval_examples,
  samples_per_prompt,
  base_seed,
  max_prompts,
)
```

要求：

- 不调用 `trainer.collect_training_batch()`。
- 不调用 `trainer.backward_on_training_batch()`。
- 不改 `trainer.prompts`。
- 不改 optimizer / EMA / previous-policy adapter。
- 用和训练同一个 collector/runtime/reward_fn，避免离线 eval 路径和训练路径采样不一致。
- 用 fixed seed grid。
- 释放生成/rollout/reward artifact 引用，避免 eval 后 host RAM creep。

实现提示：

- 可以复用 `collect_prompt_batches()` 的 PromptExample 展开逻辑，但要支持 seed 注入。
- 如果直接复用 `collect_prompt_batches()` 会缺 seed 参数，就新增 eval 专用私有 helper，或者把 seed
  参数以兼容方式加到 prompt collection helper。
- eval reward components 不从 `reward_fn.last_components` 的全局状态猜，helper 内应在 eval 前
  `reward_fn.reset_components()`，eval 后立即 snapshot component means 并写入 eval row。

测试：

- 用 fake collector/reward 证明 eval 只 collect/score，不 backward。
- 用 fake PromptExample 证明 request_overrides/reference metadata 被保留。
- 证明同一个 epoch rerun 得到相同 seed grid。
- 证明 training metrics component row 不被 eval 覆盖。

### T4 — eval metrics writer

改动：

- 新增 `_prepare_eval_metrics_csv()` 和 `_write_eval_metric_row()`。
- header 根据 reward component names 扩展 `r_<name>`。
- `resume=true` 时 append，不重写旧文件。

测试：

- 无 component 时 header 稳定。
- 有 component 时列顺序稳定。
- resume 不覆盖已有 eval rows。

### T5 — Cosmos Kling experiment config

改动：

- 在 `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
  开启 fixed eval：

```yaml
trainer:
  eval:
    enabled: true
    freq: 1
    samples_per_prompt: 2
    max_prompts: 35
    seed: 20260614
```

保持不变：

```yaml
sampling:
  num_steps: 20
  cfg: false

actor:
  optim:
    lr: 3.0e-5
  timestep_fraction: 0.5

rollout:
  n_samples_per_prompt: 8
  rollout_batch_size: 32
  microbatch_size: 1
```

这一步只测准，不调参。

验证：

- 跑一个短 smoke，必须生成：
  - `metrics.csv`;
  - `eval_metrics.csv`;
  - baseline eval row；
  - epoch eval row。
- 训练 row 的 `reward_mean` 可以乱跳；判断学习只看 `eval_reward_mean`。

### T6 — signal calibration sweep

在 T1-T5 全部通过后，再跑小范围 sweep。不要一次开太多变量。

> **2026-06-16 证据修正（先读这条，决定 sweep 顺序）**：
> 磁盘上 `outputs/_confirm_cosmos25_growth/` 的固定-seed eval **null 结果全部是 `lr=3e-5`**（每个
> eval 目录名都带 `lr3e-5`；e4→e60 在 −3.0~−3.4 间晃，baseline −3.51，std≈1.0，n=4 → stderr≈0.5，
> 非单调、无趋势 = 噪声）。但 memory 一直记着 “LoRA wants lr=1e-4 not paper's 1e-5”，而
> `outputs/cosmos25_kling_probe_1e4_80ep/`（真的用 `lr=1e-4` 训了 80 epoch）**只有轮换 prompt 的
> `metrics.csv`，没有任何 fixed-eval / summary.json**。
> **结论：`lr=1e-4` 这档 LoRA 从未在固定 eval 下被判过——它才是又便宜（单卡能跑）、又真空白的首选实验。
> 不要把 `lr=3e-5` 当 baseline 重跑（已知 null），从 `lr=1e-4` 开始。**

第一组（按修正后的优先级，`lr=1e-4` 优先；`lr=3e-5` 已是磁盘上的已知 null，不重跑）：

```text
trial_a:  lr=1e-4, advantage_high=1   # memory 推荐档 + 收紧 advantage，先跑这个
trial_b:  lr=1e-4, advantage_high=5   # 仅改 lr，隔离 advantage 影响
trial_c:  lr=3e-4, advantage_high=1   # 更激进 lr，盯 grad_norm/崩溃
trial_d:  lr=1e-4, advantage_high=1, lora rank↑          # 加 LoRA 容量（单卡仍可行）
```

停止规则：

- 如果 `eval_reward_mean` 相比 baseline eval 没有超过 `reward_stderr` 量级，不声明“reward 增长”。
- 如果 `grad_norm` 暴涨、loss 非有限、reward 崩掉，回退到前一档。
- 如果所有 LoRA（含 `lr=1e-4` / advantage scale / rank↑）都不涨，转入**多卡**全参 sprint（见下方
  “全参可行性”）。**不要在单卡上试全参，也不要继续加 epoch。**

### T6.1 — 全参可行性（为什么单卡到不了，必须多卡）

`enable_full_finetune()` 当前硬 raise（`model.py:239-243`）不是偷懒，是被 DiffusionNFT 的结构逼的。
loss 每个 microbatch 要**三个完整策略态前向**（`diffusion_nft.py:228-233`）：

```python
previous_prediction = _forward_previous_policy_adapter(transformer, ...)  # set_adapter("previous")
forward_prediction  = transformer(**inputs)[0]                            # default, 带梯度
ref_prediction      = _forward_reference(transformer, ...)                # disable_adapters() = base 参考
kl_loss = ((forward_prediction - ref_prediction) ** 2).mean()            # 论文防 reward-hacking 的 KL
```

LoRA 下三态 = 1 份冻结 base + 2 个小 adapter，靠 `set_adapter`/`disable_adapters` 切换，几乎免费。
全参打穿这个假设：`previous` 要第二份完整 2B；全参原地改了 base，`disable_adapters→base` 失效，参考
分支要**第三份**完整 2B 原始权重；再加 fused AdamW 两组 fp32 矩（2B → ~16GB），而 `_create_optimizer`
（`trainer.py:51-72`）是纯 AdamW，**没有任何 CPU-offload / 8-bit / paged**。三份 2B + 16GB optimizer +
49 帧激活，32GB 装不下，也没有 offload 基建可依赖。**全参 DiffusionNFT 是多卡实验，不是单卡能试塞的。**
真要做，需要：(a) 第二张卡放 previous/base 或 optimizer state，或 (b) 先建 CPU-offload optimizer +
streamed previous/base 前向（本身是独立 sprint）。在那之前，单卡只能在 LoRA regime 里判信号。

验收：

```text
fixed eval reward mean improves beyond measurement noise on the same prompt/seed grid
```

同时保留：

```text
train reward mean may remain noisy
```

---

## 5. 非目标

- **不 sweep denoise steps。** Cosmos Predict2.5 Kling 配置已经使用 paper step budget：
  `/sampling/denoise/20_step_no_cfg` + `actor.timestep_fraction=0.5`。
- **不把 `ppo_epochs>1` 当 quick fix。** 当前 streaming accumulation 明确禁止：

  ```python
  actor.ppo_epochs must be 1 when streaming accumulation is on
  ```

  支持 `ppo_epochs>1` 需要保存或重放已释放的 microbatch replay tensors，是另一个算法/内存设计。
- **不先做 rollout/train async。** 它是吞吐优化，不解决学习曲线不可读和梯度过小；microbatch async
  已明确退役。
- **不改 `rollout_batch_size` / `n_samples_per_prompt` / `microbatch_size` 的 paper-shaped 语义。**
- **不新增 duplicated prompt list 常量。** `data.eval_manifest` 是唯一 eval prompt source of truth。
- **不提前抽薄文件。** eval helper 先贴近 online recipe；只有成为共享协议边界时才抽文件。
- **不提交 scratch outputs。** 实验输出在 `outputs/`，结论写回 sprint 或后续 info doc。

---

## 6. 成功标准

实现成功：

- `trainer.eval.enabled=true` 时，online run 写出 `eval_metrics.csv`。
- baseline 与每个 eval epoch 使用相同 prompt set 和相同 seed grid。
- eval 不触发 backward、optimizer step、EMA、previous-policy sync。
- train metrics 和 eval metrics 的 reward component 聚合互不污染。
- 所有配置 schema / online lifecycle / reward flow 单测通过。

实验成功：

- 用 fixed eval 曲线判断 Cosmos + Kling 是否真的提升。
- 如果默认 paper-shaped LoRA 配置 flat，至少完成 `lr` + `advantage_high` sweep。
- 只在 fixed eval reward 超过误差量级后，才报告“reward is growing”。

---

## 关键文件引用

- `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
- `configs/dataset/videophy.yaml`
- `configs/base/algorithm/diffusion_nft.yaml`
- `vrl/scripts/common/online.py`
- `vrl/trainers/core/types.py`
- `vrl/config/schema.py`
- `vrl/config/builders.py`
- `vrl/trainers/data/prompts.py`
- `vrl/rollouts/collector/core.py`
- `vrl/rollouts/collector/requests.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/algorithms/diffusion_nft.py`
- `vrl/scripts/eval/cosmos_predict25_kling_eval.py`
- `/home/mingfeiguo/Desktop/wm-infra/docs/papers/cosmos_predict2_5_world_simulation_with_video_foundation_models_for_physical_ai_2511.00062v2.pdf`
