# SPRINT: Janus-Pro-R1 机制级复现，Janus-Pro-1B 真实 checkpoint 训练

## 0. 结论

本 sprint 的目标不是复现 Janus-Pro-R1 的 7B paper result，而是复现它的核心训练机制，并用 Janus-Pro-1B 真实 checkpoint 跑一个可评估的训练 run。

一句话边界：

```text
实现 Janus-Pro-R1-style AR visual RL pipeline；
第一阶段用 Janus-Pro-1B 真实 checkpoint 跑完整训练、fixed eval、checkpoint/resume。
```

这个说法成立，因为 Janus-Pro-1B 和 Janus-Pro-7B 属于同一个 Janus-Pro family，核心 AR image token generation path 一致。规模、reward、数据、分布式训练策略不同，因此第一阶段只能叫 mechanism-level reproduction，不能叫 paper-result reproduction。

## 1. 参考实现事实

参考 repo：

```text
/home/mingfeiguo/Desktop/Janus-Pro-R1
```

关键文件：

```text
/home/mingfeiguo/Desktop/Janus-Pro-R1/README.md
/home/mingfeiguo/Desktop/Janus-Pro-R1/janus-rl/README.md
/home/mingfeiguo/Desktop/Janus-Pro-R1/janus-rl/recipes/t2i_generation/grpo.yml
/home/mingfeiguo/Desktop/Janus-Pro-R1/janus-rl/recipes/accelerate_configs/zero2.yaml
/home/mingfeiguo/Desktop/Janus-Pro-R1/janus-rl/src/open_r1/grpo_t2i.py
/home/mingfeiguo/Desktop/Janus-Pro-R1/janus-rl/src/open_r1/grpo_trainer_t2iv1.py
/home/mingfeiguo/Desktop/Janus-Pro-R1/janus-rl/src/open_r1/llama.py
```

从参考实现抽出的硬事实：

- 官方 RL 使用 Janus-Pro-7B。
- 官方启动方式是 Accelerate + DeepSpeed ZeRO-2，`NUM_PROCESSES=8`。
- 官方 T2I GRPO recipe 使用 `num_generations=8`、`max_completion_length=576`、`guidance_scale=5.0`、`temperature=0.9`、`bf16=true`。
- Stage 2 把图像生成看成 token-level Markov decision process，并用 GRPO 优化。
- R1 不是普通一次性 T2I，它是三段生成：first image generation、self-check text、final image regeneration。
- 官方 trainer 对三段分别构造 logprob / KL / advantage：
  - `first_gen`: 第一张图像的 image token 轨迹。
  - `compre`: self-check 文本轨迹。
  - `final_gen`: 最终图像的 image token 轨迹。
- 官方 reward 是 bi-level QA / InternVL reward。第一阶段不直接复刻它，先用 OCR/simple reward 替代，保证链路可 debug。

## 2. 当前 repo 起点

当前 repo 已经有这些基础能力：

```text
configs/experiment/janus_pro_1b_ocr_grpo.yaml
configs/model/janus_pro/1b.yaml
configs/sampling/janus_384_576tok.yaml
vrl/models/families/janus_pro/policy.py
vrl/models/families/janus_pro/executor.py
vrl/algorithms/grpo/token.py
vrl/rollouts/evaluators/ar/token_logprob.py
vrl/rollouts/packers/ar_discrete.py
vrl/scripts/janus_pro/train.py
```

已经具备：

- Janus-Pro-1B real checkpoint loading。
- LoRA-based AR policy。
- plain T2I image token sampling。
- image token decode。
- per-token old logprob。
- TokenGRPO `[B, L]` token loss。
- OCR reward path。
- Ray single-GPU rollout path。

还缺：

- Janus-Pro-R1-style `generate_with_refine`。
- self-check text segment。
- final image regeneration segment。
- 多段 token trajectory 的统一 `OutputBatch.extra` schema。
- 多段 rollout packer。
- 多段 logprob evaluator。
- R1-style fixed eval / contact sheet。
- 与官方 7B recipe 的 gap 文档化。

## 3. 非目标

本 sprint 明确不做：

- 不做 Janus-Pro-7B full reproduction。
- 不做官方 8GPU ZeRO-2 分布式复现。
- 不做 Stage 1 SFT。
- 不做 image editing GRPO。
- 不做完整 InternVL 26B / online API reward。
- 不做 vLLM / SGLang serving。
- 不做 async RL。
- 不把 reward、advantage、ExperienceBatch 放进 engine。

如果需要直接复制 Janus-Pro-R1 代码，必须保留 Apache-2.0 license header 和来源注释。默认策略是参考行为并在本 repo 的现有 policy / executor / rollout 架构里重写。

## 4. 目标架构

目标不是新建一个平行系统，而是在现有 Janus-Pro 路径旁边增加一个 R1 task variant。

```text
janus_pro plain T2I:
  task = ar_t2i
  executor = JanusProPipelineExecutor
  packer = ARDiscreteRolloutPacker
  evaluator = TokenLogProbEvaluator

janus_pro_r1 R1-style T2I:
  task = ar_t2i_r1
  executor = JanusProR1PipelineExecutor
  packer = ARR1RolloutPacker
  evaluator = MultiSegmentTokenLogProbEvaluator
```

注意这里分两层：

```text
model.family = janus_pro      # checkpoint / builder 仍然复用 Janus-Pro-1B
rollout family = janus_pro_r1 # engine registry 里必须是独立 key
```

task variant 变成：

```text
task = ar_t2i_r1
```

这样不会污染现有 `janus_pro_1b_ocr_grpo` 普通路径，也不会让 plain Janus executor 和 R1 executor 在 engine registry 里互相覆盖。

## 5. 需要新增/修改的类

### 5.1 `JanusR1GenerationResult`

新增位置：

```text
vrl/models/families/janus_pro/r1_types.py
```

职责：

- 表达一次 R1-style generation 的三段输出。
- 不包含 reward。
- 不包含 GRPO advantage。

目标字段：

```python
@dataclass(slots=True)
class JanusR1Segment:
    name: str
    token_ids: torch.Tensor
    token_log_probs: torch.Tensor | None
    token_mask: torch.Tensor
    prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor
    visual: bool
    cfg: bool


@dataclass(slots=True)
class JanusR1GenerationResult:
    initial_image: torch.Tensor
    final_image: torch.Tensor
    selfcheck: torch.Tensor
    segments: dict[str, JanusR1Segment]
    context: dict[str, Any]
```

三个 segment 名字固定：

```text
initial_image
selfcheck_text
final_image
```

### 5.2 `JanusProPolicy.generate_with_refine`

修改位置：

```text
vrl/models/families/janus_pro/policy.py
```

职责：

- 复刻 Janus-Pro-R1 的三段 generation 机制。
- 复用当前 `sample_image_tokens` / `decode_image_tokens` / `forward_image_logits` 的底层逻辑，避免复制一套独立 AR loop。
- 返回 `JanusR1GenerationResult`。

最小接口：

```python
def generate_with_refine(
    self,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    cfg_weight: float,
    temperature: float,
    image_token_num: int,
    max_reflect_len: int,
    task_stages: tuple[str, ...] = ("initial_image", "selfcheck_text", "final_image"),
) -> JanusR1GenerationResult:
    ...
```

必须实现的行为：

- initial image：从 prompt 生成 576 image tokens。
- self-check text：把第一张图作为视觉输入，生成 Yes/No 开头的 reflection text。
- final image：如果 self-check 判断需要 refine，则生成第二张图；否则 final image 使用 initial image。
- 每段都保留 replay forward 所需的 prompt/context tensor。

第一阶段允许简化：

- `selfcheck_text` 可以先只训练 Yes/No token 或短 reflection，不要求完整 paper CoT。
- `final_image` 可以总是生成，也可以按 self-check 选择；选择策略必须写进 config，不能硬编码。

### 5.3 `JanusProR1PipelineExecutor`

新增位置：

```text
vrl/models/families/janus_pro/r1_executor.py
```

职责：

- engine/executor 层只负责 generation。
- 不算 reward。
- 不算 advantage。
- 不 import `vrl.rollouts.*`。

目标接口：

```python
class JanusProR1PipelineExecutor(ARPipelineExecutorBase):
    family = "janus_pro_r1"
    task = "ar_t2i_r1"

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        ...
```

`OutputBatch.output`：

```text
final_image tensor [B, 3, H, W]
```

`OutputBatch.extra` 必须包含：

```text
initial_image
final_image
selfcheck
segments.initial_image.token_ids
segments.initial_image.token_log_probs
segments.initial_image.token_mask
segments.selfcheck_text.token_ids
segments.selfcheck_text.token_log_probs
segments.selfcheck_text.token_mask
segments.final_image.token_ids
segments.final_image.token_log_probs
segments.final_image.token_mask
segment replay contexts
```

注意：

- R1 executor 仍然按 prompt/sample chunk 执行。
- 不做 token-level cross-request scheduler。
- 单卡真实训练先走 direct executor 或现有 Ray single-GPU rollout backend；不要为了第一版引入新的分布式系统。
- Ray large rollout 以后只需要实现 `forward_chunk()` / `gather_chunks()`，但本 sprint 不强求。

### 5.4 `ARR1RolloutPacker`

新增位置：

```text
vrl/rollouts/packers/ar_r1.py
```

职责：

- 把 `OutputBatch` 转成 trainer 可消费的 `RolloutBatch`。
- reward 默认打在 final image 上。
- 保留 initial/final/contact sheet 所需图像。
- 保留多段 token trajectory。

第一阶段 reward 规则：

```text
reward_target = final_image
reward_prompt = original prompt
reward = OCR/simple reward
```

`RolloutBatch.extras` 目标结构：

```text
extras["r1_segments"] = {
  "initial_image": {...},
  "selfcheck_text": {...},
  "final_image": {...},
}
extras["initial_image"] = ...
extras["final_image"] = ...
extras["selfcheck"] = ...
```

不要把三段强行拼成一个 `[B, L_total]`，因为 image tokens 和 text tokens 的 projection head 不同。保持 segment 结构更安全。

### 5.5 `MultiSegmentTokenLogProbEvaluator`

新增位置：

```text
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
```

职责：

- 对 `RolloutBatch.extras["r1_segments"]` 的每个 segment replay forward。
- image segment 使用 Janus image vocab / `gen_head`。
- text segment 使用 language vocab / text logits。
- 支持 LoRA-off reference pass。

目标输出：

```python
@dataclass(slots=True)
class MultiSegmentSignalBatch:
    segments: dict[str, SignalBatch]
```

或者复用 `SignalBatch.aux["segments"]`，但必须避免把 image/text logits 混成一个 tensor。

第一阶段可以只训练 image segments：

```text
train_segments = ["initial_image", "final_image"]
```

如果 self-check text replay 复杂，可以先收集但不训练：

```text
selfcheck_text.train = false
```

这个开关必须显式在 config 里，不能靠代码注释。

### 5.6 `MultiSegmentTokenGRPO`

新增位置：

```text
vrl/algorithms/grpo/multisegment.py
```

职责：

- 复用 `TokenGRPO.compute_signal_loss()` 对每个 segment 算 loss。
- 对多个 segment 做 weighted mean。

目标 config：

```yaml
algorithm:
  kind: token_grpo_multisegment
  segment_weights:
    initial_image: 1.0
    selfcheck_text: 0.0
    final_image: 1.0
```

第一阶段默认：

```text
initial_image: 1.0
selfcheck_text: 0.0
final_image: 1.0
```

原因：

- 这保留 Janus-Pro-R1 的两段 image generation 训练语义。
- 避免一开始把 text self-check 的 credit assignment 和 image OCR reward 混在一起。
- 后续再逐步打开 self-check text loss。

### 5.7 `train_janus_pro_r1_ocr_grpo`

修改位置：

```text
vrl/scripts/janus_pro/train.py
```

新增入口：

```python
async def train_janus_pro_r1_ocr_grpo(cfg: DictConfig) -> None:
    ...
```

职责：

- 加载 Janus-Pro-1B。
- 构造 `task=ar_t2i_r1` runtime。
- 使用 `ARR1RolloutPacker`。
- 使用 `MultiSegmentTokenLogProbEvaluator`。
- 使用 `MultiSegmentTokenGRPO`。
- 保存 metrics / eval / checkpoint。

不要复用普通 `train_janus_pro_ocr_grpo()` 的 evaluator/packer，否则 R1 的多段 trajectory 会被压扁。

## 6. Config 设计

新增：

```text
configs/experiment/janus_pro_1b_r1_ocr_grpo.yaml
configs/sampling/janus_r1_384_576tok.yaml
configs/base/algorithm/token_grpo_multisegment.yaml
configs/base/rollout/ar_r1.yaml
```

`configs/experiment/janus_pro_1b_r1_ocr_grpo.yaml`：

```yaml
# Janus-Pro-1B + Janus-Pro-R1-style OCR GRPO real-checkpoint run.
defaults:
  - /base/algorithm/token_grpo_multisegment
  - /base/actor
  - /base/rollout/ar_r1
  - /base/distributed/ray_rollout_single_gpu
  - /base/trainer
  - /model/janus_pro/1b
  - /sampling/janus_r1_384_576tok

reward:
  components:
    ocr: 1.0
  kwargs:
    ocr:
      debug_dir: ""

data:
  manifest: datasets/ocr/train.txt

trainer:
  entrypoint: vrl.scripts.janus_pro.train:train_janus_pro_r1_ocr_grpo
  output_dir: outputs/janus_pro_1b_r1_ocr_grpo
  total_epochs: 80
  save_freq: 10
  debug:
    first_step: false

rollout:
  n_samples_per_prompt: 2
  rollout_batch_size: 1
```

`configs/sampling/janus_r1_384_576tok.yaml`：

```yaml
sampling:
  image_token_num: 576
  image_size: 384
  cfg_weight: 5.0
  temperature: 0.9
  max_reflect_len: 80
  r1:
    final_image_policy: always_generate  # always_generate | use_selfcheck
    train_segments:
      initial_image: true
      selfcheck_text: false
      final_image: true
```

第一阶段用真实 checkpoint、小 batch、完整训练：

```text
n_samples_per_prompt = 2
rollout_batch_size = 1
total_epochs = 80
```

如果 80 epoch 曲线和显存都稳定，再升到：

```text
n_samples_per_prompt = 4
rollout_batch_size = 1 或 2
```

不要第一天直接上官方 `num_generations=8`。

## 7. 验证标准

### 7.1 单元测试

新增测试：

```text
tests/models/test_janus_r1_policy.py
tests/rollouts/test_ar_r1_packer.py
tests/rollouts/test_janus_pro_r1_wiring.py
tests/rollouts/test_multisegment_token_logprob.py
tests/algorithms/test_multisegment_token_grpo.py
tests/config/test_janus_pro_r1_config.py
tests/rollouts/test_family_registry.py
```

必须覆盖：

- R1 policy 返回三个 segment。
- image segment token shape 是 `[B, 576]`。
- self-check text segment token shape 是 `[B, <= max_reflect_len]`。
- final image shape 是 `[B, 3, 384, 384]`。
- packer 不把 image/text token 混成一个 tensor。
- evaluator 能对 image segment 算 fresh logprob。
- `selfcheck_text.train=false` 时不会进入 loss。
- `segment_weights` 为 0 的 segment 不影响 loss。
- config 能通过统一 loader 加载。

### 7.2 真实 checkpoint training run

必须用真实 Janus-Pro-1B checkpoint 跑一个多 epoch 训练，不允许只靠 fake model，也不把 1 epoch 短跑当完成标准。

命令：

```bash
python -m vrl.scripts.train \
  --config experiment/janus_pro_1b_r1_ocr_grpo \
  trainer.total_epochs=80 \
  rollout.n_samples_per_prompt=2 \
  rollout.rollout_batch_size=1 \
  trainer.output_dir=outputs/janus_pro_1b_r1_ocr_grpo_real_run
```

通过标准：

- 能 load `deepseek-community/Janus-Pro-1B`。
- 能生成 initial image。
- 能生成 self-check text。
- 能生成 final image。
- 能算 OCR reward。
- 能算至少一个 image segment 的 TokenGRPO loss。
- 能 backward。
- 至少保存 `checkpoint-10`、`checkpoint-20` 和 `checkpoint-final`。
- `metrics.csv` 至少有 80 行非 NaN loss / reward / grad_norm。
- `eval_metrics.csv` 至少包含 epoch 0 / 40 / 80 的固定 eval。
- 输出目录包含可人工检查的 initial/final contact sheet。
- 训练结束后写出 `REFERENCE_GAP.md`。

### 7.3 resume run

命令：

```bash
python -m vrl.scripts.train \
  --config experiment/janus_pro_1b_r1_ocr_grpo \
  trainer.total_epochs=100 \
  trainer.resume_from=outputs/janus_pro_1b_r1_ocr_grpo_real_run/checkpoint-80 \
  trainer.output_dir=outputs/janus_pro_1b_r1_ocr_grpo_real_run
```

通过标准：

- resume 后从 epoch 80 继续到 epoch 100。
- LoRA weights 恢复。
- optimizer state 恢复。
- RNG state 恢复。
- 继续训练不会覆盖旧 metrics header。

### 7.4 fixed eval

第一阶段 fixed eval 不追求 reward 上升，只要求可比：

- 固定 8 个 OCR test prompts。
- 固定 seed。
- 输出 initial/final 两张 contact sheet。
- 记录 `eval_metrics.csv`。
- 明确 eval reward 是 final image reward，不是 training rollout reward。

## 8. Debug 顺序

如果失败，按这个顺序查，不要跳到 reward：

1. Janus-Pro-1B 是否能 plain T2I generate。
2. R1 initial image token ids 是否 shape 正确。
3. VQ decode 是否正常。
4. self-check text 是否能生成 Yes/No 或短文本。
5. final image 是否能生成。
6. collected old logprob 是否 finite。
7. replay fresh logprob 是否和 old logprob 在 no-update 情况下接近。
8. ref logprob 是否来自 LoRA-off path，不允许静默等于 current。
9. reward 是否 finite。
10. advantage 是否非全零。
11. loss / grad_norm 是否 finite。
12. checkpoint/resume 是否恢复相同 trainable hash。

## 9. 与官方 Janus-Pro-R1 的差异记录

每次真实训练输出目录必须保存：

```text
resolved_config.yaml
REFERENCE_GAP.md
metrics.csv
eval_metrics.csv
```

`REFERENCE_GAP.md` 必须写清楚：

- 使用 Janus-Pro-1B，不是 Janus-Pro-7B。
- 单卡真实训练，不是 8GPU ZeRO-2。
- 使用 OCR/simple reward，不是官方 bi-level QA / InternVL reward。
- 使用现有 TokenGRPO / MultiSegmentTokenGRPO，不是直接运行官方 TRL trainer。
- 目标是 mechanism-level reproduction，不是 paper-result reproduction。

## 10. 完成定义

本 sprint 完成需要同时满足：

- `experiment/janus_pro_1b_r1_ocr_grpo` 能加载。
- fake-model 单元测试全部通过。
- Janus-Pro-1B 真实 checkpoint 80 epoch training run 通过。
- 从 `checkpoint-80` resume 到 epoch 100 通过。
- fixed eval 输出 initial/final contact sheets。
- README 或 experiment note 明确标注这是 Janus-Pro-R1-style 1B mechanism-level real run。
- 现有 `janus_pro_1b_ocr_grpo` 普通 AR recipe 不被破坏。
- 现有 SD3.5 OCR GRPO config/load tests 不被破坏。

## 11. 后续阶段

本 sprint 通过后，再考虑这些扩展：

- 把 `n_samples_per_prompt` 从 2 提到 4/8。
- 引入 Janus-Pro-R1 training data。
- 引入官方 QA / InternVL reward。
- 打开 `selfcheck_text` segment loss。
- 支持 image editing R1 path。
- 支持 Janus-Pro-7B + Ray/FSDP/ZeRO 分布式训练。
- 比较 Janus-Pro-R1 checkpoint 的 inference 行为。
