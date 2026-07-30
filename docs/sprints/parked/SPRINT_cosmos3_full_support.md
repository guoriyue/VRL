# SPRINT: Cosmos3 Vision Generator — RL via the diffusion seam

状态：**parked / run-verify-gated（2026-07-30 复核）**。触发条件：有能加载 16B
checkpoint 的目标硬件与可用模型下载网络。diffusers 依赖门已经完成：`cosmos` extra 声明
`diffusers>=0.39.0,<0.40`，公开类名是 `Cosmos3OmniPipeline`。范围：**只做 Cosmos3 的
vision 生成器**（`vrl/models/families/cosmos/cosmos3/`）；registry/family skeleton 与 CPU
契约已落，剩余的是 checkpoint、T2V、logprob 与 RL 的真实运行门禁。

> **reasoner-judge 已单独 ship**（`reward: add cosmos3 reasoner judge`）：`vrl/rewards/models/cosmos3_reasoner.py` + config + 注册 + 测试。它是 VLM 裁判（视频→分数），属 reward seam，**不在本 sprint**。本 sprint 只管"生成视频"的那半。

## 0. 一句话

Cosmos3（`model_type="cosmos3_omni"`）的生成器是一个 **diffusion MoT**（Mixture-of-Transformers）。它已经是 diffusers 格式，`Cosmos3OmniPipeline` 自 0.39.0 起公开发布，且不需要 NVIDIA 原生 cosmos-rl 代码。依赖与 family skeleton 已完成；剩余工作是用真实 checkpoint 验证 T2V 和 `flow_grpo` logprob。

## 0.1 Readiness verdict（2026-06-28）

**不能移到 `done/`。** 这个 sprint 的结论是"已落 skeleton，未过 run-verify gate"。

已完成：
- reasoner-judge 已单独提交并推送；它属于 reward seam，不属于本生成器 sprint 的 done gate。
- MR0 contract/probe 已跑出明确结论：Cosmos3-Nano 是 diffusers-format MoT。
- 依赖门已完成：仓库声明并锁定 diffusers 0.39，`Cosmos3OmniPipeline` 可 import。
- `cosmos3` registry、family model、full-sequence executor binding 与 replay runtime 已提交，
  并通过 CPU/registry/config 级验证。

未完成的 done gate：
- MR1：需要 `Cosmos3OmniPipeline.from_pretrained(...)` 真实 load，并产出 T2V clip。
- MR2 的代码已经落地；仍需要 family executor 真产出 T2V clip，并与 MR1 的 pipeline
  结果核对，证明 `packed_static` 装配正确。
- MR3：需要 first-step logprob diff≈0，证明 generator RL-eligible。
- MR4：需要 LoRA RL run 满足 clip_fraction>0、artifact 连贯、eval reward >2σ。

因此它留在 `docs/sprints/parked/`，直到上述硬件事件发生；目前 MR0、依赖门和 MR2 的
CPU 实现已经落地，MR1/MR2 的真实运行门以及 MR3/MR4 尚未完成。

## 1. 权威事实（已读源 / 已实测，非推断）

### 1.1 生成器 = diffusers 格式的 MoT（实测确认）

- `nvidia/Cosmos3-Nano` 的 checkpoint **本就是 diffusers 格式**：`transformer/`(7 分片) + `vae/`(Wan2.2-TI2V, z_dim=48) + `scheduler/`(UniPCMultistep, flow_shift=10) + `model_index.json`，还附 `example_t2v_diffusers_output.mp4`。
- 生成器类 `Cosmos3OmniTransformer` + pipeline `Cosmos3OmniPipeline` 已在 diffusers 0.39 公开发布（实测 `from_config()` 实例化 = **15.17B 参数**）。**不能**塞进旧的 `CosmosTransformer3DModel`（config `model_type=qwen3_vl_text`/`use_moe`/`unified_3d_mrope` 不兼容）。
- 架构 = **MoT，不是路由 MoE**：一个 transformer，Qwen3-VL 因果 reasoner 流 + 双向 diffusion 生成流，共享 joint attention 但每层**分开的参数集**；`_moe_gen` key = 生成流权重（`use_moe=true` 选它）。Nano 16B = 8B reasoner + 8B generator。flow-matching DiT，无独立 text encoder。

### 1.2 forward 不是干净的扩散 DiT（MR2 实现已落，真实 parity 待验）

实测 `Cosmos3OmniTransformer.forward`：
```
forward(input_ids, text_indexes, position_ids, und_len, sequence_length,
        vision_tokens[], vision_sequence_indexes, vision_timesteps, vision_noisy_frame_indexes,
        sound_*, action_*) -> (preds_vision, preds_sound, preds_action)
```
text/vision/sound/action 交错成一个联合序列。denoise 蓝本（`pipeline_cosmos3_omni.py:1618-1744`）：每步 `vision_timesteps=full(timestep)` + splice `vision_tokens=[latents]` → `transformer(**cond_pack,...)` 得 `preds_vision`(velocity) → `_mask_velocity_predictions` → CFG 合成 → `scheduler.step`。
- cond/uncond 的 `packed_static`（text/vision segment + MRoPE `position_ids` 拼接 +
  `vision_sequence_indexes` 等）在 diffusers pipeline `__call__` 中内联，没有公开复用方法。
  仓库已经在 `vrl/models/families/cosmos/cosmos3/model.py` 实现
  `_assemble_packed_static`：`prepare_sampling` 与 `restore_eval_state` 都调用同一 helper，
  每步只注入 `vision_tokens` 与 `vision_timesteps`。
- ⚠️ **剩余难点不是再写一份 assembly，而是验证现有镜像没有随上游漂移。** 代码仍依赖
  pipeline 的私有 `_prepare_text_segment`、`_prepare_vision_segment` 与
  `_mask_velocity_predictions`，因此必须用真实 checkpoint 做 pipeline/family parity；
  import-only 或 registry 测试不能把 MR2 判为完成。

### 1.3 logprob = 复用现有 flow_grpo，不是新算法

每个 denoise step 是高斯转移 `x_{t-1}=μ(x_t,σ)+std·ε`，policy-gradient 的 ratio = 当前策略均值 μ_θ vs 参考均值 μ_ref 的高斯似然比：
```text
log_prob_ratio = -( ||x_sample-μ_θ||² - ||x_sample-μ_ref||² ) / (2·σ_next²·(η²+2η))
```
- **这就是仓库已有的
  `vrl/math/denoise/flow_matching.py:sde_step_with_logprob`
  （DanceGRPO/Flow-GRPO 同构）**。collect 路径由
  `vrl/generation/steps/denoise/loop.py` 调用，replay 路径由
  `vrl/rollouts/evaluators/denoise/sde_logprob.py` 复算；Cosmos3 model 只返回 raw flow
  velocity。→ MR3 验证 Cosmos3 的 scaling/scheduler 接线，**不写新算法**。
- 数学参考实现：NVIDIA `~/Desktop/cosmos-rl/cosmos_rl/policy/trainer/wfm_trainer.py:464-490`（注意它是 **Predict2.5 的 WFM** 不是 Cosmos3，只借 logprob 公式）。⚠️ **移植坑**：cosmos-rl 把方差归一化 `/(2·std²)` 注释掉了（`:486` TODO），**用归一化的正确式，别照搬 bug**。

### 1.4 模型与规模

- HF：`nvidia/Cosmos3-Nano`(16B) / `Cosmos3-Super`(64B) 等。
- 15.17B bf16 ≈ 30GB 权重 → **单卡 32GB 训练 forward 装不下**（推理需 sequential CPU offload，RAM 够但慢）；**可信 RL 曲线必须多卡 FSDP2**（本仓库 online FSDP2 仍 gated）。generator RL 单卡只能 LoRA smoke。

### 1.5 MR0 历史实测与当前依赖状态

2026-06-27/28 曾用 diffusers main(0.39.0.dev0)做先行验证；当前仓库已使用正式版
0.39.0，因此下列兼容性结论不再是 dependency blocker：
- ✅ `Cosmos3OmniTransformer.from_config()` = 15.17B（只下 config，不下权重）。
- ✅ **diffusers 0.39 向后兼容**：现有 cosmos2/wan/predict2 + 已 ship 的 reasoner judge 在 0.39 下导入+测试全过。
- 🧱 **本机下载墙**：Cosmos3-Nano 16B 下到 6/7 分片（33GB）后，最后一个 transformer 分片被 HF xet/CDN 连接重置死死卡住（试遍 xet/非xet/单文件/hf_transfer 都冻在 ~88%）→ 本机加载不了、MR1+ 跑不了。换网络/换机器即可。

## 2. MR 拆分（生成器线）

```text
MR0 (probe) ✅
  → dependency + MR2 CPU implementation ✅
  → MR1 (pipeline real load/T2V)
  → MR2 (family/pipeline real parity)
  → MR3 (logprob recipe/parity)
  → MR4 (LoRA RL run)

audio/action = 非目标
```

### MR0 — 契约盘点（✅ 已完成 + run-verified）

- `vrl/scripts/eval/cosmos3_nano_generator_probe.py`（已内联 model id，跑出 decision note）。
- 实测：generator 从 config 实例化 15.17B、forward 签名读出、denoise 蓝本读出、pack 装配难点定位、diffusers 0.39 兼容性确认。见 §1.1-1.5。

### MR1 — 真实 checkpoint 加载与 T2V

- 依赖已完成，不 vendor 上游实现。
- **gate**：`Cosmos3OmniPipeline.from_pretrained(...)` 在多卡/offload 上 load（bf16 only），跑出一个非 RL 的 T2V clip。
- **blocker**：本机下载墙（§1.5）与目标硬件容量；不再包含 Python dependency availability。

### MR2 — `cosmos3` diffusion family（代码已落，真实 gate 待完成）

已在 `vrl/models/families/cosmos/cosmos3/` 落完整 CPU implementation，**包住 diffusers
的 `Cosmos3OmniPipeline`**：

- `model.py`：`Cosmos3Model` + `Cosmos3ReplayModel` + `Cosmos3SamplingState`。
  `from_build` 载 pipeline；`encode_prompt` 走 pipeline tokenization；`prepare_sampling`
  通过唯一 `_assemble_packed_static` 建 cond/uncond pack；`restore_eval_state` 复用同一
  helper；`forward_step` 每步 splice `vision_timesteps` + latents，返回 raw velocity、
  cond 与 uncond，CFG 与 logprob 留给共享 denoise loop。
- `runtime.py`：`Cosmos3ChunkExecutor` 复用
  `vrl/generation/bindings/full_sequence_denoise/`，并因 pipeline 契约固定 batch=1；
  `build_cosmos3_replay_runtime_bundle` 构造 trainer replay runtime。
- `vrl/families/registry.py` 是 family/executor/replay builder 的唯一 binding；
  `vrl/config/presets/model/cosmos/cosmos3_nano.yaml` 是当前唯一 Cosmos3 model preset，
  必须保留，不能在 preset sweep 中当作孤儿删除。
- **复用**：`DiffusionModelBase`、`vrl/generation/steps/denoise/loop.py`、
  full-sequence executor/gatherer、`vrl/math/denoise/flow_matching.py` 与
  `vrl/rollouts/evaluators/denoise/sde_logprob.py`。
- **剩余 gate**：family executor 出一个非 RL T2V clip，并与 MR1 pipeline 的相同
  checkpoint/seed/recipe 输出核对。此 gate 验证的是私有 upstream pack seam，不再新增
  第二个 assembly helper。

### MR3 — logprob 接线 + train recipe

- `forward_step` 返回的 velocity 已由
  `vrl/generation/steps/denoise/loop.py` 喂给
  `vrl/math/denoise/flow_matching.py:sde_step_with_logprob`；本 MR 验证 Cosmos3 的
  scaling/scheduler 与 replay 接线，不复制 math。
- 新增尚不存在的
  `vrl/config/presets/experiment/cosmos3/online_grpo_t2v.yaml`，并接到现有 train entry。
- **gate**：old-vs-new denoise logprob 复算一致（first-step diff≈0，同 Predict2 判据）。

### MR4 — LoRA RL run

- 16B → **LoRA 优先**（单卡 smoke）；可信曲线走多卡 FSDP2。
- 数据：机器人 per-sample reference（复用 `video_world_v2w`，见 data-factory sprint）。reward：已 ship 的 Cosmos3 reasoner-judge 或 Kling 过渡。
- **gate**：clip_fraction>0、first-step logprob diff≈0、生成 artifact **肉眼**连贯（480p 教训：neighbor-diff 统计骗人，必须看）、eval reward >2σ 才算"学到"。

### 非目标

audio / action 塔不做；action 走 `SPRINT_physical_ai_model_support.md` 的 VLA/Env seam。Super 64B 只 track。不在单卡硬上 full-param 16B。

## 3. 工程落点

| MR | 路径 | 复用/参考 |
|---|---|---|
| dependency | `pyproject.toml` / `uv.lock`（✅ 已完成） | diffusers 0.39 public release |
| MR1/MR2 | `vrl/models/families/cosmos/cosmos3/`、`vrl/families/registry.py`、`vrl/config/presets/model/cosmos/cosmos3_nano.yaml` | `vrl/models/families/cosmos/predict2/`、diffusers `Cosmos3OmniPipeline` |
| MR3 | `vrl/config/presets/experiment/cosmos3/online_grpo_t2v.yaml`、现有 train entry | `vrl/math/denoise/flow_matching.py`、`vrl/generation/steps/denoise/loop.py`、`vrl/rollouts/evaluators/denoise/sde_logprob.py` |
| MR4 | experiment config | data-factory `video_world_v2w` |

**保持不变**：Predict2/2.5/Wan/Echo diffusion seam 不动；executor SDE-logprob 循环不动。

## 4. 验收

- **MR1**：pipeline 在目标机器上 load + 出 T2V clip。
- **MR2**：cosmos3 family executor 出 T2V clip（pack 装配正确）。
- **MR3**：logprob 复算一致（first-step diff≈0）→ generator RL-eligible。
- **MR4**：clip_fraction>0 + 生成连贯 + eval reward >2σ（多卡）。

## 5. 风险

| 风险 | 处理 |
|---|---|
| diffusers dependency 漂移到不含 Cosmos3 的版本 | `cosmos` extra 下限钉为 0.39，并由 clean resolve 验证 |
| pack 装配（§1.2）与 upstream 私有方法漂移 | 保持仓库唯一 `_assemble_packed_static`，用同 checkpoint/seed/recipe 做 pipeline/family parity；不再写第二份 helper |
| 16B 单卡装不下 | LoRA smoke 先行；可信曲线 gated 多卡 FSDP2 |
| 本机下载墙 | 换网络/换目标多卡机器下载 |
| 照搬 cosmos-rl 未归一化 logprob bug | 用 §1.3 归一化式 + first-step diff≈0 验 |

## 6. 参考

- diffusers 0.39 生成器源：`pipelines/cosmos/pipeline_cosmos3_omni.py`、`models/transformers/transformer_cosmos3.py`
- logprob 数学参考：`~/Desktop/cosmos-rl/cosmos_rl/policy/trainer/wfm_trainer.py:464-490`（Predict2.5 WFM，只借公式）
- 本仓库复用：`vrl/math/denoise/flow_matching.py`、
  `vrl/generation/steps/denoise/loop.py`、
  `vrl/rollouts/evaluators/denoise/sde_logprob.py`、
  `vrl/models/families/cosmos/predict2/`、`vrl/families/registry.py`
- 探针：`vrl/scripts/eval/cosmos3_nano_generator_probe.py`
- 承接/下游：`docs/sprints/done/SPRINT_physical_ai_model_support.md`、
  `docs/sprints/done/SPRINT_cosmos_robotic_data_factory_domain_rl.md`（reward + 数据）、
  `docs/sprints/done/SPRINT_multi_gpu_training.md`（16B 多卡）
- 模型：`nvidia/Cosmos3-Nano`(16B)，HF collection https://huggingface.co/collections/nvidia/cosmos3
