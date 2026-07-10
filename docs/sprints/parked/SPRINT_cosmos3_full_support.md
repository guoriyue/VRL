# SPRINT: Cosmos3 Vision Generator — RL via the diffusion seam

状态：**parked / run-verify-gated（2026-07-09 复核）**。触发条件：有多卡、
网络可用且能安装含 `Cosmos3OmniDiffusersPipeline` 的受控 diffusers 版本。范围：**只做 Cosmos3 的 vision 生成器**（`vrl/models/diffusion/cosmos/cosmos3/`），让它能进本仓库的 diffusion seam 被 RL 训练。本仓先落 registry/family skeleton + 已验证契约；权重加载和 RL run-verify 需要有多卡 + 网络通的机器。

> **reasoner-judge 已单独 ship**（`reward: add cosmos3 reasoner judge`）：`vrl/rewards/models/cosmos3_reasoner.py` + config + 注册 + 测试。它是 VLM 裁判（视频→分数），属 reward seam，**不在本 sprint**。本 sprint 只管"生成视频"的那半。

## 0. 一句话

Cosmos3（`model_type="cosmos3_omni"`）的生成器是一个 **diffusion MoT**（Mixture-of-Transformers）。它**已经是 diffusers 格式、可加载**——之前"被 diffusers 永久卡死"是错的，真相是**唯一硬阻塞 = diffusers 版本**（类在 git-main，不在装的 0.37.1），且**不需要 NVIDIA 原生 cosmos-rl 代码**。要支持它 = 新建一个 `cosmos3` diffusion family（包住 diffusers 的 `Cosmos3OmniDiffusersPipeline`）+ 复用现有 `flow_grpo` 算 logprob。

## 0.1 Readiness verdict（2026-06-28）

**不能移到 `done/`。** 这个 sprint 的结论是"已落 skeleton，未过 run-verify gate"。

已完成：
- reasoner-judge 已单独提交并推送；它属于 reward seam，不属于本生成器 sprint 的 done gate。
- MR0 contract/probe 已跑出明确结论：Cosmos3-Nano 是 diffusers-format MoT，当前仓库环境缺少 git-main 的 `Cosmos3OmniDiffusersPipeline`。
- `cosmos3` registry/family skeleton 已提交并通过 CPU/registry/config 级验证。

未完成的 done gate：
- MR1：需要 diffusers git-main/vendor 后 `Cosmos3OmniDiffusersPipeline.from_pretrained(...)` 真实 load，并产出 T2V clip。
- MR2：需要 family executor 真产出 T2V clip，证明 `packed_static` 装配正确。
- MR3：需要 first-step logprob diff≈0，证明 generator RL-eligible。
- MR4：需要 LoRA RL run 满足 clip_fraction>0、artifact 连贯、eval reward >2σ。

因此它留在 `docs/sprints/parked/`，直到上述硬件/依赖事件发生；目前只有 MR0 和 MR2 skeleton 已经落地。

## 1. 权威事实（已读源 / 已实测，非推断）

### 1.1 生成器 = diffusers 格式的 MoT（实测确认）

- `nvidia/Cosmos3-Nano` 的 checkpoint **本就是 diffusers 格式**：`transformer/`(7 分片) + `vae/`(Wan2.2-TI2V, z_dim=48) + `scheduler/`(UniPCMultistep, flow_shift=10) + `model_index.json`(pipeline=`Cosmos3OmniDiffusersPipeline`)，还附 `example_t2v_diffusers_output.mp4`。
- 生成器类 `Cosmos3OmniTransformer` + pipeline `Cosmos3OmniDiffusersPipeline` **在 diffusers git-main**（实测 `from_config()` 实例化 = **15.17B 参数**），**不在装的 0.37.1**。**不能**塞进旧的 `CosmosTransformer3DModel`（config `model_type=qwen3_vl_text`/`use_moe`/`unified_3d_mrope` 不兼容）。
- 架构 = **MoT，不是路由 MoE**：一个 transformer，Qwen3-VL 因果 reasoner 流 + 双向 diffusion 生成流，共享 joint attention 但每层**分开的参数集**；`_moe_gen` key = 生成流权重（`use_moe=true` 选它）。Nano 16B = 8B reasoner + 8B generator。flow-matching DiT，无独立 text encoder。

### 1.2 forward 不是干净的扩散 DiT（MR2 真实难点）

实测 `Cosmos3OmniTransformer.forward`：
```
forward(input_ids, text_indexes, position_ids, und_len, sequence_length,
        vision_tokens[], vision_sequence_indexes, vision_timesteps, vision_noisy_frame_indexes,
        sound_*, action_*) -> (preds_vision, preds_sound, preds_action)
```
text/vision/sound/action 交错成一个联合序列。denoise 蓝本（`pipeline_cosmos3_omni.py:1618-1744`）：每步 `vision_timesteps=full(timestep)` + splice `vision_tokens=[latents]` → `transformer(**cond_pack,...)` 得 `preds_vision`(velocity) → `_mask_velocity_predictions` → CFG 合成 → `scheduler.step`。
- ⚠️ **核心工作量**：cond/uncond 的 `packed_static`（text/vision segment + MRoPE `position_ids` 拼接 + `vision_sequence_indexes` 等）是**写在 pipeline `__call__` 内联的 ~100 行，不是可复用方法**。MR2 要么把它重构成 `_assemble_packed_static` 复用，要么复刻（bug 面大）——**必须 run-verify，不能 import-only 当完成**。

### 1.3 logprob = 复用现有 flow_grpo，不是新算法

每个 denoise step 是高斯转移 `x_{t-1}=μ(x_t,σ)+std·ε`，policy-gradient 的 ratio = 当前策略均值 μ_θ vs 参考均值 μ_ref 的高斯似然比：
```text
log_prob_ratio = -( ||x_sample-μ_θ||² - ||x_sample-μ_ref||² ) / (2·σ_next²·(η²+2η))
```
- **这就是仓库已有的 `vrl/algorithms/flow_matching.sde_step_with_logprob`（DanceGRPO/Flow-GRPO 同构）**，而且**是 executor（不是 model）在调它**；Cosmos3 是 flow-matching 同族，executor 在 collect 时已自动用带 logprob 的 SDE 替换确定性 UniPC。→ MR3 只把 Cosmos 的 scaling/solver 喂进去，**不写新算法**。
- 数学参考实现：NVIDIA `~/Desktop/cosmos-rl/cosmos_rl/policy/trainer/wfm_trainer.py:464-490`（注意它是 **Predict2.5 的 WFM** 不是 Cosmos3，只借 logprob 公式）。⚠️ **移植坑**：cosmos-rl 把方差归一化 `/(2·std²)` 注释掉了（`:486` TODO），**用归一化的正确式，别照搬 bug**。

### 1.4 模型与规模

- HF：`nvidia/Cosmos3-Nano`(16B) / `Cosmos3-Super`(64B) 等。
- 15.17B bf16 ≈ 30GB 权重 → **单卡 32GB 训练 forward 装不下**（推理需 sequential CPU offload，RAM 够但慢）；**可信 RL 曲线必须多卡 FSDP2**（本仓库 online FSDP2 仍 gated）。generator RL 单卡只能 LoRA smoke。

### 1.5 MR0 实测确认（diffusers@main, 2026-06-27/28）

装 diffusers@main(0.39.0.dev0)到一次性 venv（复用基础 torch 2.11/transformers 4.57.6）实测：
- ✅ `Cosmos3OmniTransformer.from_config()` = 15.17B（只下 config，不下权重）。
- ✅ **diffusers@main 向后兼容**：现有 cosmos2/wan/predict2 + 已 ship 的 reasoner judge 在 0.39 下导入+测试全过；唯一报错（`tests/models/interfaces` echo 注册）在 0.37.1 下一模一样，是预先存在缺口 → **升级 diffusers 安全**。
- 🧱 **本机下载墙**：Cosmos3-Nano 16B 下到 6/7 分片（33GB）后，最后一个 transformer 分片被 HF xet/CDN 连接重置死死卡住（试遍 xet/非xet/单文件/hf_transfer 都冻在 ~88%）→ 本机加载不了、MR1+ 跑不了。换网络/换机器即可。

## 2. MR 拆分（生成器线）

```text
MR0 (probe) ✅done → MR1 (bump diffusers + load) → MR2 (cosmos3 family) → MR3 (logprob recipe) → MR4 (LoRA RL run)
                                                                                  audio/action = 非目标
```

### MR0 — 契约盘点（✅ 已完成 + run-verified）

- `vrl/scripts/eval/cosmos3_nano_generator_probe.py`（已内联 model id，跑出 decision note）。
- 实测：generator 从 config 实例化 15.17B、forward 签名读出、denoise 蓝本读出、pack 装配难点定位、diffusers@main 兼容性确认。见 §1.1-1.5。

### MR1 — 升 diffusers + 加载生成器

- `pyproject.toml` 把 diffusers pin 到含 Cosmos3 的 git-main commit（或 vendor `transformer_cosmos3.py` + `pipeline_cosmos3_omni.py` 两个模块，避免全量升级）。
- **gate**：`Cosmos3OmniDiffusersPipeline.from_pretrained(...)` 在多卡/offload 上 load（bf16 only），跑出一个非 RL 的 T2V clip。
- **blocker**：①本机下载墙（§1.5）→ 换机器；②git-main pin 与仓库 pin 的 transformers/torch 兼容性 → 按"verify against declared deps"在 clean install 上验。

### MR2 — `cosmos3` diffusion family（核心工作量）

新建 `vrl/models/diffusion/cosmos/cosmos3/{model,runner,runtime}.py`，**包住 diffusers 的 `Cosmos3OmniDiffusersPipeline`**（像 predict2 包 `Cosmos2VideoToWorldPipeline`）：
- `model.py`：`Cosmos3Model(DiffusionModelBase)` + `Cosmos3SamplingState`。`from_spec` 载 pipeline；`encode_prompt` 走 pipeline 的 tokenization 出 input_ids；`prepare_sampling` 建 **cond/uncond `packed_static`**（§1.2 的难点：把 pipeline `__call__` 内联的 pack 装配抽出来复用）；`forward_step` 每步 splice `vision_timesteps`+latents → `transformer(**pack)` → `preds_vision`(velocity)，返回 `{noise_pred, noise_pred_cond, noise_pred_uncond}`，CFG+logprob 交给 executor。
- `register_rollout_family(family="cosmos3", diffusion 分支)`（`vrl/rollouts/families/registry.py`）+ `configs/model/diffusion/cosmos/cosmos3_nano.yaml`。
- **复用**：`DiffusionModelBase`、executor SDE-logprob 循环（`vrl/generation/diffusion/executor.py`）、loader、gatherer、CFG caller。
- **gate**：family executor 出一个非 RL T2V clip（与 MR1 pipeline 输出一致）。

### MR3 — logprob 接线 + train recipe

- `forward_step` 返回的 velocity 喂 `vrl/algorithms/flow_matching.sde_step_with_logprob`（§1.3，executor 已在调）；配好 Cosmos3 的 scaling/solver（UniPC flow_shift=10）。
- `configs/experiment/diffusion/cosmos3/online_grpo_t2v.yaml` + train entry。
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
| MR1 | `pyproject.toml`(diffusers pin / vendor 模块) | diffusers `transformer_cosmos3.py`/`pipeline_cosmos3_omni.py` |
| MR2 | `vrl/models/diffusion/cosmos/cosmos3/{model,runner,runtime}.py`、registry、`configs/model/diffusion/cosmos/cosmos3_nano.yaml` | `vrl/models/diffusion/cosmos/predict2/*`、diffusers `Cosmos3OmniDiffusersPipeline` |
| MR3 | `configs/experiment/diffusion/cosmos3/*`、train entry | `vrl/algorithms/flow_matching`、`vrl/generation/diffusion/executor.py` |
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
| diffusers git-main pin 破坏现有 pin 的 transformers/torch | clean-install 验；或 vendor 两个模块而非全量升级 |
| pack 装配（§1.2）复刻出错 | 优先把 pipeline `__call__` 重构出 `_assemble_packed_static` 复用，别手抄；run-verify |
| 16B 单卡装不下 | LoRA smoke 先行；可信曲线 gated 多卡 FSDP2 |
| 本机下载墙 | 换网络/换目标多卡机器下载 |
| 照搬 cosmos-rl 未归一化 logprob bug | 用 §1.3 归一化式 + first-step diff≈0 验 |

## 6. 参考

- diffusers 生成器源（git-main）：`pipelines/cosmos/pipeline_cosmos3_omni.py`、`models/transformers/transformer_cosmos3.py`
- logprob 数学参考：`~/Desktop/cosmos-rl/cosmos_rl/policy/trainer/wfm_trainer.py:464-490`（Predict2.5 WFM，只借公式）
- 本仓库复用：`vrl/algorithms/flow_matching`、`vrl/models/diffusion/cosmos/predict2/*`、`vrl/generation/diffusion/executor.py`、`vrl/rollouts/families/registry.py`
- 探针：`vrl/scripts/eval/cosmos3_nano_generator_probe.py`
- 承接/下游：`SPRINT_physical_ai_model_support.md`、`SPRINT_cosmos_robotic_data_factory_domain_rl.md`(reward + 数据)、`SPRINT_multi_gpu_training.md`(16B 多卡)
- 模型：`nvidia/Cosmos3-Nano`(16B)，HF collection https://huggingface.co/collections/nvidia/cosmos3
