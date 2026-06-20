# SPRINT: algorithm-shaped rollout payload + loss-validates-its-inputs 契约（done）

状态：**核心交付完成 → done**。声明 + 校验契约 DONE（`required_data_keys` + `validate_inputs` 已落地，代码佐证见「实现状态」段）；唯一 deferred 的「携带集 generation 侧物理裁剪」因架构解耦受限、明确划为独立后续 sprint。**声明 + 校验契约 DONE / 携带集 generation 侧物理裁剪 deferred**（2026-06-20，分支 `spike/vllm-omni-rollout`）。让"rollout 携带哪些张量"由 **algorithm** 选择（NFT 只带 `latents_clean`，flow-GRPO 带完整 SDE trajectory），并补一个 verl-omni 有、vrl 缺的 `required_data_keys` fail-fast 契约。**非重写**：vrl 已大半到位（per-family capability + per-segment ReplayInput + `uses_evaluator` 分叉），本 sprint 只把"携带张量的选择键"从 family 收紧到 algorithm，再加一层声明式入参校验。

## 实现状态（2026-06-20）

**已实现（§1 声明 + §2 校验契约，完整落地）：**
- `vrl/algorithms/base.py` — `Algorithm` Protocol 新增 `required_signal_keys` / `required_data_keys`（默认空 tuple，算法显式声明才参与校验），与 `uses_evaluator`/`tolerates_off_policy_staleness` 同位（Protocol 默认值被各算法子类继承）。
- `vrl/algorithms/grpo/continuous.py` — `GRPO.required_signal_keys = ("log_prob","old_log_prob")`（TokenGRPO / MultiSegmentTokenGRPO 继承）。`ref_log_prob` 仍由 KL 分支按 `init_kl_coef>0` **条件**校验，不进硬契约（保留其详尽报错，见 non-goal）。
- `vrl/algorithms/diffusion_nft.py` — `DiffusionNFT.required_data_keys = ("latents_clean","prompt_embeds","timesteps")`；删除内联逐键 `isinstance` 检查（原 :152-160），改为直接读取（presence/type 由中心 gate 保证）。
- `vrl/algorithms/trajectory.py` — 新增 `AlgorithmAdapter.validate_inputs`：单一声明式 gate，signal 分支查 `SegmentSignal` 非 None 字段、replay 分支查 `replay_tensor_dict()` 的 tensor key，缺则 raise 并打印 **missing + available**（照搬 verl-omni `_format_available_keys` 语义）；在 `compute_loss` 入口调用。吃掉 GRPO/TokenGRPO/MultiSegment 的 `signals is None` 内联 guard 与 NFT 的逐键 raise。
- 保留（non-goal）：token-GRPO `ref_log_prob is None` 详尽报错、`log_prob/old_log_prob` shape-mismatch、multisegment missing-segment 报错。
- 测试：新增 `tests/algorithms/test_input_contract.py`（signal 缺 `old_log_prob`、replay 缺 `latents_clean` 各断言 missing+available；happy-path 静默）。`tests/algorithms/` **67 passed**。

**Deferred（§1 第 3 点"携带集由 algorithm 选择"的 generation 侧物理裁剪）—— 架构所限，据实记录：**
- 验收 #3「NFT rollout payload 不再携带 SDE per-step `observations/log_prob`」需要 generation 路径知道算法。但 `vrl/generation/diffusion/gather.py:44-109` **无条件**把完整 SDE 超集（observations/actions/log_probs/kl）拼进 trajectory，`build_diffusion_trajectory` 处于 generation 路径，且 `GenerationRequest`（`vrl/generation/types.py:35-47`）不携带 recipe/algorithm 名——generation 路径全程**零算法引用**（grep 实证）。
- 让 builder 读 `required_*_keys` 做物理裁剪，必须把算法穿过 rollout↔train 边界（改 `GenerationRequest` + executor），这与本 sprint 自己的 non-goal「不动 planner 的 chunk/stage 逻辑、不碰 SDE evaluator replay 路径」**直接冲突**，也违背 generation 与 algorithm 解耦的既有设计。
- 现状：`required_data_keys` 已是该"携带集"的**权威声明**，且由 `validate_inputs` 强制；物理裁剪应作为后续独立 sprint（先决定是否把 algorithm/recipe 注入 `GenerationRequest`）。当前隐式链**功能正确**，裁剪仅为省去未用 per-step 张量的 CPU↔object-store 搬运（本 sprint §3.4 已自评为低优先）。

关联：
- [[SPRINT_nft_invariant]]（`done/`，NFT 的 advantage-flip 不变量；本 sprint 的校验契约与它互补：一个查"数值是否中立"，一个查"输入是否齐全"）
- [[SPRINT_cosmos_kling_fixed_eval_signal]]（`done/`，固定 eval 信号；同属 trajectory/signal 链路）
- [[SPRINT_design_smell_loose_ends]]（架构卫生尾巴的同类低优先收口范式）

## 0. Core Decision（先看这一段）

**结论：把 rollout payload 的"携带哪些张量"这件事的选择键，从 `FamilyCapability`（family/task）下沉为 algorithm 选择，并补一个声明式 `required_data_keys` 校验契约——但这是结构性、低优先项，不是 bug。** vrl 今天的 `EnginePlan` 信封（`vrl/generation/execution/planner.py:99-111` 的 `sample_rows/expected_axes/chunks/execution_stages`）完全 keyed on `FamilyCapability`，而"训练真正读哪些张量"已经在两个地方按 algorithm 分叉了：NFT 走 `uses_evaluator=False` 的 replay-tensor 分支只读 `latents_clean/prompt_embeds/timesteps`，flow/token-GRPO 走 evaluator 分支读 `signals.primary.log_prob/old_log_prob/ref_log_prob`（完整 SDE 轨迹）。verl-omni 已经把这件事做成显式的 per-algorithm `DiffusionOutput.custom_output` 填充 + loss 侧 `required_data_keys` fail-fast；vrl 的等价物是**隐式的**——payload schema 由 per-segment `ReplayInput.tensor_refs`（family/builder 决定）+ `SignalRequest`（trainer 按 `init_kl_coef` 决定）拼出来，algorithm 自己**没有**一处声明"我需要哪些 key"，缺 key 时只能靠 NFT 内联的 `isinstance(value, torch.Tensor)` 逐键 raise（`diffusion_nft.py:152-160`）兜底，没有 available-vs-missing 的诊断。本 sprint 把这条隐式链收成显式契约：(1) algorithm 声明 `required_data_keys`/`required_signal_keys`；(2) payload builder/trainer 按 algorithm 的声明决定携带集；(3) loss 入口统一 fail-fast 并打印 available vs missing。

## 1. 两个 algorithm 已经携带不同张量 —— 但选择键是 family 不是 algorithm

**verl-omni 现在**：payload schema 由 algorithm-specific rollout adapter **显式**填充。`DiffusionOutput` 是统一信封 = `tensor + optional log_probs + extra_fields`：

```python
class DiffusionOutput(BaseModel):
    diffusion_output: Any
    log_probs: Optional[Any] = None
    extra_fields: dict[str, Any] = {}
```
（`verl_omni/workers/rollout/replica.py:20-32`）

NFT 的 adapter 只往 `custom_output` 塞 forward-process 所需的 clean latent + timesteps + prompt embeds，**不收 reverse-SDE 轨迹也不收 log_probs**：

```python
return DiffusionOutput(
    output=decoded.output,
    custom_output={
        "latents_clean": latents_clean,
        "train_timesteps": ...,
        "prompt_embeds": ctx["prompt_embeds"],
        ...
    },
    to_cpu=True,
)
```
（`verl_omni/pipelines/qwen_image_diffusion_nft/vllm_omni_rollout_adapter.py:258-269`，类 docstring 直说 "the rollout side does not collect reverse-SDE trajectories or log-probabilities" :38-41）

flow-GRPO 的 adapter 则跑完整 SDE loop，逐步收 `all_latents / all_log_probs / all_timesteps`（`verl_omni/pipelines/sd3_flow_grpo/vllm_omni_rollout_adapter.py:212-216,265-279`）。**同一个 `DiffusionOutput` 信封，algorithm 决定填什么。**

**vrl 现状**：携带张量已按 algorithm 分叉，但**选择键是 family**。`EnginePlan` 是 per-request immutable 信封，字段全部由 `FamilyCapability` 推出：

```python
return EnginePlan(
    ...
    capability=self.capability,
    trajectory_kind=self.capability.trajectory_kind,
    expected_axes=resolved_axes,
    chunks=chunk_schedule.chunks,
    execution_stages=execution_stages,
    ...
)
```
（`vrl/generation/execution/planner.py:223-237`；`capability` 来自 family/task，见 `FamilyCapability` 定义 `vrl/generation/capabilities.py:116-133`）

而训练侧真正读的张量已经按 algorithm 不同：
- NFT：`replay_tensor_dict("denoise")` 取 `latents_clean/prompt_embeds/timesteps`（`vrl/algorithms/diffusion_nft.py:151-163`），且 `uses_evaluator = False`（:37）。
- token/flow-GRPO：`signals = inputs.signals.primary`，读 `log_prob/old_log_prob/ref_log_prob`（`vrl/algorithms/grpo/token.py:41-44,80`）——这是完整 SDE 轨迹经 evaluator replay 出来的信号。

**差距**：携带集今天是 `ReplayInput.tensor_refs`（builder 在 `vrl/trajectory/builders.py:88-93,114-119` 按 family 拼，把所有 sample-aligned 的 replay 张量都塞进 `denoise` 段）+ `SignalRequest`（trainer 按 `init_kl_coef` 决定 `need_ref`，`vrl/trainers/online/trainer.py:675-678`）共同决定。algorithm 自己不声明携带集——它只在 loss 里**事后**发现 key 在不在。即 verl-omni 是"algorithm 选 schema → 填 → loss 校验"，vrl 是"family 填一个超集 → algorithm 事后挑/事后报错"。

## 2. loss-validates-its-inputs：verl-omni 有声明式契约，vrl 只有内联逐键 raise

**verl-omni 现在**：每个 loss 类**声明** `required_model_output_keys` / `required_data_keys`，基类 `validate_inputs` 统一 fail-fast 并打印 **available vs missing**：

```python
missing_model_output = [key for key in self.required_model_output_keys if key not in model_output]
missing_data = [key for key in self.required_data_keys if key not in data]
if not missing_model_output and not missing_data:
    return
details = [f"Diffusion loss `{loss_name}` is missing required inputs."]
if missing_data:
    details.append(f"Missing data keys: {missing_data}.")
    details.append(f"Available data keys: {_format_available_keys(data)}.")
raise KeyError(" ".join(details))
```
（`verl_omni/trainer/diffusion/diffusion_algos.py:63-84`）

声明随 algorithm 不同：FlowGRPO = `required_model_output_keys=("log_probs",)`, `required_data_keys=("old_log_probs","advantages")`（:273-274）；FlowDPPO 还多要 `("prev_sample_mean","std_dev_t","sqrt_dt")` + `old_prev_sample_mean`（:365-366）；另有 `("ref_noise_pred","sample_level_rewards")`、`("reward_prob",)`、`("ref_prev_sample_mean",)` 等（:595,793,1018）。这正是"loss 把自己的入参契约钉死在自己身上"。

**vrl 现状**：**没有**这层声明式契约。最接近的是 NFT 在 loss 里内联逐键检查，错误信息只说缺哪个 key、**不列 available**：

```python
for key in ("latents_clean", "prompt_embeds", "timesteps"):
    value = replay_tensors.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"DiffusionNFT requires trajectory replay tensor {key!r}; "
            f"got {type(value).__name__}",
        )
```
（`vrl/algorithms/diffusion_nft.py:152-160`）

token-GRPO 同样是内联的 `if inputs.signals is None: raise`（`vrl/algorithms/grpo/token.py:37-40`）、`if signals.ref_log_prob is None: raise`（:71-79，错误信息很详尽但仍是 per-algorithm 手写的）。`AlgorithmInput`（`vrl/algorithms/trajectory.py:13-26`）和 `Algorithm` Protocol（`vrl/algorithms/base.py:13-34`）都**没有** `required_data_keys` 字段——所以每个 algorithm 重复手写自己的缺键检查，错误格式不统一、且不打 available 集，调试时看不到"我到底拿到了哪些 key"。

## 3. vrl 已经大半到位 —— 本 sprint 的真实 scope

vrl 不需要重写。它已有的、与 verl-omni 同构的部件：
- **统一信封**：`EnginePlan`（planner.py:99-111）≈ `DiffusionOutput`；`TrajectoryBatch`/`SegmentSignal`（`vrl/rollouts/evaluators/types.py:9-27`）已是 first-class schema，`SegmentSignal` 的 `prev_sample_mean/ref_prev_sample_mean/std_dev_t/dt` 等 optional 字段恰好覆盖了 verl-omni FlowDPPO 那批 key。
- **algorithm 已能声明行为标志**：`uses_evaluator`（`diffusion_nft.py:37`，trainer 在 :622 读它分叉 payload 路径）、`tolerates_off_policy_staleness`（:46，trainer 在 :370-371 读它喂给 rollout schedule）。**新增 `required_data_keys`/`required_signal_keys` 与它们同位**，是同一类"algorithm 自描述"。
- **携带集已有声明点**：`ReplayInput.tensor_refs`（`vrl/trajectory/types.py:83-94`）已是显式的"这个 segment 携带哪些张量"列表，只是今天由 family-builder 填超集。

**改造（最小集）**：
1. 在 `Algorithm` Protocol（`base.py`）/ `AlgorithmInput`（`trajectory.py`）上加 `required_data_keys: tuple[str,...]`（replay 段）+ `required_signal_keys: tuple[str,...]`（SegmentSignal 字段），NFT 与各 GRPO 各自声明（值与 §2 列出的一致）。
2. 在 `AlgorithmAdapter.compute_loss`（`trajectory.py:44-63`）入口加一个 `validate_inputs`——对 replay 分支查 `replay_tensor_dict` 的 keys、对 signal 分支查 `SegmentSignal` 的非 None 字段，缺则 raise 并打印 **available vs missing**（照搬 verl-omni `_format_available_keys` 的格式）。这吃掉 NFT 内联检查（:152-160）和 token-GRPO 的 `signals is None` 检查（:37-40），错误格式统一。
3. **携带集由 algorithm 选择**：让 trainer 在构造 `SignalRequest`（trainer.py:675-678）和 builder 在拼 `ReplayInput.tensor_refs`（builders.py:88-93）时，读 algorithm 的 `required_*_keys` 作为**携带的下界/上界**——NFT 不再被动接收 family 塞进来的 SDE 轨迹超集，flow-GRPO 显式声明它要完整轨迹。这是把"family 填超集"收成"algorithm 点单"。

**为何低优先**：今天的隐式链**功能上是对的**（NFT 跑得通、GRPO 跑得通），缺的是(a)缺键时的诊断质量、(b)携带超集带来的潜在浪费（NFT 段里若被塞了用不到的 per-step 张量，是 CPU↔object-store 的无谓搬运）。这是结构性收敛与可调试性收益，不是 correctness bug，故排在 fp8/memory/rollout 优化之后。

## 4. 验收

- 单测（CPU 即可，参 [[SPRINT_nft_invariant]] 的 CPU 不变量测法）：构造一个缺 `old_log_probs` 的 GRPO `AlgorithmInput` 与缺 `latents_clean` 的 NFT batch，断言 `validate_inputs` raise 且消息**同时**含 missing 列表与 available 列表。
- 回归：NFT 内联逐键 raise（:152-160）删除后，原有 `tests/algorithms/` 全绿；`uses_evaluator`/`tolerates_off_policy_staleness` 两条既有 algorithm-自描述路径行为不变。
- 携带集：断言 NFT 的 rollout payload 不再携带 SDE per-step `log_prob/observations`（只带 `required_data_keys` 声明的集合），flow-GRPO 携带集不变。

## Non-Goals

- **不重写 `EnginePlan` / `FamilyCapability`**。family capability 仍是 trajectory_kind/axes/execution-stage 的真相源；本 sprint 只在其上叠一层 algorithm-keyed 的"携带哪些张量"，不动 planner 的 chunk/stage 逻辑（planner.py:281-374）。
- **不改 NFT/GRPO 的数学**。`normalized_mse`、advantage-flip 不变量（diffusion_nft.py:67-112）、TIS 漂移校正（grpo/token.py:60）一律不动——只动入参的**声明与校验**，不动 loss 计算。
- **不引入新的单类瘦文件**。`required_data_keys` 落在 `base.py`/`trajectory.py` 既有文件，`validate_inputs` 落在 `AlgorithmAdapter` 既有类（遵循 MEMORY 的 no-new-lean-files / no-single-caller-helpers）。
- **不碰 SDE evaluator replay 路径**（`vrl/rollouts/evaluators/diffusion/sde_logprob.py`）的内部计算；只读它产出的 `SegmentSignal` 字段做校验。
- 不顺手统一 token-GRPO 那条已经很详尽的 `ref_log_prob is None` 报错信息（:71-79）的措辞——它已经可用，归入统一契约后保留其诊断细节即可，非本 sprint 的 churn 目标。

## References

阅读依据（reading doc / 对照算法文档）：
- verl-omni §3.2（algorithm-shaped `DiffusionOutput` payload）+ §3.6-tail（loss-validates-its-inputs 契约）——对应代码与算法文档：
  - `/home/mingfeiguo/Desktop/verl-omni/docs/algo/diffusionnft.md`
  - `/home/mingfeiguo/Desktop/verl-omni/docs/algo/flowgrpo.md`
  - `/home/mingfeiguo/Desktop/verl-omni/docs/contributing/integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md:168-169`（contributing 指南把 `required_model_output_keys/required_data_keys` 列为新算法接入契约）

verl-omni 代码（本次实读）：
- `/home/mingfeiguo/Desktop/verl-omni/verl_omni/workers/rollout/replica.py:20-32`（`DiffusionOutput` 统一信封）
- `/home/mingfeiguo/Desktop/verl-omni/verl_omni/pipelines/qwen_image_diffusion_nft/vllm_omni_rollout_adapter.py:38-41,258-269`（NFT 只填 clean latent，无轨迹/无 log_probs）
- `/home/mingfeiguo/Desktop/verl-omni/verl_omni/pipelines/sd3_flow_grpo/vllm_omni_rollout_adapter.py:212-216,265-279`（flow-GRPO 收完整 SDE 轨迹 + per-step log_probs）
- `/home/mingfeiguo/Desktop/verl-omni/verl_omni/trainer/diffusion/diffusion_algos.py:42-84`（`validate_inputs` + available-vs-missing 诊断）、:273-274,365-366,595,793,1018（各 loss 的 `required_data_keys` 声明）

vrl 代码（本次实读，改造目标）：
- `/home/mingfeiguo/Desktop/wm-infra/vrl/generation/execution/planner.py:99-111,223-237,281-374`（`EnginePlan` family-keyed 信封）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/generation/capabilities.py:116-133`（`FamilyCapability` = 携带集的当前选择键）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/diffusion_nft.py:37,46,151-163`（NFT 只读 `latents_clean/prompt_embeds/timesteps`；内联逐键 raise）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/token.py:37-44,71-79,80`（GRPO 读完整轨迹信号；内联缺键检查）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/trajectory.py:13-26,44-63`（`AlgorithmInput` + `AlgorithmAdapter.compute_loss` 入口，校验落点）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/base.py:13-34`（`Algorithm` Protocol，`required_*_keys` 新声明落点）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/evaluators/types.py:9-27,84-90`（`SegmentSignal` 字段 + `SignalRequest`，signal 侧携带集）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trajectory/types.py:83-94`、`/home/mingfeiguo/Desktop/wm-infra/vrl/trajectory/builders.py:88-93,114-119`（`ReplayInput.tensor_refs`，replay 侧携带集声明点）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/trainer.py:370-371,622,646-693,675-678`（`uses_evaluator`/`tolerates_off_policy_staleness` 既有 algorithm-自描述分叉；`SignalRequest` 构造点）
