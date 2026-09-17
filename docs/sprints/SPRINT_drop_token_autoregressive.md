# SPRINT：主线只保留 diffusion，移除 token-autoregressive 家族

状态：**done（2026-09-16）**，在 `review/combined-20260916` 上按四个提交落地（见文末「结果」）。

## 决定与边界

- 移除的是 **token-AR**（`generation_regime == "token_autoregressive"`）：registry 里 6 个 entry、
  5 个实现目录（janus_pro 与 janus_pro_r1 共用一个目录）：janus_pro / janus_pro_r1、llamagen、emu3、
  glm_image、nextstep_1。
- **保留** chunk-autoregressive denoise（causvid、magi_1）：它们是 denoise step，不依赖 token 代码
  （`grep token_autoregressive|steps.token|ARDiscrete` 在两者目录下为空）。
- **保留** vLLM 的两个 diffusion 侧消费者：`vrl/models/parking.py` 的 CuMemAllocator（共享 GPU
  的 sleep/wake）和 `vrl/nn/quantization/fp8.py`。`ar-vllm` extra 因此保留，只改注释；删除的是
  它的 token 专属消费者（paged attention 内核与层）。
- 依据：7 类证据里 diffusion 与 token-AR 的比例（家族 21:6、实验 preset 约 47:8、训练曲线记录
  多条:0、本地 outputs 多个:0）；`Janus-R1` 唯一的 AR 计划自 7 月起未动；miles_diffusion 不做 AR。
  "本地没有 AR outputs"只说明当前没有可查验的持续实验，不证明历史上从未跑过。

## 归档

- tag `archive/token-ar-20260916` → `5f1362c6`（review/combined-20260916 当时的头），所有 token-AR
  的模型、binding、算法、preset、测试在那里完整。tag 能找回历史代码，不保证未来无冲突接回。

## 逐消费者核对后的删除清单（不凭名字删）

| 项 | 消费者（删除前） | 处置 |
|---|---|---|
| `vrl/models/families/{janus_pro,llamagen,emu3,glm_image,nextstep_1}` | registry 6 个 entry | 删 |
| `vrl/generation/bindings/token_autoregressive`、`generation/composition/token_autoregressive`、`generation/steps/token` | 上述家族 | 删 |
| `vrl/models/steps/token`、`registry.TokenFamilyBuild`、`checkpoint_identity` 的 token 分支 | 上述家族 | 删 |
| `vrl/nn/modules/{ar_decoder,ar_attention_backends,torch_attention}`、`vrl/nn/layers/attention/{paged,cache_rows}`、`vrl/nn/kernels/{attention/vllm_paged,fused_linear_logprob}` | token executor、llamagen/glm runner、`vocab_head`、`replay.py:logprobs` | 删（`replay.py:logprobs` 是 categorical segment 专用，一并删） |
| `vrl/math/token` | 以上 | 删 |
| `vrl/algorithms/grpo/{token,multisegment}.py`、`vrl/rollouts/evaluators/token/` | `config/algorithm.py`、`scripts/common/factory.py` 的 token 分支 | 删 |
| `trajectory/builders.py` 三个 `build_ar_*` + `_segment_trainable`；`types.py` 的 `discrete_token/continuous_token/text_token` 轴 | 以上 | 删 |
| `sampling_schema.py` 的 AR 各节；`schema.py` 的 `token_grpo*` kind 与 `final_image_policy`；`rules.py` 的 janus/nextstep 规则；`names.py` 别名 | 以上 | 删 |
| `semantics.py` 的 `token` step kind / `token`、`multisegment_token` 布局 / `categorical` 分布 | 以上 | 删 |
| `batch_builder._pack_ar` 与 multisegment 路由、`trajectory_layout` 在 core/context 里的穿线 | 以上 | 删 |
| preset：experiment 5 目录、model 5 目录、`sampling/token/`、`base/algorithm/token_grpo*`、`recipe/online/*token_grpo*` | 以上 | 删 |
| `third_party/janus`、MODULE.bazel `janus_src`、pyproject `janus` extra、`tools/python/defs.bzl` 的依赖 | janus_pro | 删（构建提交） |
| `tests/BUILD.bazel` `gpu_vllm_tests` 的 token 目录、README 家族表、`docs/MODEL_TAXONOMY.md` 等 | 文档 | 改（构建/文档提交） |
| `vrl/models/parking.py` CuMem、`vrl/nn/quantization/fp8.py`、reward 层、checkpoint/轨迹/分布式 | diffusion 仍在用 | **保留** |
| `geneval` reward | flux 的两个 preset 在用 | **保留** |

## 提交顺序

1. 本文档（范围决定 + tag）。
2. 移除 token-AR 入口、配方、专属实现与测试，保证 import 闭合。
3. 清理因此成为死代码的共享分支（semantics、batch_builder、trajectory、replay、config 各节）。
4. 依赖、Bazel、CI 与文档。

每步跑受影响目录；结束跑全套。不做原 sprint 里"逐个重构 AR 家族"的部分。

## 结果（2026-09-16）

| 提交 | 内容 | 规模 |
|---|---|---|
| `354b218a` | 本文档 + tag `archive/token-ar-20260916` | — |
| `3e340ed2` | 删 token-AR 家族/binding/composition/steps/nn/math/算法/评估器/preset/测试；registry、`algorithm.kind`、别名表、checkpoint identity、`ReplaySegmentResult.logprobs`、online factory 收口 | 195 文件，-23991 行 |
| `82f98b49` | 共享死分支：`PolicySemantics` 只剩 `generation_regime`；轨迹 AR builder / token 轴 / categorical 分布；`batch_builder._pack_ar` 与 `trajectory_layout` 穿线；`train_segments`；AR sampling 各节、`final_image_policy`、janus/nextstep 规则、3 个 token rollout preset；`TrajectorySignalBuilder` 按 role 取 mask。用 AR builder 当夹具的测试改成两步 denoise 轨迹 | 43 文件，-1139 行 |
| 第 4 个 | pyproject（删 `janus` extra，`ar-vllm` 注释改为 CuMem/fp8 消费者）、`uv lock`、MODULE.bazel（`janus_src`、main profile 的 `janus` extra）、`tools/python/defs.bzl`、`third_party/janus`、`tests/BUILD.bazel`、README 与 4 篇活文档 | — |

保留且已核对的共享消费者：`vrl/models/parking.py` 的 CuMemAllocator、`vrl/nn/quantization/fp8.py`、`geneval` reward、chunk-AR（causvid、magi_1）。

验证：每步跑受影响目录（config/models/generation/rollouts/algorithms/trainers/scripts/trajectory/nn/architecture/math）；上游原有 3 个红测继续 deselect。历史 sprint 文档（`docs/sprints/done|parked|planned|info|reading`、`docs/research`、`RECIPE_EVIDENCE_INVENTORY`）是当时的观察记录，未改写。
