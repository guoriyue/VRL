# SPRINT: Training-side MFU — selective activation checkpointing (SAC)

状态：**P2 已落地 / 剩 P3 GPU-gated（2026-07-09 对账；原 planned 2026-06-27）**。性质：**把现在 full / 二值的 gradient checkpointing 换成 selective（只重算便宜的 norm/elementwise，保留贵的 GEMM/attention 激活），把腾出的显存花在更大的 microbatch 上 → 抬算术强度 → 抬训练 MFU。** 这是 [[SPRINT_training_mfu_compile]] 的姊妹杠杆：compile 靠融合腾显存，本 sprint 靠「只重算便宜的」摘掉 recompute 税，两者都服务同一个目标——**让 microbatch 顶到 MFU 最优点**。

> **对账（2026-07-09，对 main 实况）**：P0 probe 与 P1 selective×compile 均已实测（§4）；
> **P2 已在 main 落地**——`vrl/trainers/activation_checkpointing.py` 就是三值 policy
> `off | full | selective`（bool 向后兼容，module docstring 引用本 sprint P0 数据）。
> 剩余 = P3（按 probe 重定 microbatch、大卡 MFU 确认）+ 验收 3 的端到端 samples/s 净胜——都需要 GPU。

> 证据：`vrl/trainers/activation_checkpointing.py`（5 家族共用的唯一 ckpt 入口，从 online.py 提到训练层；online.py re-export 保持家族 import 不变）、`vrl/trainers/core/types.py:323`（`gradient_checkpointing: bool = False`）、`vrl/config/schema.py:515`、`vrl/scripts/perf/backward_mfu_probe.py`、diffusers `enable_gradient_checkpointing(gradient_checkpointing_func=...)`（`modeling_utils.py:283`）、torch 2.11 `torch.utils.checkpoint.create_selective_checkpoint_contexts` / `CheckpointPolicy`。
> 相关：[[project_rollout_bound_class_probe]]（SD3.5 train fwd+bwd ~60-65% MFU、绑激活、grad-ckpt 26.8→9.8GB）、[[feedback_mfu_bound_north_star]]（probe-first，先定 bound class 再背书）、[[project_torch_compile_wan]]（compile+ckpt 冲突的已知坑）。

## 0. 一句话 + 为什么这不是「小显存优化」

**先纠正一个会带偏整件事的框架**：full checkpointing 是小显存生存税（每个 block 全重算，~30-40% recompute 开销，**压低** MFU）；本 sprint 要做的 selective 是**相反方向**的东西——它是让「大卡团队」能把 batch 顶得**更大**的工具，从而抬 MFU。证据是 Megatron 在 **1T 参数**（地球上最奢侈的 GPU 预算）也把 selective 设成默认、靠它从 42%→56% MFU。所以面向好 GPU / 真实多卡团队，正确的处置不是「关掉 checkpointing」，而是「把 full 换成 selective + 顶大 batch」。

抬训练 MFU 的最优策略就是「把 microbatch 开到最大、抬算术强度」。可一旦你这么做，激活显存（随 `batch x seq^2` 涨）立刻又成为绑定约束——**不管卡多大**。selective 把 recompute 税从 ~30-40% 降到 ~3%，又能继续往上顶 batch，正好接住这堵墙。

## 1. 现状（坐实，全是 full / 二值）

- **唯一入口**：5 个扩散家族（sd3_5 / flux / cosmos / wan_2_1 / qwen_image）的 `train.py` 全部调 `enable_transformer_gradient_checkpointing(bundle, cfg)`，实现在 `vrl/trainers/activation_checkpointing.py`（与 fsdp.py/precision.py 并列的训练-setup 模块；online.py re-export，家族 import 路径不变）。改这一个函数 = 改所有家族。
- **当前语义是二值**：`online.py:248` `if not bool(enabled): return`，开启就调 diffusers 的 `module.enable_gradient_checkpointing()`（无参）→ 默认 `torch.utils.checkpoint.checkpoint(block, use_reentrant=False)`，**每个 transformer block 全包重算**。这就是 full。
- **默认 OFF**：`TrainerConfig.gradient_checkpointing: bool = field(default=False)`（`types.py:323`）。user-facing key 是 `actor.gradient_checkpointing`（`schema.py:515` `Any = None` → 缺省回落 dataclass 默认）。
- **probe 只有两态**：`backward_mfu_probe.py:88` `ckpt_modes = (False, True)`，没有 selective。

结论：今天只有「全开 / 全关」两挡，中间的 selective 挡位不存在。

## 2. 注入点（原生，不 fork diffusers）

diffusers 的 `ModelMixin.enable_gradient_checkpointing` **接受自定义函数**（`modeling_utils.py:283`）：

```python
def enable_gradient_checkpointing(self, gradient_checkpointing_func: Callable | None = None): ...
```

SD3 transformer 在 `transformer_sd3.py:306` 用 `self._gradient_checkpointing_func(block, ...)` 逐 block 调它。所以 selective 的实现就是传一个用 **SAC（Selective Activation Checkpointing）** 包装的函数（torch 2.11 已验证可用）：

```python
# 落在 vrl/trainers/activation_checkpointing.py（训练-setup 模块，与 fsdp.py/precision.py 并列）
import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

# 保存「贵且 recompute 划不来」的算子输出（GEMM + attention），
# 其余便宜的 pointwise / norm / dropout 一律重算。这是 Megatron selective 的原生 PyTorch 等价物。
_SAVE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
}

def _sac_policy(ctx, op, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE if op in _SAVE_OPS else CheckpointPolicy.PREFER_RECOMPUTE

def selective_checkpoint_func(module, *args):
    return checkpoint(
        module.__call__, *args,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(_sac_policy),
    )
```

为什么这一组 op 是对的：H100/5090 上 matmul 峰值 ~989/232 TFLOPS，而 special-function（norm/exp/softmax 的逐元素部分）只有它的零头——重算 norm 几乎免费，重算 GEMM 是纯浪费。SAC 让 backward 直接读回保存的 GEMM/attention 输出，只补算便宜的那部分。

**和 compile 的关系**（回应 [[project_torch_compile_wan]] 的坑）：full checkpointing + `torch.compile` 会重编译/冲突，所以现有 probe 在 `--compile` 时强制 `ckpt=False`。SAC 走 AOTAutograd 的 min-cut 路径，**理论上**和 compile 共存得更好，但这是要 probe 实测的假设，不是承诺（见 P1）。

## 3. 北极星纪律：probe 先行，不先背书

按 [[feedback_mfu_bound_north_star]]：在目标好 GPU 上，先用 probe 证明「selective + 更大 batch」在**同等显存**下的 samples/s 和 MFU 真的胜过 full，再落配置。绝不先改 recipe 再说。

判据不是「MFU 数字好看」，而是**固定质量下的 samples/s**：selective 让 batch 更大 → 每步算得更满，但每步也更大，要的是 throughput 净赢。

## 4. 阶段

### P0 — 扩 probe：加 selective 第三挡（一次性验证产物）
- 改 `vrl/scripts/perf/backward_mfu_probe.py`：把 `ckpt_modes = (False, True)` 扩成 `(off, full, selective)` 三态，selective 用第 2 节的 `selective_checkpoint_func` 经 `tf.enable_gradient_checkpointing(gradient_checkpointing_func=...)` 注入。
- 在**目标大卡**（不只在 5090；至少补一张 H100/40-80GB 级别）上跑 `--batches 1 2 4 8 16`，每挡记 `fwd ms / bwd ms / fwd+bwd MFU / peak GB`。
- 产出一张表回答两个问题：(a) selective 的 recompute 税是不是 ~3% 量级（vs full 的 ~30-40%）？(b) 在 full 撞 OOM 的 batch，selective 能不能继续往上顶、且 MFU 单调上升直到压满？
- 这是 one-shot 验证产物（`*_probe`），答案记进本 sprint 即可，**不进 import graph**。

#### P0 实测结果（SD3.5-medium 1024² / RTX 5090，2026-06-27，measured bf16 peak=231 TFLOPS）

```
batch |      ckpt |  fwd ms |  bwd ms | bwd/fwd | fwd+bwd MFU | peak GB
    1 |       off |    89.8 |   176.6 |    1.97 |        111% |   15.78
    1 |      full |    87.2 |   258.7 |    2.97 |         86% |    9.31
    1 | selective |    83.2 |   263.2 |    3.16 |         85% |    9.43
    2 |       off |   160.7 |   338.7 |    2.11 |        118% |   26.90
    2 |      full |   163.5 |   495.5 |    3.03 |         90% |    9.85
    2 | selective |   160.8 |   382.9 |    2.38 |        109% |   13.50
    4 |       off |   OOM (>32GB)
    4 |      full |   308.3 |  1005.6 |    3.26 |         90% |   11.02
    4 | selective |   309.9 |   770.6 |    2.49 |        110% |   22.30
```

**结论：selective 在 batch≥2 完胜 full，验证了 sprint 的核心假设。**
- **recompute 税**（用 bwd/fwd 比和 MFU 读）：full 在 batch2 把 MFU 从 off 的 118% 压到 90%（≈24% 吞吐税）；selective 只压到 109%（≈8% 税）——**selective 收回了约 2/3 的 recompute 税**。bwd/fwd 同向：off 2.11 → full 3.03 → selective 2.38。
- **money shot（batch4）**：off 直接 OOM（>32GB）；full 90% MFU @ 11GB；**selective 110% MFU @ 22.3GB**。selective 在 off 跑不动的 batch 上拿到了**比 full 任何 batch 都高**的 MFU，且 5090 就能放下。这正是「selective 让你顶到更大 batch → 抬 MFU」的直接证据。
- **batch1 selective≈full（85% vs 86%、显存也几乎一样）**：seq=4429 很长，attention 激活在 batch1 就占满，selective 省不下东西，收益随 batch 增长才显现。所以 selective 的价值场景是「中大 batch」，不是 batch1。
- **caveat**：MFU>100% 是标定假象（解析 FLOP 计数 vs 实测 231 TFLOPS 峰值，attn 项可能略高估），但三态同口径，**相对比较成立**，不影响结论。
- **5090 已给出正信号；H100/80GB 那一半（batch 能往上顶多高、MFU 在更大 batch 是否继续爬）仍需大卡补测**——这是 P0 唯一没在 5090 上回答的部分。

### P1 — selective x compile 共存实测 → **负结论（2026-06-27，已测）**
probe 放开 `--compile` 下测 selective 后，实测：

```
batch |      ckpt |  fwd+bwd MFU | peak GB
    1 |       off |        134%  |  12.42   <- compile 把 eager off 的 111% 抬到 134%
    1 | selective | InternalTorchDynamoError
    2 |       off |        143%  |  20.05
    2 | selective | InternalTorchDynamoError
    4 |       off | OOM
    4 | selective | InternalTorchDynamoError
```

**compiled × selective 不能共存**：`torch.compile` 追踪 `torch.utils.checkpoint(..., context_fn=create_selective_checkpoint_contexts(...))` 抛 `InternalTorchDynamoError`，三个 batch 全挂——是编译追踪失败，不是 OOM、不是偶发。结论：**两个 MFU 杠杆（compile / 手写 SAC）今天不叠加**，每个 recipe 二选一：
- 能放下就 **compiled + off**（图像侧最高 MFU：134-143%，比 eager off 还高），这也是 [[SPRINT_training_mfu_compile]] 的主线。
- 放不下（compiled off OOM，如 batch4）就 **eager + selective**（顶到更大 batch，MFU 110% 仍远高于 eager full 的 90%）。
- **关键认知**：`torch.compile` 的 AOTAutograd min-cut partitioner **本身就在做自动 selective recompute**（自己决定存/算以最小化显存+时间）。所以 compiled recipe 不需要手写 SAC——inductor 已经在干这件事。手写 SAC 是 **eager 路径**的杠杆。
- 生产含义：现有 compile-on recipe（sd3_5 ocr/pickscore/geneval）本就跑 ckpt=off，和这个约束一致。**不要在同一 recipe 同时设 `model.torch_compile.enable=true` 和 `gradient_checkpointing=selective`**——会在运行时撞同样的 dynamo 错误。P2 的 policy 不阻止这个组合（helper 不感知 compile 态），靠 recipe 纪律 + 本条文档约束。

### P2 — 把二值开关升级成 policy（落地，仅当 P0 证明胜出）
- `activation_checkpointing.py` 的 helper：把 `bool(enabled)` 解读升级为三值 `off | full | selective`（保留 bool 向后兼容：`true→full`、`false→off`）。`selective` 时传 `gradient_checkpointing_func=selective_checkpoint_func`。
- `types.py:323`：`gradient_checkpointing` 从 `bool` 改为接受 `bool | str`（默认仍 `False=off`，遵循 [[feedback_explicit_field_spelling]] 显式 `field(...)` 写法）。`schema.py:515` 注释补一行合法值。
- helper 里对 `selective_checkpoint_func` 的 op 集合按家族校验：wan/cosmos 用的 attention backend 若不是 `aten._scaled_dot_product_*`（比如自定义 FA kernel），policy 的 MUST_SAVE 命中不到，要么补 op，要么对该家族回落 full 并 warn——**不要静默退化成 full 还报 selective**。

### P3 — 按 probe 结果重定 microbatch（兑现 MFU）
- 对 selective 实测能放下更大 batch 的 recipe，按 [[SPRINT_training_mfu_compile]] §0 同样口径，用实测峰值显存重新评估 `microbatch_size` / `gradient_accumulation_steps`，把腾出的显存换成更大的有效 batch。
- 多卡：selective 减少重算 = 减少要和 FSDP all-gather 重叠的计算量，对 comm-overlap 友好；但 FSDP recipe 仍受 `strategy.py` 对 compile 的硬门约束，selective 本身和 FSDP 正交可共存（Megatron/PyTorch 都这么做），单独验证。

## 5. 验收（finishing criteria）
1. P0 表：在目标大卡上，**selective recompute 税 <= ~5%**（vs full 的 ~30-40%），且 selective 能在 full-OOM 的 batch 继续跑。否则：selective 在本硬件无意义，sprint 收敛为「记录负结论」并停。
2. selective 路径数值 parity：同 seed 下 `selective` vs `full` vs `off` 的 loss / grad-norm 在数值噪声内（SAC 不改数学，只改存/算）。复用 `compile_benchmark.py` 的 parity 口径。
3. 至少一个真实 recipe（建议 sd3_5 fullparam）端到端：`selective + 更大 microbatch` 的 **samples/s 净胜** full，MFU 上升，质量曲线不退。
4. 二值→policy 升级后，老配置（`gradient_checkpointing: true/false`）行为不变（回归测：true 仍走 full）。

## 6. 非目标
- 不引入 op-level 自定义 CUDA kernel（那是 fused-AdaLN 那条线，[[SPRINT_cosmos_video_mfu_kernels]] / 视频侧）。
- 不动 FSDP+compile 的硬门（`strategy.py`）；selective 与之正交。
- 不为「省显存让小卡能跑」做任何额外工作——本 sprint 的唯一目标是大卡上的 MFU / samples/s。
- 不碰 RL 正确性路径：SAC 是纯 autograd 存/算重排，不改 `old_log_prob`，与 fp8/cache 那类会污染 ratio 的优化**无关**（[[project_lossless_diffusion_rl_research]]）。
