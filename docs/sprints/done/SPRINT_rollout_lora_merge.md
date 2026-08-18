# SPRINT：rollout 侧 merge LoRA → dense —— 已落地（opt-in）

状态：**done（2026-08-17）**。KILL-RISK 门通过（这是本 program 三项里唯一过门的），
机制 + pass + worker 重折全部落地，默认关。
基线 main @ `abb8e4da`。实施提交：`d97d4069`（机制）、`59101209`（pass 接线）。

执行中有**三处证据推翻了计划里的假设**，已按证据修正（见 §4）。

## 0. 结论先行

`SPRINT_gemm_utilization.md` 杠杆表里唯一「列了但没写」的一条，现在写了。

实测（wan 形状：12 blocks / d=1536 / rank 32 / 注意力投影 / bf16 / eval+no_grad，
交替 A/B 取中位数）：

| arm | seq 1024 | seq 4096 |
|---|---|---|
| eager | 8.30 → 7.14 ms（**14.0%**） | 28.54 → 26.95 ms（**5.6%**） |
| compiled | 8.07 → 7.11 ms（**11.9%**） | 28.36 → 27.01 ms（**4.7%**） |

走真实 pass 的端到端复核（seq 2048）：15.00 → 13.53 ms，**9.8%**。

**收益扛得住 compile**，这是最关键的一条：inductor 能融 scaling，但**消不掉那两个
额外 GEMM**。所以这不是已经默认开的 compile pass 顺手拿走的东西。

## 1. 为什么有效

rollout 每个 chunk 把 policy 重放 `steps × CFG` 次。adapter 活跃时每个被 target 的
投影是 3 个 GEMM（base、`lora_A`、`lora_B`）+ 一次 scaling 乘；折叠后是 1 个。
这正是 `SPRINT_cross_model_performance.md` §0 记的「~70% elementwise 来自 LoRA
plumbing（280 个 PEFT 层）」。

相对 P1.5（`use_lora:false` 转全参）的好处：**拿到全参的 GEMM 形状，不付全参的
显存**——训练侧仍然只存 LoRA 的优化器状态。

## 2. 落地形态

- `vrl/nn/optimization/lora_merge.py` —— 机制。按 `peft.tuners.lora.LoraLayer`
  **走树**而不是问模型，因为本仓两个 adapter 表面接法不同（`get_peft_model`
  包整棵树；diffusers `PeftAdapterMixin.add_adapter`（Wan / cosmos-predict2）不包），
  层类型是唯一同时覆盖两者的 seam —— 和 `swap_linears` 走 `nn.Linear` 同理。
- `LoraMergePass`（`vrl/nn/optimization/passes.py`）—— 排在**最前**，
  在 quantization 之前，让低精度替换看到**有效权重**。
- `model.lora.merge_for_rollout`（默认 `null`/off），identity 标 `exclude`：
  折叠是同一个 policy 的改写，不是另一个模型。
- `update_weights` 每次同步后重折（`vrl/generation/execution/worker.py`）。
- `require_every_core_merged` 查**层**不查自报，照抄
  `require_every_core_quantized` 的理由（Wan 双专家半覆盖 bug）。

## 3. 验收结果

- ✅ 数值：端到端前向 max rel diff **0.0**（bf16，本例恰好逐位相同；
  `introduces_replay_drift=True` 仍按保守声明，因为累加顺序确实变了）。
- ✅ 独立复算：折叠权重 == `pristine + (alpha/r)·B@A`，由 adapter 张量独立算出对拍。
- ✅ 不漂移：200 轮随机 adapter 重折后，折入零 adapter 得到与建库快照**逐位相同**的基权重。
- ✅ 覆盖完整：多核半折叠被 `require_every_core_merged` 拒绝。
- ✅ 权重同步：重折后前向 == 持同一 adapter 的未折叠模型；并有反向对照测试证明
  「不重折就会服务旧权重」是可观测的（防止上一条空过）。
- ✅ 全量 CPU gate：**4126 passed**。剩余 2 个
  `tests/architecture/test_generation_rollout_boundaries.py` 失败**在 `origin/main`
  上预存在**（stash 验证过），与本 sprint 无关。
- ⬜ **未做**：真实模型的 rollout-vs-replay logprob parity 红线（需要真实
  checkpoint + 一次真实 run）。**开启此开关前必须先过这一关**，见 §5。

## 4. 计划被推翻的三处

**① 「colocated 必须排除」——错。**
计划里写 colocated 共用同一个 model 对象，所以 merge 会破坏训练梯度和
`disable_adapter()`。读代码后发现不成立：rollout 走
`build_denoise_runtime_bundle`，replay 走 `assemble_replay_bundle`，
**永远是两个实例**——权重同步存在本身就是证据（共享对象就不需要同步）。
colocated 共享的是 GPU（靠 parking 交接），不是模型对象。
所以**不需要 colocated 特例**，`enabled()` 只看 `rollout is not None`。

**② 「用 PEFT 的 merge/unmerge 往返」——不可行。**
计划假设 merge/unmerge 可以反复做。实测 bf16 下相对误差：1 轮 4.8e-3，
1000 轮 **2.4e-1**——基权重在一次正常 run 里就被毁掉。
（fp32 无此问题：1000 轮 2.6e-6。）
改为：建库时快照一份 pristine 基权重（CPU 常驻），每次折叠都是
`pristine + delta`，**永不累加**。CPU 常驻是因为 rollout worker 的显存才是稀缺项，
而这份快照每次权重同步读一次，不是每个 denoise step 读一次。

**③ 发现了计划里没有的冲突：versioned trainable-state slots。**
非 draining 同步会同时保留多个 adapter 版本，让 in-flight 请求各自激活自己那版。
折叠后**没有 adapter 可切**——版本就是基权重本身，而基权重只有一份。
`conflicts()` 看不到这个（`versioned_weight_sync` 在 launch contract 上，不在
`ModelBuild` 上），所以拒绝落在 worker 里两者都可见的地方。
`conflicts()` 保持返回 `()` 并在 docstring 说明原因 ——
**不能触发的 guard 比没有 guard 更糟**（`CompilePass` 同款理由）。

## 5. 开启前的前置（留给下一个人）

`model.lora.merge_for_rollout: true` **默认关，不要直接在生产打开**。开之前：

1. 真实模型上跑 rollout-vs-replay logprob parity，**均差 ≤ 0.01**
   （`trainer.py` 的红线）。过不了就停，把数字记在这里，**不放宽阈值**。
2. 确认该 run 没开 versioned weight sync（否则 worker 会直接报错，这是设计行为）。
3. 收益随序列长度下降（seq 1024 约 12%，seq 4096 约 5%），
   低分辨率 / 短序列的家族收益最大。

## 6. 相关

- 杠杆出处：`docs/sprints/done/SPRINT_gemm_utilization.md:195`
- elementwise 归因：`docs/sprints/info/SPRINT_cross_model_performance.md` §0
- 挂载 seam：`docs/sprints/SPRINT_plug_and_play_optimization_layer.md`
- 半覆盖反模式：`vrl/nn/optimization/passes.py` 的 `require_every_core_quantized`
- 父 program：`docs/sprints/done/SPRINT_train_phase_efficiency_program.md`
