# SPRINT: torch.compile 全家族默认态审计 + eager-vs-compiled parity 实测

状态：info / measured（2026-06-17，RTX 5090 32GB）。结论可直接落地：**Cosmos
Predict2.5 RL 用 compile-both（`model.torch_compile.enable=true`）而非 rollout-only**；
其余家族的默认态是"测过的开、没跑过的关",非 bug，缺口已逐个记账。

> 起因：问"是不是所有模型默认都在开 compile、这是不是我们要的"。于是 (1) 审计了全部
> 11 个模型 config 的 compile 默认态，(2) 用新加的 `compile_benchmark.py --parity` 实测了
> eager-vs-compiled 的数值漂移。本 doc 是这两件事的测量归档，配套
> `SPRINT_compile_rollout_lifecycle.md`（生命周期 + §4.3 加速 + §4.4 parity）与
> `SPRINT_cosmos_generation_compile_bottleneck.md`（生成侧 1.68×）。

## 0. TL;DR

- **不是"全部默认开"。** compile 默认态按 checkpoint 异构（§1）：测过的 DiT 默认开，没跑过
  parity proof 的默认关，AR 家族**根本没接** compile。这是"保守地正确",不是已经全拍板。
- **parity 实测（§4）：fp32 前向 eager-vs-compiled 漂 = epsilon（2–4e-6）→ compile 不改数学，
  rollout-logprob 结构性安全。** bf16 漂 ~3–5e-2 是 kernel 重排的 bf16 舍入，**只在"一侧编、
  另一侧不编"时暴露**。
- **关键结论（纠正 rollout-only 直觉）：predict2.5 应 compile-both。** rollout 与 train 用同一张
  compiled transformer → 逐位一致 → compile 引入的 logprob drift = 0，且白拿 train 1.33×。
  **rollout-only（编 rollout、train eager）恰恰是 bf16 下两侧最不一致的配置。**
- **三个缺口逐个记账（§5）**：anima = 验证过（真赢+安全），只差一个跑它的 recipe；wan i2v /
  wan2.2 = compile 安全但 **base run 本身没跑**，gate 在 base run 不在 compile；predict2.5 train
  侧 = 这次的主交付，改 compile-both。

## 1. 全家族 compile 默认态（审计，已逐文件核实）

compile 有两个开关：`model.torch_compile`（model 块，**train + rollout 都吃**，rollout 继承整个
model 块）和 `rollout.denoise_compile`（**rollout-only、单向覆盖**，只能额外开、`enable:false` 是
no-op，且被 `capability.supports_torch_compile` 门控）。base 默认态：

| checkpoint | train | rollout | 来源 / 判定 |
|---|---|---|---|
| sd3.5 medium | ✅ ON | ✅ ON（继承） | `medium.yaml` `torch_compile.enable:true`；launch-bound DiT，正确 |
| cosmos predict2 2b | ✅ ON | ✅ ON（继承） | `predict2_2b.yaml:21` true；已测 1.37×/1.25×（#11） |
| wan 2.1 1.3b / 14b | ✅ ON | ✅ ON（继承） | 均 true；t2v 已验证 |
| **cosmos predict2.5 2b** | ❌ OFF | ⚠️ 仅 kling 实验 ON | base off；kling 实验 rollout-only → 本 doc 改 compile-both |
| cosmos anima_preview3 | ❌ OFF | ❌ OFF | 同 cosmos DiT，验证过但**无 recipe 在跑**（§5） |
| wan 2.1 i2v_14b | ❌ OFF | ❌ OFF | base GRPO run 未跑（多卡卡着）（§5） |
| wan 2.2 a14b / i2v_a14b | ❌ OFF | ❌ OFF | dual-stage（`transformer_2`）proof run 未跑（§5） |
| AR janus_pro / nextstep_1 | **未接线** | **未接线** | 无 `torch_compile` 块；`capabilities.py` gate=False；KV-cache 自回归路径，正确 |

> AR 的 gate 不是疏漏：`launcher.py` 会在 AR family 误开 `denoise_compile` 时直接 raise。

## 2. 加速实测（compile off→on，mode=default，fullgraph=False）

`compile_benchmark.py`（config-init 合成 DiT，无需 checkpoint；compile 效果是结构性的，与权重值
无关）。grid 受单卡 32G 限：cosmos 用 production 深度，wan 用 12 层。

| family | path | 步延迟 | launches/step | speedup |
|---|---|---|---|---|
| cosmos-predict2.5 | rollout | 61.7→47.1ms | 2763→947 | **1.31×** |
| cosmos-predict2.5 | train | 240→180ms | 10098→3838 | **1.33×** |
| wan_2_1 | rollout | 125.8→110.6ms | 818→296 | **1.14×** |
| wan_2_1 | train | 470→428ms | 3026→1532 | **1.10×** |

机制 = kernel launch 锐减（inductor 融 elementwise epilogue），显存几乎不变。cosmos 比 wan 赢得多
（更 launch-bound）。与 `SPRINT_compile_rollout_lifecycle.md` §4.3 / 生成侧 1.68× 一致。

## 3. parity 实测（eager vs compiled，max|Δ|；`compile_benchmark.py --parity`，本轮新增）

同权重、同输入，先 eager 再 compiled（compile 包同一张 module → 任何差异都是 kernel 融合/规约顺序，
不是权重差异）。fp32 判"compile 是否改变数学"，bf16 给生产精度漂移量级。

| family | dtype | rollout 前向 | train 前向 | train 梯度 |
|---|---|---|---|---|
| cosmos-predict2.5 | **fp32** | **2.2e-6** | 2.1e-6 | 4.7e-2 (570 params) |
| cosmos-predict2.5 | bf16 | 4.7e-2 | 4.7e-2 | 量级随深度爆，abs 不可比 |
| wan_2_1 | **fp32** | **3.6e-6** | 4.1e-6 | 8.4e-9 (231 params) |
| wan_2_1 | bf16 | 3.1e-2 | 2.7e-2 | 6.1e-5 |

**读法：**
1. **fp32 前向 = epsilon（2–4e-6）→ compile 不改前向数学。** rollout logprob 由前向算出，所以
   rollout-logprob parity 是结构性安全的。
2. **bf16 前向漂 ~3–5e-2 = bf16 舍入（kernel 重排），且只在"一侧编、另一侧不编"时暴露。** 生产里
   rollout 与 train 用同一张 compiled transformer → 两侧走完全相同 kernel → 逐位一致 → compile 引入的
   rollout/train logprob drift = 0。compile-neither 同样安全；**只编一侧最危险。**
3. train 也实测了 forward+backward 穿过 compiled graph + grad-ckpt **不 graph-break**（这正是
   LoRA+grad-ckpt 那个担心点），mode=default 全程没崩。

> 边界：合成 probe 证明的是"compiled transformer 对 eager 数值忠实"（必要条件）；完整 RL-loop 的
> logprob drift ≤ 0.01（真 checkpoint + reward）仍要现场跑（lifecycle sprint P3）。

## 4. 给 predict2.5 的结论：compile-both 不是 rollout-only

| 配置 | train | rollout | bf16 两侧一致性 | train 1.33× |
|---|---|---|---|---|
| rollout-only（旧 kling 配置）| eager | compiled | ⚠️ 差 ~5e-2（一侧编） | ❌ 没拿到 |
| **compile-both（本 doc）** | compiled | compiled | ✅ 逐位一致 | ✅ 拿到 |
| compile-neither | eager | eager | ✅ 一致 | — |

→ **compile-both 严格优于 rollout-only**：parity 更稳 + 多拿 train 1.33×。落地见 §6。最后一关是现场
logprob drift ≤ 0.01（P3）；若真漂，回退 compile-neither（不要回 rollout-only）。

## 5. 三个缺口的 disposition（已写进各 config 注释）

- **anima_preview3** — 走 cosmos DiT compile 路径，**实测 1.31×/1.33× + fp32 epsilon = 真赢且安全**；
  OFF 的唯一原因是**没有任何 recipe 在跑 anima**。有 recipe 即可翻 ON。
- **wan i2v_14b** — compile 安全（fp32 epsilon）+ ~1.1× 小赢，但 **gate 是 i2v base GRPO run 还没跑
  （多卡卡着）**，不是 compile。base run 落地再翻。
- **wan 2.2 (a14b / i2v_a14b)** — dual-stage（`transformer_2`）的 base proof run 都没跑，且合成网格不
  覆盖双 transformer（`SPRINT_wan_2_2_dual_expert`），无法负责任地验证 compile。base run 后再说。
- **predict2.5 train 侧** — 本轮主交付，改 compile-both（§4/§6）。

## 6. 落地动作（本轮）

- **`compile_benchmark.py` 加 `--parity` 模式** — eager-vs-compiled 前向/梯度数值对比（CPU-offload
  eager-grad 快照以适配重网格）。已提交 `dff4c82`。
- **kling 训练 yaml 翻 compile-both** — `online_nft_kling_video_reward.yaml` 把 rollout-only 的
  `rollout.denoise_compile` 换成 `model.torch_compile.enable: true`（train+rollout 都编）。cross_node
  变体 `defaults`-继承本文件，自动跟随，无需单独改。
- **4 个 gap config 写 defer 注释** — i2v_14b / wan_2_2 a14b+i2v / anima：记录"已验证安全 + 为什么先
  OFF + 何时翻"。
- **`SPRINT_compile_rollout_lifecycle.md`** — 加 §4.4 parity 表，§5 改为 compile-both 优先，P3 标
  "合成已做、现场待跑"。

## 关键文件引用

- `vrl/scripts/perf/compile_benchmark.py` — `--family` 加速 A/B + `--parity` 数值对比（本 doc 数据源）
- `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml` — compile-both 落地
- `configs/model/diffusion/{cosmos/anima_preview3,wan_2_1/i2v_14b,wan_2_2/a14b,wan_2_2/i2v_a14b}.yaml` — defer 注释
- `vrl/generation/ray/launcher.py` `_apply_rollout_compile_override` — rollout-only 单向覆盖 + capability gate
- `vrl/models/interfaces/runtime.py` `RuntimeBuildSpec.torch_compile` — model 块 → worker（rollout 继承的来源）
- `vrl/models/diffusion/{base,cosmos/predict2_5/model,wan_2_1/model}.py` `torch_compile_transformer`
- 配套：`docs/sprints/planned/SPRINT_compile_rollout_lifecycle.md`（§4.3 加速 / §4.4 parity / P3）、
  `docs/sprints/info/SPRINT_cosmos_generation_compile_bottleneck.md`（生成侧 1.68×）
